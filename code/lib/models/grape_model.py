from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn.functional as F
import copy

from .conv_block import BasicBlock, Bottleneck, AdaptBlock
from .position_encoding import PositionEmbeddingSine
from typing import Optional, List
from torch import nn, Tensor
from .backbone import get_backbone


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock_upsample(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(BasicBlock_upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=inplanes,
            out_channels=inplanes,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False)
        self.bn = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos, src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = tgt
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ADAPTIVE': AdaptBlock
}

class OurNet(nn.Module):
    def __init__(self, cfg, backbone):
        super(OurNet, self).__init__()
        self.backbone = backbone
        inp_channels = backbone.output_channels
        self.spec = cfg.MODEL.SPEC
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET

        config_offset_attention = self.spec.HEAD_OFFSET_ATTENTION
        config_visibility_attention = self.spec.HEAD_VISIBILITY_ATTENTION

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.head_offset = self._make_head_offset(config_offset)

        d_model = cfg.MODEL.DIM_MODEL
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        decoder_layers_num = cfg.MODEL.DECODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING

        # bottleneck layer
        self.bottleneck = self._make_transition_for_head(inp_channels, config_heatmap['NUM_CHANNELS'])

        self.bottleneck_encoder = nn.Sequential(
            nn.Conv2d(config_heatmap['NUM_CHANNELS'] + self.num_joints, d_model, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(d_model, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

        self.bottleneck_decoder = nn.Sequential(
            nn.Conv2d(self.num_joints * 2 + 1, d_model, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(d_model, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))

        # Attention
        self.pos_embed = PositionEmbeddingSine(d_model=d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, activation='relu')
        self.global_encoder = TransformerEncoder(encoder_layer, encoder_layers_num)

        decoder_layer = TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=0.1, activation='relu')
        self.global_decoder = TransformerDecoder(decoder_layer, decoder_layers_num)

        # deconvolution layers
        self.deconv = nn.Sequential(BasicBlock_upsample(d_model))


        self.head_offset_attention = self._make_head_offset(config_offset_attention)
        self.head_visibility_attention = self._make_visibility_head(config_visibility_attention)
        ############################################

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE'])
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_head_offset(self, layer_config):
        offset_final_layer = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        offset_final_layer.append(feature_conv)

        offset_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints * 2,
            kernel_size=self.spec['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if self.spec['FINAL_CONV_KERNEL'] == 3 else 0
        )
        offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_final_layer)

    def _make_visibility_head(self, layer_config):
        visibility_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        visibility_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints,
            kernel_size=self.spec['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if self.spec['FINAL_CONV_KERNEL'] == 3 else 0
        )
        visibility_head_layers.append(heatmap_conv)

        return nn.ModuleList(visibility_head_layers)

    def _make_center_head(self, layer_config):
        visibility_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        visibility_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=1,
            kernel_size=self.spec['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if self.spec['FINAL_CONV_KERNEL'] == 3 else 0
        )
        visibility_head_layers.append(heatmap_conv)

        return nn.ModuleList(visibility_head_layers)

    def _make_keypoints_head(self, layer_config):
        visibility_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        visibility_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints,
            kernel_size=self.spec['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if self.spec['FINAL_CONV_KERNEL'] == 3 else 0
        )
        visibility_head_layers.append(heatmap_conv)

        return nn.ModuleList(visibility_head_layers)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)


    def forward(self, x):
        bb_out = self.backbone(x)
        fi = self.bottleneck(bb_out)
        heatmap_init = self.head_heatmap[1](self.head_heatmap[0](fi))
        keypoints_map_init = heatmap_init[:, :-1, :, :]
        center_heatmap_init = heatmap_init[:, -1:, :, :]
        offset_init = self.head_offset[1](self.head_offset[0](fi))
        input_encoder = torch.cat([keypoints_map_init, fi], 1)
        input_encoder = self.bottleneck_encoder(input_encoder)
        bs, c, h, w = input_encoder.shape
        # Pos and masks create
        pos_embedding = self.pos_embed(input_encoder).permute(2, 0, 1)
        mask = input_encoder.new_ones((bs, h, w))
        mask[:, :h, :w] = 0
        mask = F.interpolate(mask[None], size=input_encoder.shape[-2:]).to(torch.bool).squeeze(0)
        mask = mask.flatten(1)
        input_encoder = input_encoder.flatten(2).permute(2, 0, 1)
        output_encoder = self.global_encoder(input_encoder, src_key_padding_mask=mask, pos=pos_embedding)
        input_decoder = torch.cat([offset_init, center_heatmap_init], 1)
        input_decoder = self.bottleneck_decoder(input_decoder)
        input_decoder = input_decoder.flatten(2).permute(2, 0, 1)
        output_decoder = self.global_decoder(input_decoder, output_encoder, memory_key_padding_mask=mask, pos=pos_embedding)
        x = output_decoder[0].permute(1, 2, 0).view(bs, c, h, w)
        x = self.deconv(x)

        offset_final = self.head_offset_attention[1](self.head_offset_attention[0](x))
        visibility = self.head_visibility_attention[1](self.head_visibility_attention[0](x))

        return heatmap_init, offset_init, offset_final, visibility, x

def get_pose_net(cfg, is_train, **kwargs):
    backbone = get_backbone(cfg, is_train, **kwargs)
    model = OurNet(cfg, backbone)
    return model
