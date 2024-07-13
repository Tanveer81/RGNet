# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .dab_attention import MultiheadAttention
from rgnet.gumble_softmax import GumbelSoftmax


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=6, num_decoder_layers_2=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,decoder_gating=False,
                 qddetr=False,query_dim=2,keep_query_pos=True, query_scale_type='cond_elewise',
                 num_patterns=0,modulate_t_attn=True,bbox_embed_diff_each_layer=True,dabdetr=False,gumbel=False,gumbel_2=False,
                 gumbel_3=False,multiscale=False,gumbel_eps=0.66667,position_embedding=None,gumbel_single_proj=False):
        super().__init__()
        self.gumbel=gumbel
        self.gumbel_2 = gumbel_2
        self.gumbel_3 = gumbel_3
        self.multiscale=multiscale
        self.qddetr= qddetr
        self.dabdetr=dabdetr
        self.gumbel_single_proj=gumbel_single_proj
        if qddetr:
            t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_encoder_layers, encoder_norm)
        if self.gumbel:
            self.prop_instance = nn.Linear(d_model, 1) #todo also test with MLP(hidden_dim, hidden_dim, 1, 3)
            #self.threshold = nn.Parameter(torch.tensor(0.5))
        if self.gumbel_3 and not self.gumbel_single_proj:
            self.prop_instance2 = nn.Linear(d_model, 1)
        if self.gumbel or self.gumbel_3:
            self.gumble_gate = GumbelSoftmax(eps=gumbel_eps)
        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # TransformerDecoderLayerThin
        #decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        if self.dabdetr:
            decoder_layer = DabTransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation,
                                                       normalize_before,keep_query_pos=keep_query_pos)
            self.decoder = DabTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec,
                                              d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
                                              query_scale_type=query_scale_type,
                                              modulate_t_attn=modulate_t_attn,
                                              bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                    normalize_before)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.decoder_gating=decoder_gating
        if decoder_gating:
            #self.mask_c = Mask_c()
            decoder_layer2 = TransformerDecoderLayer(d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
            decoder_norm2 = nn.LayerNorm(d_model)
            self.decoder2 = TransformerDecoder(decoder_layer2, num_decoder_layers_2, decoder_norm2,return_intermediate=return_intermediate_dec)
        if self.multiscale:
            num_feature_levels = 4
            self.backbone = BackbonePool(position_embedding, num_feature_levels)
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, video_length):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        if self.dabdetr:
            refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)
            tgt = torch.zeros(refpoint_embed.shape[0], bs, d).cuda()
        else:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)
            tgt = torch.zeros_like(query_embed)
        pred_prop_hard = None
        if self.qddetr:
            src_text, mask_text, pos_embed_text = src[video_length + 1:], mask[:, video_length + 1:], pos_embed[video_length + 1:]
            # window to text cross attention
            src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed,video_length=video_length)  # (L, batch_size, d)
            src = src[:video_length + 1]
            mask = mask[:, :video_length + 1]
            pos_embed = pos_embed[:video_length + 1]
            # [CLS] token and window self attention
            attn_mask = None
            if self.gumbel:# and not self.gumbel_3:
                attn_mask, mask_idx, pred_prop_soft, pred_prop_hard = self.method_gumbel(bs, src, video_length)

            memory, attn_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, mask=attn_mask)  # (L, batch_size, d)
            memory_global, memory_video = memory[0], memory[1:]
            mask_video = mask[:, 1:]
            pos_embed_video = pos_embed[1:]
        else:
            src_text, mask_text, pos_embed_text = src[video_length:], mask[:, video_length:], pos_embed[video_length:]
            attn_mask = None
            if self.gumbel and not self.gumbel_3:
                attn_mask, mask_idx, pred_prop_soft, pred_prop_hard = self.method_gumbel(bs, src, video_length)
            memory, attn_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, mask=attn_mask)  # (L, batch_size, d)
            memory_global = None
            memory_video = memory
            mask_video = mask
            pos_embed_video = pos_embed

        if self.gumbel_3:
            if self.gumbel_single_proj:
                pred_prop_hard = pred_prop_hard[:,1:]
                mask_idx = mask_idx[:,1:]
            else:
                pred_prop_hard, pred_prop_soft = self.gumble_gate(self.prop_instance2(memory_video))
                if pred_prop_soft is None:
                    pred_prop_soft = pred_prop_hard
                if pred_prop_soft is not None:
                    pred_prop_soft = pred_prop_soft[1:].permute(2,1,0)
                pred_prop_hard = pred_prop_hard.permute(2, 1, 0)[0]
                mask_idx = pred_prop_hard == 0
            if self.gumbel_2:
                memory_video = memory_video + pred_prop_hard.transpose(0,1)[:,:,None] * memory_global[None]
        if self.multiscale:
            ori_video = memory_video
            mask_video, memory_video, pos_embed_video, level_start_index \
                = (self.method_multiscale(mask_video, memory_video, pos_embed_video))

        references = None
        if self.decoder_gating:
            prob_soft = None
            # query to text cross-attention
            tgt = self.decoder(tgt, src_text, memory_key_padding_mask=mask_text, pos=pos_embed_text, query_pos=query_embed)
            # query to frame cross-attention
            hs = self.decoder2(tgt[-1], memory_video,memory_key_padding_mask=mask_video,pos=pos_embed_video, query_pos=query_embed)
            #hs = hs[:2]
            hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        else:
            prob_soft = None
            if self.dabdetr:
                hs, references = self.decoder(tgt, memory_video, memory_key_padding_mask=mask_video,
                                              pos=pos_embed_video, refpoints_unsigmoid=refpoint_embed)
            else:
                hs = self.decoder(tgt, memory_video,
                    memory_key_padding_mask=torch.logical_or(mask_video,mask_idx) if self.gumbel_3 else mask_video,
                    tgt_key_padding_mask=None,
                    pos=pos_embed_video, query_pos=query_embed)  # (#layers, #queries, batch_size, d)
                hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        if self.multiscale:
            memory_video = ori_video
        memory_local = memory_video.transpose(0, 1)  # (batch_size, L, d)

        return hs, memory_local, prob_soft, memory_global, pred_prop_soft if self.gumbel else attn_weights, pred_prop_hard, references

    def method_gumbel(self, bs, src, video_length):
        pred_prop_hard, pred_prop_soft = self.gumble_gate(self.prop_instance(src))
        if pred_prop_soft is None:
            pred_prop_soft = pred_prop_hard
        if pred_prop_soft is not None:
            pred_prop_soft = pred_prop_soft[1:].permute(2, 1, 0)
        pred_prop_hard = pred_prop_hard.permute(2, 1, 0)[0]
        attn_mask = torch.zeros([bs, video_length + 1, video_length + 1]).to(src)
        mask_idx = pred_prop_hard == 0
        mask_idx[:, 0] = False
        if self.gumbel_2:
            attn_mask[mask_idx] = 1
        for b in range(attn_mask.size(0)):
            if self.gumbel_2:
                attn_mask[b].fill_diagonal_(0)
            else:
                attn_mask[b, 0, mask_idx[b]] = 1
        attn_mask = attn_mask[:, :, None, :].repeat(1, self.nhead, 1, 1).view(bs * self.nhead, video_length + 1,
                                                                              video_length + 1)
        return attn_mask, mask_idx, pred_prop_soft, pred_prop_hard

    def method_multiscale(self, mask_video, memory_video, pos_embed_video):
        memory_video, pos_embed_video, mask_video = self.backbone(memory_video, mask_video, pos_embed_video)
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, pos_embed in enumerate(pos_embed_video):
            spatial_shape = (pos_embed.shape[0],)
            spatial_shapes.append(spatial_shape)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        memory_video = torch.cat(memory_video, 0)  # bs, \sum{hxw}, c
        mask_video = torch.cat(mask_video, 1)  # bs, \sum{hxw}
        pos_embed_video = torch.cat(lvl_pos_embed_flatten, 0)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory_video.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return mask_video, memory_video, pos_embed_video, level_start_index


class BackbonePool(nn.Sequential):
    def __init__(self,position_embedding,num_feature_levels):
        super().__init__()
        self.pool = nn.AvgPool1d(3, stride=3)
        self.position_embedding=position_embedding
        self.num_feature_levels=num_feature_levels
        #self.pool = nn.Conv1d(256,256,2, stride=2)

    def forward(self, x, mask, pos):
        srcs, pos_embeds, masks = [x], [pos], [mask]
        x = x.permute(1, 2, 0)
        for i in range(self.num_feature_levels-1):
            x = self.pool(x)
            m = F.interpolate(mask[None].float(), size=x.shape[2]).to(torch.bool)[0]
            p = self.position_embedding(x.permute(0,2,1), m).to(x.dtype)
            srcs.append(x.permute(2, 0, 1))
            pos_embeds.append(p.permute(1, 0, 2))
            masks.append(m)
        return srcs, pos_embeds, masks

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []
        attn_weights = []
        attn_weight = None
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if isinstance(output, tuple):
                output, attn_weight = output
            if self.return_intermediate:
                intermediate.append(output)
            if attn_weight is not None:
                attn_weights.append(attn_weight)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        if attn_weight is not None:
            return output, torch.stack(attn_weights)
        else:
            return  output


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


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        assert video_length is not None

        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]

        # print(src_key_padding_mask.shape) # torch.Size([32, 102])
        # print(src_key_padding_mask[:, 1:76].permute(1,0).shape) # torch.Size([75, 32])
        # print(src_key_padding_mask[:, 76:].shape) # torch.Size([32, 26])

        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:,
                                                                                 video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        # print(attn_mask.shape)
        # print(attn_mask[0][0])
        # print(q.shape) 75 32 256
        # print(k.shape) 26 32 256

        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])
        # print('after src shape :',src.shape)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        print('before src shape :', src.shape)
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]
        # print(q.shape) # 100 32 256

        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1, 0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        print('after src shape :', src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
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

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights[:,0,1:]

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src , attn_weights[:,0,:]

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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
        try:
            return tensor if pos is None else tensor + pos
        except:
            print()
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
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
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
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


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
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
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
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


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DabTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        # self.bbox_embed = None
        # for DAB-deter
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for i in range(num_layers)])
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # print('line230', query_sine_embed.shape)
            query_pos = self.ref_point_head(query_sine_embed)
            # print('line232',query_sine_embed.shape)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            # print(query_sine_embed.shape) # 10 32 512
            query_sine_embed = query_sine_embed * pos_transformation

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 1
                # print(reft_cond.shape, reft_cond[..., 0].shape) # 10 32 1, 10 32
                # print(obj_center.shape, obj_center[..., 1].shape) # 10 32 2, 10 32
                # print(query_sine_embed.shape) # 10 32 256

                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)


            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)

class DabTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, position_embedding):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        decoder_gating=args.decoder_gating,
        qddetr=args.qddetr,
        num_decoder_layers_2=args.dec_layers_2,
        dabdetr=args.dabdetr,
        gumbel=args.gumbel,
        gumbel_2=args.gumbel_2,
        gumbel_3=args.gumbel_3,
        gumbel_eps=args.gumbel_eps,
        multiscale=args.multiscale,
        position_embedding=position_embedding,
        gumbel_single_proj=args.gumbel_single_proj
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
