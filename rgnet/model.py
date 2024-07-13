# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from rgnet.span_utils import generalized_temporal_iou, span_cxw_to_xx
from rgnet.matcher import build_matcher
from rgnet.transformer import build_transformer
from rgnet.position_encoding import build_position_encoding
from rgnet.misc import accuracy

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class RGNet(nn.Module):
    """ This is the RGNet model that performs moment localization in the long-form video. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_motion_dim, vid_appear_dim,
                 num_queries, input_dropout, aux_loss=False, max_v_l=75, span_loss_type="l1", use_txt_pos=False,
                 n_input_proj=2, adapter_module="linear", decoder_gating=False, qddetr=False, winret_neg_loss=False,
                 samp_loc_loss=False, dabdetr=False, gumbel=False,multiscale=False,topk=10):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_motion_dim: int, video visual motion feature input dimension
            vid_appear_dim: int, video visual appearance feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            adapter_module: str, one of [linear, none]
                linear: additional FFN adpter module
        """
        super().__init__()
        self.topk=topk
        self.gumbel=gumbel
        self.samp_loc_loss=samp_loc_loss
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.winret_neg_loss = winret_neg_loss
        self.class_embed = nn.Linear(2 * hidden_dim if winret_neg_loss else hidden_dim, 2)  # 0: background, 1: foreground

        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.dabdetr = dabdetr
        if self.dabdetr:
            self.query_embed = nn.Embedding(num_queries, 2)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[
                                                 LinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
                                                 LinearLayer(vid_motion_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])

        self.saliency_proj = nn.Linear(hidden_dim, 1)
        self.aux_loss = aux_loss

        self.adapter_module = adapter_module
        if self.adapter_module == "linear":
            self.adapter_layer = MLP(vid_appear_dim, hidden_dim, vid_appear_dim, 2)
        self.decoder_gating = decoder_gating
        self.qddetr = qddetr
        self.multiscale=multiscale
        if qddetr:
            self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
            self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
            self.saliency_proj3 = nn.Linear(hidden_dim, 1)
            self.hidden_dim = hidden_dim
            self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
            self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))


    def forward(self, src_txt, src_txt_mask, src_vid_motion, src_vid_motion_mask):
        """
        The forward expects two tensors:
           - src_txt: [batch_size, L_txt, D_txt]
           - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer
           - src_vid_motion: [batch_size, L_vid, D_vid]
           - src_vid_motion_mask: [batch_size, L_vid], containing 0 on padded pixels,
                will convert to 1 as padding later for transformer

        It returns a dict with the following elements:
           - "pred_spans": The normalized boxes coordinates for all queries, represented as
                           (center_x, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid_motion)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_motion_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_motion_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        if self.qddetr:
            # for global token
            mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
            mask = torch.cat([mask_, mask], dim=1)
            src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
            src = torch.cat([src_, src], dim=1)
            pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
            pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]
        hs: object
        hs, memory, prob_soft, memory_global, attn_weights, pred_prop_hard, reference = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length)
        if self.winret_neg_loss:
            outputs_class = self.class_embed(torch.cat([src_[None].repeat(6,1,5,1), hs],-1))
        else:
            outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        tmp = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.dabdetr:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coord = tmp + reference_before_sigmoid
        else:
            outputs_coord = tmp
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1],
               'prob_soft': self.saliency_proj3(memory_global) if self.qddetr else prob_soft}
        #if self.samp_loc_loss:
        out['attn_weights'] = attn_weights
        out['pred_prop_hard'] = pred_prop_hard
        #txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        if self.qddetr:
            out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) *
                        self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        else:
            out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1)  # (bsz, L_vid)

        out["video_mask"] = src_vid_motion_mask
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return out

    def forward_clip_matching(self, src_cls_txt, src_vid_appear, src_vid_appear_mask, proposal=None,
                              is_groundtruth=False):
        """
        The forward expects following tensors:
            - src_cls_txt: [batch_size, D_txt]
            - src_vid_appear: [batch_size, L_vid, D_vid]
            - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
            - proposal:
            - is_groundtruth: whether the proposal comes from the ground-truth (during training)
            or proposal generation prediction (during inference).
        It returns a proposal-query similarity matrix.
        """
        text_cls_features = src_cls_txt / src_cls_txt.norm(dim=1, keepdim=True)

        if is_groundtruth:
            tgt_proposals = torch.vstack([t["proposal"][0] for t in proposal])  # (#spans, 2)
            proposal_score = self._get_groundtruth_proposal_feat(src_vid_appear, tgt_proposals, text_cls_features)
            #proposal_features = proposal_feat / proposal_feat.norm(dim=1, keepdim=True)
            #return torch.einsum('bd,ad->ba', proposal_features, text_cls_features)
            return proposal_score
        else:
            proposal_score = self._get_predicted_proposal_feat(src_vid_appear, src_vid_appear_mask, proposal, text_cls_features)
            #proposal_features = proposal_feat / proposal_feat.norm(dim=2, keepdim=True)
            #return torch.einsum('bld,bd->bl', proposal_features, text_cls_features)
            return proposal_score

    #TODO: RGNet
    def _get_groundtruth_proposal_feat(self, src_vid_appear, groundtruth_proposal, text_cls_features):
        """
        The forward expects following tensors:
           - src_vid_appear: [batch_size, L_vid, D_vid]
           - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
           - proposal: [batch_size, 2], ground-truth start and end timestamps
       It returns proposal features for ground-truth moments.
        """
        proposal_feat_list = []
        for idx, (feat, start_end_list) in enumerate(zip(src_vid_appear, groundtruth_proposal)):
            clip_feat = feat[start_end_list[0]:start_end_list[1]]
            if self.adapter_module == "linear":
                clip_feat = self.adapter_layer(clip_feat) + clip_feat
            clip_feat = clip_feat / clip_feat.norm(dim=1, keepdim=True)
            #score = torch.einsum('ld,bd->lb', clip_feat, text_cls_features)
            #proposal_feat_list.append(torch.topk(score, min(clip_feat.shape[0], 3), 0)[0].mean(0))
            clip_feat = self._topk_pooling(text_cls_features, clip_feat[None], min(clip_feat.shape[0], 3))[0]
            #clip_feat = _attention_pooling(text_cls_features, clip_feat[None], 0.01)[0]
            score = torch.einsum('ld,ld->l', clip_feat, text_cls_features)
            proposal_feat_list.append(score)
        proposal_feat = torch.vstack(proposal_feat_list)
        return proposal_feat

    #TODO: RGNet
    def _get_predicted_proposal_feat(self, src_vid_appear, src_vid_appear_mask, pred_proposal, text_cls_features):
        """
        The forward expects following tensors:
          - src_vid_appear: [batch_size, L_vid, D_vid]
          - src_vid_appear_mask: [batch_size, L_vid], containing 0 on padded pixels
          - proposal: [batch_size, N_query, 2], predicted start and end timestamps for each moment queries
        It returns proposal features for predicted proposals.
        """
        vid_appear_dim = src_vid_appear.shape[2]
        duration = torch.sum(src_vid_appear_mask, dim=-1)
        proposal = torch.einsum('bld,b->bld', span_cxw_to_xx(pred_proposal), duration)  # .to(torch.int32)
        bsz, n_query = proposal.shape[:2]
        proposal_start = F.relu(torch.floor(proposal[:, :, 0]).to(torch.int32))
        proposal_end = torch.ceil(proposal[:, :, 1]).to(torch.int32)
        proposal_feat_list = []
        for idx, (feat, start_list, end_list) in enumerate(zip(src_vid_appear, proposal_start, proposal_end)):
            for start, end in zip(start_list, end_list):
                clip_feat = feat[start:end]
                if self.adapter_module == "linear":
                    clip_feat = self.adapter_layer(clip_feat) + clip_feat
                clip_feat = clip_feat / clip_feat.norm(dim=1, keepdim=True)
                clip_feat = self._topk_pooling(text_cls_features[idx][None], clip_feat[None], min(clip_feat.shape[0], 3))[0]
                #clip_feat = _attention_pooling(text_cls_features[idx][None], clip_feat[None], 0.01)[0]
                score = torch.einsum('ld,d->l', clip_feat, text_cls_features[idx])
                #proposal_feat_list.append(torch.topk(score, min(clip_feat.shape[0], 3), 0)[0].mean(0))
                proposal_feat_list.append(score)
        proposal_feat = torch.vstack(proposal_feat_list)
        proposal_feat = proposal_feat.reshape(bsz, n_query)#, vid_appear_dim)
        return proposal_feat

    def _topk_pooling(self, text_embeds, video_embeds, k):
        """
        Pooling top-k frames for each video based on
        similarities with each text query

        Output
            video_embeds_pooled: num_vids x num_texts x embed_dim
        """
        num_texts, embed_dim = text_embeds.shape

        # num_vids x num_frames x num_texts
        sims = video_embeds @ text_embeds.t()
        sims_topk = torch.topk(sims, k, dim=1)[1]

        # Make format compatible with torch.gather
        video_embeds = video_embeds.unsqueeze(-1).expand(-1, -1, -1, num_texts)
        sims_topk = sims_topk.unsqueeze(2).expand(-1, -1, embed_dim, -1)

        # num_vids x k x embed_dim x num_texts
        video_embeds_topk = torch.gather(video_embeds, dim=1, index=sims_topk)

        # Top-k pooling => num_vids x embed_dim x num_texts
        video_embeds_pooled = video_embeds_topk.sum(dim=1)
        return video_embeds_pooled.permute(0, 2, 1)

    def _attention_pooling(self, text_embeds, video_embeds, temperature):
        """
        Pooling frames for each video using attention-based
        similarity with each text query

        Output
            video_embeds_pooled: num_vids x num_texts x embed_dim
        """
        # num_vids x num_frames x num_texts
        sims = video_embeds @ text_embeds.t()
        attention_weights = F.softmax(sims / temperature, dim=1)

        # num_vids x embed_dim x num_frames
        video_embeds = video_embeds.permute(0, 2, 1)

        # num_vids x embed_dim x num_texts
        video_embeds_pooled = torch.bmm(video_embeds, attention_weights)
        return video_embeds_pooled.permute(0, 2, 1)



class SetCriterion(nn.Module):
    """ This class computes the loss of RGNet modified from DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model inside the positive window
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, qddetr=False, winret_neg_loss=False,samp_loc_loss=False):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        self.qddetr= qddetr
        self.winret_neg_loss=winret_neg_loss
        self.samp_loc_loss=samp_loc_loss

    def loss_retrieval(self, outputs, targets, indices=None, neg_outputs=None, n_outputs=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        #assert 'prob_soft' in outputs
        if outputs['prob_soft'] is None:
            return  {'loss_retrieval':torch.as_tensor(0.).to('cuda')}
        #assert neg_outputs is not None
        if n_outputs is None:
            pos_logits = outputs['prob_soft']  # (batch_size, #queries, #classes=2)
            neg_logits = neg_outputs['prob_soft']  # (batch_size, #queries, #classes=2)
            #TODO: weight negative classes less
            loss_ce = torch.clamp(self.saliency_margin + neg_logits - pos_logits, min=0).sum() \
                                / (len(pos_logits)) * 2  # * 2 to keep the loss the same scale
            losses = {'loss_retrieval': loss_ce}
        else:
            pos_logits = outputs['prob_soft']  # (batch_size, #queries, #classes=2)
            neg_logits = n_outputs['prob_soft']
            bsz = pos_logits.shape[0]
            logits = torch.diagonal_scatter(torch.zeros([bsz,bsz]).to(pos_logits.device), pos_logits.flatten())
            idx = [[i for i in range(bsz) if i!=j] for j in range(bsz)]
            idx = torch.tensor(idx).to(torch.int64).to(neg_logits.device)
            logits = logits.scatter_(1, idx, neg_logits.reshape([bsz,-1]))
            loss_ret = self.loss_adapter({"logits_per_video": logits})
            losses = {'loss_retrieval': loss_ret['loss_adapter']}
        return losses

    def loss_samp_loc(self, outputs, targets, indices=None, neg_outputs=None, n_outputs=None, log=True):
        span = torch.stack([span_cxw_to_xx(t['spans']).squeeze() for t in targets['span_labels']]) * self.max_v_l
        bsz = span.shape[0]
        span[:, 0] = span[:, 0].floor()
        span[:, 1] = span[:, 1].ceil()
        span = span.to(int)
        attn_weights = outputs['attn_weights']
        target_span = torch.zeros_like(attn_weights)
        for i in range(bsz):
            target_span[:,i, span[i,0]:span[i,1]] = 1
        sample_loc_loss = F.binary_cross_entropy(attn_weights.flatten(), target_span.flatten().to(attn_weights))
        return {'sample_loc_loss': sample_loc_loss}

    #TODO: RGNet
    def loss_adapter(self, pos_outputs):
        ######
        # additional adapter NCE loss, followed by CLIP implementation
        #####
        assert 'logits_per_video' in pos_outputs

        logits_per_video = pos_outputs["logits_per_video"] / self.temperature
        bsz = len(logits_per_video)
        diagonal_indices = torch.arange(bsz).to(logits_per_video.device)

        criterion = nn.CrossEntropyLoss(reduction="mean")
        loss_per_video = criterion(logits_per_video, diagonal_indices)
        loss_per_text = criterion(logits_per_video.T, diagonal_indices)
        loss = (loss_per_video + loss_per_text) / 2
        return {'loss_adapter': loss}

    def loss_spans(self, outputs, targets, indices, neg_outputs=None, n_outputs=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        ######
        # no modification
        #####
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {'loss_span': loss_span.mean(), 'loss_giou': loss_giou.mean()}
        return losses

    # TODO: RGNet
    def loss_labels(self, outputs, targets, indices=None, neg_outputs=None, n_outputs=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        bsz = src_logits.shape[0]
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch

        ######
        # additional contrastive classification loss to distinguish positive and negative window via proposal-level comparison
        # summary: assign all proposal label in the negative window is the background label
        ######
        if neg_outputs is not None:
            neg_src_logits = neg_outputs['pred_logits']  # (batch_size, #queries, #classes=2)
            src_logits = torch.cat((src_logits, neg_src_logits), dim=1)
        elif n_outputs is not None and self.winret_neg_loss:
            # choose moments from one of the negative windowsf
            neg_src_logits = n_outputs['pred_logits'][[i*(bsz-1) for i in range(bsz)],...]  # (batch_size, #queries, #classes=2)
            src_logits = torch.cat((src_logits, neg_src_logits), dim=1)
        #####

        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        if indices is not None:
            idx = self._get_src_permutation_idx(indices)
            target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if indices is not None and log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    # TODO: RGNet
    def loss_saliency(self, outputs, targets, indices, neg_outputs=None, n_outputs=None):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        ######
        # original saliency loss for Moment-DETR
        # summary: saliency score of random frame inside the ground-truth  is larger than that outside of ground-truth
        ######
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or motion_window_80
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale


        # additional contrastive saliency loss to distinguish positive and negative window via saliency-level comparison
        # summary: saliency score of random frame inside the ground-truth  is larger than that the maximum saliency score of the negative window
        ######
        if neg_outputs is not None:
            neg_saliency_scores = neg_outputs["saliency_scores"]  # (N, L)
            neg_saliency_max_scores, _ = torch.max(neg_saliency_scores, 1)
            neg_window_max_scores = torch.stack(
                [neg_saliency_max_scores for _ in range(num_pairs)], dim=1)
            loss_neg_saliency = torch.clamp(self.saliency_margin + neg_window_max_scores - pos_scores, min=0).sum() \
                                / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            loss_saliency += loss_neg_saliency
            if self.qddetr:
                # Neg pair loss
                loss_saliency += self.qddetr_neg_loss(neg_outputs, outputs)
                # conntrastive loss
                #loss_saliency += self.qddetr_cont_loss(neg_saliency_scores, outputs, saliency_scores)

        return {"loss_saliency": loss_saliency}

    def qddetr_cont_loss(self, neg_saliency_scores, outputs, saliency_scores):
        vid_token_mask = outputs["video_mask"]
        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores_all = torch.cat([saliency_scores.clone(), neg_saliency_scores.clone()], dim=1)
        pos_target = torch.full(saliency_scores.shape, self.foreground_label, dtype=torch.float32,
                                device=saliency_scores.device)
        neg_target = torch.full(neg_saliency_scores.shape, self.background_label, dtype=torch.float32,
                                device=neg_saliency_scores.device)  # (batch_size, #queries)
        target_classes = torch.cat((pos_target, neg_target), dim=1)
        # numerical stability
        logits = saliency_scores_all - torch.max(saliency_scores_all, dim=1, keepdim=True)[0]
        # softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        pos_mask = target_classes > 0  # negative sample indicator
        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0
        mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
        loss = - mean_log_prob_pos * batch_drop_mask
        return loss.mean()

    def qddetr_neg_loss(self, neg_outputs, outputs):
        vid_token_mask = neg_outputs["video_mask"]
        # Neg pair loss
        saliency_scores_neg = neg_outputs["saliency_scores"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()
        return loss_neg_pair

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, neg_outputs, n_outputs, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "retrieval": self.loss_retrieval,
            "samp_loc_loss": self.loss_samp_loc
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, neg_outputs, n_outputs, **kwargs)

    def forward(self, outputs, targets, neg_outputs=None, n_outputs=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts for positive window, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             neg_outputs: list of dicts for negative window, such that len(neg_outputs) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        if targets is None:
            losses = {}
            losses.update(self.loss_labels(outputs, targets, None))
            return losses

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, neg_outputs, n_outputs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ["saliency", "retrieval","samp_loc_loss"]:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, neg_outputs, n_outputs, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


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


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)


    position_embedding, txt_position_embedding = build_position_encoding(args)
    transformer = build_transformer(args, position_embedding)
    model = RGNet(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_motion_dim=args.v_motion_feat_dim,
        vid_appear_dim=args.v_appear_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        span_loss_type=args.span_loss_type,
        adapter_module=args.adapter_module,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        decoder_gating=args.decoder_gating,
        qddetr=args.qddetr,
        winret_neg_loss=args.winret_neg_loss,
        samp_loc_loss=args.samp_loc_loss,
        dabdetr=args.dabdetr,
        gumbel=args.gumbel,
        multiscale=args.multiscale,
        topk=args.topk
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.adapter_loss:
        weight_dict["loss_adapter"] = args.adapter_loss_coef
    if args.decoder_gating or args.qddetr:
        weight_dict["loss_retrieval"] = args.retrieval_loss_coef
    if args.samp_loc_loss:
        weight_dict["sample_loc_loss"] = args.samp_loc_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if (args.decoder_gating or args.qddetr) and args.retrieval_loss_coef!=0:
        losses.append('retrieval')
    if args.samp_loc_loss:
        losses.append('samp_loc_loss')
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,qddetr=args.qddetr,
        winret_neg_loss=args.winret_neg_loss,samp_loc_loss=args.samp_loc_loss
    )
    criterion.to(device)
    return model, criterion
