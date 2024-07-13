import torch
import tqdm
from torch.utils.data import Dataset
import numpy as np
import io
import time
import lmdb
import math
import random
import logging
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from rgnet.span_utils import span_xx_to_cxw
from scipy.stats import norm
import utils.misc as util

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "query_id": "ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b_0",
      "query": "what did I put in the black dustbin?",
      "video_id": "38737402-19bd-4689-9e74-3af391b15feb",
      "clip_id": "93231c7e-1cf4-4a20-b1f8-9cc9428915b2",
      "timestamps": [425.0, 431.0],
      "duration": 480,
    }
    """
    def __init__(self, dset_name, data_path, motion_feat_dir, appearance_feat_dir, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=20, max_v_l=90, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True, query_id2windowidx=None,
                 topk_window=30, clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0, is_eval=False,
                 ret_eval=False, comb_ret_eval=False, winret=False, online_loader=False, input_fps_reduction=1):
        self.online_loader=online_loader
        self.winret=winret
        self.comb_ret_eval=comb_ret_eval
        self.ret_eval = ret_eval
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.motion_feat_dir = motion_feat_dir
        self.appearance_feat_dir = appearance_feat_dir
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l

        #TODO: check window length
        self.max_v_l = max_v_l
        self.clip_len = clip_len
        self.input_fps_reduction = input_fps_reduction
        self.ctx_mode = ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        # a hyperparameter to control the top-k window pre-filtering
        self.topk_window = topk_window
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0
        # define the window stride as the half size of the window length
        self.slide_window_size = int(max_v_l / 2)
        self.eval = is_eval
        timer_start = time.time()

        # use lmdb to load visual and textual feature
        ###
        # An explanation of what are two visual feat sources
        # We name visual_appearance_feat (appearance_visual_txn) as the visual input of coarse-grained window selection and fine-grained ranking modules
        # We name visual_motion_feat (motion_visual_txn) as the visual input of Moment-DETR module
        # In current version, the appearance_visual and motion_visual features are the same (e.g., CLIP features for MAD dataset, EgoVLP features for Ego4D-NLQ dataset). However, we do note that those two features can be different.
        ###
        # Moreover, we speculate that two kind of feature inputs leads to two complementary ranking scores.
        # 1. Adapted fine-grained matching score computed by visual_appearance_feat captures appearance contents (objects, scene)
        # 2. Moment-DETR proposals score computed by visual_motion_feat captures motion-related contents (actions), because it conduct self-attention among each feature inside the window
        ###
        if self.online_loader:
            self.appearance_visual_env = None
            self.appearance_visual_txn = None
            self.textual_env = None
            self.textual_txn = None
        else:
            self._init_db()

        self.same_visual_path = self.motion_feat_dir == self.appearance_feat_dir
        if not self.same_visual_path:
            self.motion_visual_env = lmdb.open(self.motion_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                               readahead=False)
            self.motion_visual_txn = self.motion_visual_env.begin(buffers=True)

        print("load lmdb time:", time.time() - timer_start)
        print("clip_len:", self.clip_len)
        # data
        self.data = self.load_data()
        if self.winret:
            v2q_dict = {}
            self.q2q_list = []
            for i, d in enumerate(self.data):
                if d['video_id'] in v2q_dict:
                    v2q_dict[d['video_id']].add(i)
                else:
                    v2q_dict[d['video_id']] = {i}
            self.movies = [list(a) for a in list(v2q_dict.values())]
        # the window rank-list computed by the contrastive pre-trained model
        self.query_id2windowidx = query_id2windowidx

        self.videofeat = self.load_video_feat() if not online_loader else {}
        if not self.same_visual_path:
            self.motion_videofeat = self.load_video_motion_feat()

    def _init_db(self):
        self.appearance_visual_env = lmdb.open(self.appearance_feat_dir, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_visual_env.begin(buffers=True)
        self.textual_env = lmdb.open(self.q_feat_dir, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)
        #self.load_video_feat()

    def load_video_feat(self):
        video_set = set([item['clip_id'] for item in self.data])
        video2feat = {}
        if util.is_main_process():
            iterator = tqdm.tqdm(video_set, desc="load video feat")
        else:
            iterator = video_set
        for video_id in iterator:
            video_clip_feat = self._get_video_appearance_feat_by_vid(video_id)
            video2feat[video_id] = video_clip_feat
        return video2feat

    def load_video_motion_feat(self):
        video_set = set([item['clip_id'] for item in self.data])
        video2feat = {}
        for video_id in tqdm.tqdm(video_set, desc="load video feat"):
            video_clip_feat = self._get_video_motion_feat_by_vid(video_id)
            video2feat[video_id] = video_clip_feat
        return video2feat

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # delay loading lmdb data until after initialization
        if self.appearance_visual_env is None:
            self._init_db()
        _meta = self.data[index]
        query_id = _meta["query_id"]
        model_inputs = []
        model_clip_inputs = []
        query_feat, query_cls_feat = self._get_query_feat_by_qid(_meta["query_id"])

        assert self.use_video
        if self.online_loader and _meta["clip_id"] not in self.videofeat:
            self.videofeat[_meta["clip_id"]] = self._get_video_appearance_feat_by_vid(_meta["clip_id"])
        video_clip_feat = self.videofeat[_meta["clip_id"]]
        if self.same_visual_path:
            video_motion_feat = video_clip_feat
        else:
            video_motion_feat = self.motion_videofeat[_meta["clip_id"]]
            #video_motion_feat = self._get_video_motion_feat_by_vid(_meta["clip_id"])
        video_clip_feat = video_clip_feat[::self.input_fps_reduction]
        video_motion_feat = video_motion_feat[::self.input_fps_reduction]
        ctx_l = len(video_clip_feat)
        assert ctx_l > 0, ctx_l
        num_window = math.ceil(ctx_l / self.slide_window_size) + 1

        if self.eval:
            # select top-k windows for inference
            if self.ret_eval and self.comb_ret_eval:
                windowidx = self.query_id2windowidx[query_id]
            elif self.ret_eval:
                windowidx = list(range(num_window))
            else:
                windowidx = self.query_id2windowidx[query_id][:self.topk_window]
            for i in windowidx:
                new_start = max((i - 1) * self.slide_window_size, 0)
                new_end = min((i - 1) * self.slide_window_size + self.max_v_l, ctx_l)
                tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
                tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
                model_inputs.append({
                    'video_length': new_end - new_start,
                    'video_start': new_start,
                    'video_motion_feat': tmp_video_motion_feat,
                    'query_feat': query_feat})
                model_clip_inputs.append({
                    'video_appear_feat': tmp_video_appearance_feat,
                    'query_cls_feat': query_cls_feat, })
        else:
            # calculate the positive window list for training
            start = _meta["timestamps"][0] / self.clip_len
            end = _meta["timestamps"][1] / self.clip_len
            #assert start < end, (end, start, _meta)
            if start > end: # inserted for NaQ
                temp = start
                start = end
                end = temp
            start = min(ctx_l, start)
            end = min(ctx_l, end)
            pos_window_id_list = range(math.floor(start / self.slide_window_size),
                                       math.ceil(end / self.slide_window_size) + 1)
            assert len(pos_window_id_list), (_meta, ctx_l, _meta["timestamps"], pos_window_id_list)
            neg_window_pool = list(set(range(num_window)) - set(pos_window_id_list))
            assert len(neg_window_pool), (_meta, ctx_l, _meta["timestamps"], pos_window_id_list)

            ####
            # There are at least two positive windows for each query, we choose a strategy to choose more
            # middle-position window because it is more likely to cover the whole duration rather than partial coverage.
            ####
            pos_window_id_list = np.array(pos_window_id_list)
            temp_number = pos_window_id_list - pos_window_id_list.mean()
            temp_weight = norm.pdf(temp_number)
            weight = temp_weight / np.sum(temp_weight)
            idx = np.random.choice(pos_window_id_list, p=weight)

            new_start = max((idx - 1) * self.slide_window_size, 0)
            new_end = min((idx - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            tmp_video_motion_feat = video_motion_feat[new_start:new_end, :]
            tmp_video_appearance_feat = video_clip_feat[new_start:new_end, :]
            tmp_model_inputs = {
                'video_length': new_end - new_start,
                'video_start': new_start,
                'video_motion_feat': tmp_video_motion_feat,
                'query_feat': query_feat}
            tmp_model_clip_inputs = {
                'video_appear_feat': tmp_video_appearance_feat,
                'query_cls_feat': query_cls_feat, }

            # span_proposal ground-truth
            start_pos = max((idx - 1) * self.slide_window_size, start) - tmp_model_inputs["video_start"]
            end_pos = min((idx - 1) * self.slide_window_size + self.max_v_l, end) - tmp_model_inputs["video_start"]
            tmp_span_labels = self.get_span_labels([[start_pos, end_pos]], tmp_model_inputs['video_length'])
            tmp_model_inputs.update({'span_labels': tmp_span_labels})
            assert 0 <= math.floor(start_pos) < math.ceil(end_pos), [start, end, idx, tmp_model_inputs["video_start"],
                                                                     start_pos, end_pos, _meta]
            tmp_model_clip_inputs.update(
                {'span_proposal': torch.IntTensor([[math.floor(start_pos), math.ceil(end_pos)]])})

            # Choose one positive saliency frame inside the ground-truth
            # and one negative saliency frame out of ground-truth
            rel_clip_ids = list(range(math.floor(start_pos), math.ceil(end_pos)))
            if not len(rel_clip_ids):
                rel_clip_ids = [math.floor(start_pos)]
            easy_neg_pool = list(set(range(tmp_model_inputs['video_length'])) - set(rel_clip_ids))
            if not len(easy_neg_pool):
                easy_neg_pool = [0]
            tmp_model_inputs.update({"saliency_pos_labels": random.sample(rel_clip_ids, k=1)})
            tmp_model_inputs.update({"saliency_neg_labels": random.sample(easy_neg_pool, k=1)})

            # Randomly choose one negative window
            neg_window_id = random.choice(neg_window_pool)
            #neg_index = random.choice(list(self.q2q_list[index]))
            neg_start = max((neg_window_id - 1) * self.slide_window_size, 0)
            neg_end = min((neg_window_id - 1) * self.slide_window_size + self.max_v_l, ctx_l)
            tmp_model_inputs.update(
                {"neg_window_motion_feat": video_motion_feat[neg_start:neg_end, :]})
            tmp_model_clip_inputs.update(
                {"neg_window_appear_feat": video_clip_feat[neg_start:neg_end, :]})

            model_clip_inputs.append(tmp_model_clip_inputs)
            model_inputs.append(tmp_model_inputs)

        meta = []
        for idx in range(len(model_inputs)):
            item = _meta.copy()
            item['duration'] = model_inputs[idx]['video_length']
            item['video_start'] = model_inputs[idx]['video_start']
            meta.append(item)

        return dict(meta=meta, model_inputs=model_inputs, model_clip_inputs=model_clip_inputs)

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        """
        qid: query_id
        returns both textual token feature and holistic text feature for each query
        """
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            q_feat = q_dump['token_features']
            try:
                cls_q_feat = q_dump['cls_features']
            except:
                cls_q_feat = q_dump['eot_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]

        if self.q_feat_type == "last_hidden_state":
            q_feat = q_feat[:self.max_q_l]

        if self.normalize_t:
            q_feat = l2_normalize_np_array(q_feat)

        cls_q_feat = l2_normalize_np_array(cls_q_feat)

        return torch.from_numpy(q_feat), cls_q_feat  # (Lq, D), (D, )

    def _get_video_motion_feat_by_vid(self, vid):
        dump = self.motion_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features'].astype(np.float32)

        if self.normalize_v:
            _v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(_v_feat)  # (Lv, D)

    def _get_video_appearance_feat_by_vid(self, vid):
        dump = self.appearance_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features']

        if self.normalize_v:
            _v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    batch_meta = [item for e in batch for item in e["meta"]]  # seems no need to collate ?
    for e in batch:
        if len(e["model_inputs"]):
            model_inputs_keys = e["model_inputs"][0].keys()
            break
    for e in batch:
        if len(e["model_clip_inputs"]):
            model_clip_inputs_keys = e["model_clip_inputs"][0].keys()
            break

    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=item["span_labels"]) for e in batch for item in e['model_inputs']]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([item[k] for e in batch for item in e['model_inputs']])
            continue
        if k in ["video_start", "video_length"]:
            batched_data[k] = torch.IntTensor([item[k] for e in batch for item in e['model_inputs']])
            continue

        seq = [item[k] for e in batch for item in e['model_inputs']]
        batched_data[k] = pad_sequences_1d(
            seq, dtype=torch.float32, fixed_length=None)

    batched_clip_data = dict()
    for k in model_clip_inputs_keys:
        if k in ["span_proposal"]:
            batched_clip_data[k] = [dict(proposal=item["span_proposal"]) for e in batch for item in
                                    e['model_clip_inputs']]
            continue
        if k in ["query_cls_feat"]:
            #batched_clip_data[k] = torch.FloatTensor([item[k] for e in batch for item in e['model_clip_inputs']])
            batched_clip_data[k] = torch.from_numpy(np.array([item[k] for e in batch for item in e['model_clip_inputs']]))
            continue
        seq = [item[k] for e in batch for item in e['model_clip_inputs']]
        batched_clip_data[k] = pad_sequences_1d(
            seq, dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data, batched_clip_data


def prepare_movie_inputs(batched_model_inputs, batched_clip_model_inputs, opt, non_blocking=False):
    device = opt.device
    bsz = opt.bsz
    pos_model_inputs = dict(
        src_txt=torch.repeat_interleave(batched_model_inputs["query_feat"][0], bsz, dim=0).to(device, non_blocking=non_blocking),
        src_txt_mask=torch.repeat_interleave(batched_model_inputs["query_feat"][1], bsz, dim=0).to(device, non_blocking=non_blocking),
        src_vid_motion=batched_model_inputs["video_motion_feat"][0].repeat([bsz,1,1]).to(device, non_blocking=non_blocking),
        src_vid_motion_mask=batched_model_inputs["video_motion_feat"][1].repeat([bsz,1]).to(device, non_blocking=non_blocking),
    )
    pos_clip_model_inputs = dict(
        src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
        src_vid_appear=batched_clip_model_inputs["video_appear_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_appear_mask=batched_clip_model_inputs["video_appear_feat"][1].to(device, non_blocking=non_blocking),
    )
    if "neg_window_motion_feat" in batched_model_inputs:
        neg_model_inputs = dict(
            src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            src_vid_motion=batched_model_inputs["neg_window_motion_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_motion_mask=batched_model_inputs["neg_window_motion_feat"][1].to(device,
                                                                                     non_blocking=non_blocking),
        )
        neg_clip_model_inputs = dict(
            src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
            src_vid_appear=batched_clip_model_inputs["neg_window_appear_feat"][0].to(device,
                                                                                     non_blocking=non_blocking),
            src_vid_appear_mask=batched_clip_model_inputs["neg_window_appear_feat"][1].to(device,
                                                                                          non_blocking=non_blocking),
        )
    else:
        neg_model_inputs = None
        neg_clip_model_inputs = None

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
    if "span_proposal" in batched_clip_model_inputs:
        targets["span_proposal"] = [
            dict(proposal=e["proposal"].to(device, non_blocking=non_blocking))
            for e in batched_clip_model_inputs["span_proposal"]
        ]

    targets = None if len(targets) == 0 else targets
    return pos_model_inputs, pos_clip_model_inputs, (neg_model_inputs, neg_clip_model_inputs), targets

def prepare_batch_inputs(batched_model_inputs, batched_clip_model_inputs, device, non_blocking=False):
    pos_model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid_motion=batched_model_inputs["video_motion_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_motion_mask=batched_model_inputs["video_motion_feat"][1].to(device, non_blocking=non_blocking),
    )
    pos_clip_model_inputs = dict(
        src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
        src_vid_appear=batched_clip_model_inputs["video_appear_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_appear_mask=batched_clip_model_inputs["video_appear_feat"][1].to(device, non_blocking=non_blocking),
    )
    if "neg_window_motion_feat" in batched_model_inputs:
        neg_model_inputs = dict(
            src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
            src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
            src_vid_motion=batched_model_inputs["neg_window_motion_feat"][0].to(device, non_blocking=non_blocking),
            src_vid_motion_mask=batched_model_inputs["neg_window_motion_feat"][1].to(device,
                                                                                     non_blocking=non_blocking),
        )
        neg_clip_model_inputs = dict(
            src_cls_txt=batched_clip_model_inputs["query_cls_feat"].to(device, non_blocking=non_blocking),
            src_vid_appear=batched_clip_model_inputs["neg_window_appear_feat"][0].to(device,
                                                                                     non_blocking=non_blocking),
            src_vid_appear_mask=batched_clip_model_inputs["neg_window_appear_feat"][1].to(device,
                                                                                          non_blocking=non_blocking),
        )
    else:
        neg_model_inputs = None
        neg_clip_model_inputs = None

    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)
    if "span_proposal" in batched_clip_model_inputs:
        targets["span_proposal"] = [
            dict(proposal=e["proposal"].to(device, non_blocking=non_blocking))
            for e in batched_clip_model_inputs["span_proposal"]
        ]

    targets = None if len(targets) == 0 else targets
    return pos_model_inputs, pos_clip_model_inputs, (neg_model_inputs, neg_clip_model_inputs), targets


class PreFilteringDataset(Dataset):
    """One line in data loaded from data_path."
    {
      "query_id": "ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b_0",
      "query": "what did I put in the black dustbin?",
      "video_id": "38737402-19bd-4689-9e74-3af391b15feb",
      "clip_id": "93231c7e-1cf4-4a20-b1f8-9cc9428915b2",
      "timestamps": [425.0, 431.0],
      "duration": 480,
    }
    """

    def __init__(self, dset_name, data_path, appearance_feat_dir, q_feat_dir,
                 ctx_mode="video", data_mode="context", data_ratio=1,input_fps_reduction=1):
        self.dset_name = dset_name
        self.data_ratio = data_ratio
        self.data_path = data_path
        self.appearance_feat_dir = appearance_feat_dir
        self.q_feat_dir = q_feat_dir
        self.data_mode = data_mode
        self.ctx_mode = ctx_mode
        self.use_video = "video" in ctx_mode
        self.input_fps_reduction=input_fps_reduction

        timer_start = time.time()
        self.appearance_feat_dir = lmdb.open(self.appearance_feat_dir, readonly=True, create=False,
                                             max_readers=4096 * 8, readahead=False)
        self.appearance_visual_txn = self.appearance_feat_dir.begin(buffers=True)
        self.textual_env = lmdb.open(q_feat_dir, readonly=True, create=False, max_readers=4096 * 8,
                                     readahead=False)
        self.textual_txn = self.textual_env.begin(buffers=True)
        print("load lmdb time:", time.time() - timer_start)
        # data
        self.query_data = self.load_data()
        self.video_data = list(set([item["clip_id"] for item in self.query_data]))
        self.video2idx = {v: idx for idx, v in enumerate(self.video_data)}

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def set_data_mode(self, data_mode):
        """context or query"""
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        else:
            return len(self.query_data)

    def _get_video_appearance_feat_by_vid(self, vid):
        dump = self.appearance_visual_txn.get(vid.encode())
        with io.BytesIO(dump) as reader:
            img_dump = np.load(reader, allow_pickle=True)
            v_feat = img_dump['features']

        v_feat = l2_normalize_np_array(v_feat)
        return torch.from_numpy(v_feat)  # (Lv, D)

    def _get_query_feat_by_qid(self, qid):
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            try:
                cls_q_feat = q_dump['cls_features']
            except:
                cls_q_feat = q_dump['eot_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]
        cls_q_feat = l2_normalize_np_array(cls_q_feat)
        return cls_q_feat  # (D, )

    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        else:
            return self._get_item_query(index)

    def _get_item_query(self, index):
        """Need to batch"""
        raw_data = self.query_data[index]

        meta = dict(
            query_id=raw_data["query_id"],
            query=raw_data["query"],
            video_id=raw_data["clip_id"]
        )

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta['query_id'])
        return dict(meta=meta, model_inputs=model_inputs)

    def _get_item_context(self, index):
        """No need to batch, since it has already been batched here"""
        video_id = self.video_data[index]

        # initialize with basic data
        meta = dict(
            video_id=video_id,
        )

        model_inputs = dict()
        model_inputs["video_feat"]= self._get_video_appearance_feat_by_vid(meta['video_id'])[::self.input_fps_reduction]
        return dict(meta=meta, model_inputs=model_inputs)
