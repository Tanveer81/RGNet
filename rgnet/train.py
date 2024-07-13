import os
import time
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rgnet.config import BaseOptions
from rgnet.ego4d_mad_dataloader import StartEndDataset, start_end_collate, prepare_batch_inputs, PreFilteringDataset, prepare_movie_inputs
from rgnet.inference import eval_epoch, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from torch.utils.data.distributed import DistributedSampler
import utils.misc as util

import logging

import torch.multiprocessing

from utils.sampler import DistributedMovieSampler, MovieSampler

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    if util.is_main_process():
        logger.info(f"[Epoch {epoch_i + 1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    if util.is_main_process():
        iterator = tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples)
    else:
        iterator = enumerate(train_loader)
    for batch_idx, batch in iterator:
        if opt.winret and batch_idx==num_training_examples-1: #TODO: hack fix for last batch
            break
        # global_step = epoch_i * num_training_examples + batch_idx
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.winret:
            pos_model_inputs, pos_clip_model_inputs, neg_model_inputs_tuple, targets \
                = prepare_movie_inputs(batch[1], batch[2], opt, non_blocking=True)
        else:
            pos_model_inputs, pos_clip_model_inputs, neg_model_inputs_tuple, targets \
                = prepare_batch_inputs(batch[1], batch[2], opt.device, non_blocking=True)

        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        outputs = model.forward(**pos_model_inputs)

        if opt.winret:
            idx = torch.LongTensor([i*i-1 for i in range(1,opt.bsz+1)]).to(outputs['attn_weights'].device)
            outputs['attn_weights'] = outputs['attn_weights'].index_select(1, idx)
            pos_outputs = {}
            n_outputs = {}
            pos_idx = [i*opt.bsz+i for i in range(opt.bsz)]
            neg_idx = [i for i in range(opt.bsz*opt.bsz) if i not in pos_idx]
            for k in outputs.keys():
                if 'aux' not in k:
                    if 'attn_weights' in k:
                        pos_outputs[k] = outputs[k]
                    else:
                        pos_outputs[k] = outputs[k][pos_idx]
                        n_outputs[k] = outputs[k][neg_idx]
                else:
                    pos_outputs[k] = []
                    for dct in outputs[k]:
                        pos_aux = {}
                        for k2 in dct.keys():
                            pos_aux[k2] = dct[k2][pos_idx]
                        pos_outputs[k].append(pos_aux)
        else:
            pos_outputs = outputs
            n_outputs = None

        if opt.neg_loss:
            neg_outputs = model.forward(**neg_model_inputs_tuple[0])
            loss_dict = criterion(pos_outputs, targets, neg_outputs, n_outputs)
        else:
            loss_dict = criterion(pos_outputs, targets, None, n_outputs)

        weight_dict = criterion.weight_dict

        losses = 0
        for k in loss_dict.keys():
            if k in weight_dict:
                losses += loss_dict[k] * weight_dict[k]

        if opt.adapter_loss and epoch_i >= opt.start_epoch_for_adapter:
            mod = model.module if opt.distributed else model
            pos_outputs["logits_per_video"] = mod.forward_clip_matching(**pos_clip_model_inputs,
                                                                 proposal=targets["span_proposal"],
                                                                 is_groundtruth=True)
            adapter_loss = criterion.loss_adapter(pos_outputs)["loss_adapter"]
            losses += adapter_loss * weight_dict["loss_adapter"]
            loss_dict['adapter_loss'] = adapter_loss

        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()

        optimizer.zero_grad()
        losses.backward(retain_graph=False)
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = losses  # for logging only
        weight_dict["loss_overall"] = 1
        # reduce losses over all GPUs for logging purposes
        if opt.distributed:
            torch.distributed.barrier()
        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        for k, v in loss_dict_reduced_scaled.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        # if opt.adapter_loss and epoch_i >= opt.start_epoch_for_adapter:
        #     k = "loss_adapter"
        #     loss_meters["loss_adapter"].update(
        #         float(adapter_loss) * weight_dict[k] if k in weight_dict else float(adapter_loss))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break
    torch.cuda.empty_cache()
    if opt.distributed:
        torch.distributed.barrier()
    if util.is_main_process():
        # print/add logs
        tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i + 1)
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i + 1)

        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i + 1,
            loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        if epoch_i%10==0:
            logger.info("Epoch time stats:")
            for name, meter in time_meters.items():
                d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
                logger.info(f"{name} ==> {d}")


def train(model, model_without_ddp, criterion, optimizer, lr_scheduler, train_dataset, eval_inter_window_dataset, eval_intra_window_dataset, opt):
    if opt.device.type == "cuda":
        if util.is_main_process():
            logger.info("CUDA enabled.")
        model.to(opt.device)
    tb_writer=None
    if util.is_main_process():
        tb_writer = SummaryWriter(opt.tensorboard_log_dir)
        tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    # if opt.winret:
    #     batch_sampler = DistributedMovieSampler(dataset=train_dataset.movies, batch_size=opt.bsz, shuffle=False) \
    #         if opt.distributed else MovieSampler(train_dataset.movies,batch_size=opt.bsz)
    #     train_loader = DataLoader(
    #         train_dataset,
    #         collate_fn=start_end_collate,
    #         num_workers=opt.num_workers,
    #         pin_memory=opt.pin_memory,
    #         sampler=None,
    #         batch_sampler=batch_sampler,
    #     )
    # else:
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) if opt.distributed else None
    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle= False,
        pin_memory=opt.pin_memory,
        sampler=train_sampler,
    )

    prev_best_score = 0.
    es_cnt = 0
    # start training
    if util.is_main_process():
        score_writer = open(
            os.path.join(opt.results_dir, "eval_results.txt"), mode="w", encoding="utf-8"
        )

    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = 0
    else:
        start_epoch = opt.start_epoch
    if opt.debug:
        start_epoch = 0
    if opt.dset_name == "mad":
        save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    else:
        save_submission_filename = "latest_{}_{}_preds.json".format(opt.dset_name, opt.eval_split_name)

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        logger.info(opt.exp_id)
        if opt.distributed:
            # if opt.winret:
            #     train_loader.batch_sampler.set_epoch(epoch_i)
            # else:
            train_loader.sampler.set_epoch(epoch_i)
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        if opt.eval_path is not None and ((epoch_i + 1) % opt.eval_epoch_interval == 0 or epoch_i==0):# and util.is_main_process():
            with torch.no_grad():
                results, mIoU, score_str, latest_file_paths, display_data = \
                    eval_epoch(model, eval_inter_window_dataset, eval_intra_window_dataset, opt,
                               save_submission_filename, epoch_i=epoch_i, criterion=None,  tb_writer=tb_writer)

            if util.is_main_process():
                for score_item in score_str:
                    score_writer.write(score_item)
                score_writer.flush()
                id_dict = ['pre_filter', 'fusion', 'proposal', 'matching']
                for id, d in enumerate(display_data):
                    for k, v in zip(d[0], d[1]):
                        key = k.replace('\n','')
                        tb_writer.add_scalar(f"val/{id_dict[id]}/{key}", float(v), epoch_i + 1)

            if opt.dset_name == "mad":
                stop_score = torch.mean(results[0])
                print("stop_score ", stop_score)
            else:
                stop_score = (results[0][0] + results[1][0]) / 2
                print("stop_score ", stop_score)

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                #torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                util.save_on_master(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                if util.is_main_process():
                    for src, tgt in zip(latest_file_paths, best_file_paths) :
                        os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    if util.is_main_process():
                        with open(opt.train_log_filepath, "a") as f:
                            f.write(f"Early Stop at epoch {epoch_i}")
                        logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score * 100}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            #torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))
            util.save_on_master(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 #if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if opt.eval_path is None:
            save_interval = 3
        #if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
        if ((epoch_i + 1) % save_interval == 0 or 0 in [(epoch_i + 1) % ld for ld in opt.lr_drop]):
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            #torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))
            util.save_on_master(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug and epoch_i==2:
            break
    if util.is_main_process():
        tb_writer.close()


def start_training():
    if util.is_main_process():
        logger.info("Setup config, data and model...")
        #logger.info(f"PID_{os.getpid()}")
    opt = BaseOptions().parse()

    # Distributed training setup
    if opt.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, opt.cuda_visible_devices))
    util.init_distributed_mode(opt)
    opt.clip_length = opt.clip_length * opt.input_fps_reduction
    if opt.single_gpu: #TODO: remove the hack
        opt.distributed = False
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True

    model, model_without_ddp, criterion, optimizer, lr_scheduler = setup_model(opt)

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        motion_feat_dir=opt.motion_feat_dir,
        appearance_feat_dir=opt.appearance_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        topk_window=opt.topk_window,
        winret=opt.winret,
        online_loader=opt.online_loader,
        input_fps_reduction=opt.input_fps_reduction
    )

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features").replace("_naq", "")  # for pretraining
        dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        dataset_config["is_eval"] = True
        dataset_config["ret_eval"] = opt.ret_eval
        dataset_config["comb_ret_eval"] = opt.comb_ret_eval
        dataset_config["input_fps_reduction"] = opt.input_fps_reduction
        #dataset_config['online_loader'] = opt.online_loader
        eval_intra_window_dataset = StartEndDataset(**dataset_config)
        pre_filtering_dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.eval_path,
            appearance_feat_dir=opt.appearance_feat_dir,
            q_feat_dir=opt.t_feat_dir.replace("_naq", "") ,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            input_fps_reduction = opt.input_fps_reduction
        )
        eval_inter_window_dataset = PreFilteringDataset(**pre_filtering_dataset_config)
    else:
        eval_intra_window_dataset = None
        eval_inter_window_dataset = None

    if util.is_main_process():
        #logger.info(f"Model {model}")
        count_parameters(model_without_ddp)
        logger.info("Start Training...")
    train(model, model_without_ddp, criterion, optimizer, lr_scheduler, train_dataset, eval_inter_window_dataset, eval_intra_window_dataset, opt)
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug


#def main(args):
if __name__ == "__main__":
    #args.gpu = os.environ['SLURM_PROCID'] % args.num_gpus
    best_ckpt_path, eval_split_name, eval_path, debug = start_training()

# if __name__ == "__main__":
#     if util.is_main_process():
#         logger.info("Setup config, data and model...")
#         logger.info(f"PID_{os.getpid()}")
#     args = BaseOptions().parse()
#     args.distributed = args.num_gpus > 1
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )
