######## Data paths, need to replace them based on your directory
train_path=data/mad_train.jsonl
eval_path=data/mad_val.jsonl
eval_split_name=val
results_root=results
######## Setup video/textual feature path, need to replace them based on youf
# r directory
motion_feat_dir=data/mad_data_for_cone/offline_lmdb/CLIP_B32_frames_features_5fps
appearance_feat_dir=data/mad_data_for_cone/offline_lmdb/CLIP_B32_frames_features_5fps
text_feat_dir=data/mad_data_for_cone/offline_lmdb/clip_clip_text_features

# Feature dimension
v_motion_feat_dim=512
v_appear_feat_dim=512
t_feat_dim=512

#### training
n_epoch=35
lr=1e-4
lr_drop=25
device_id=0
num_queries=5
max_v_l=900
bsz=8
eval_bsz=8
clip_length=0.2 ##  video feature are extracted by 5 FPS, thus a clip duration is 1/5 = 0.2 second
max_q_l=25
num_workers=4

######## Hyper-parameter
dset_name=mad
seed=2020
adapter_module=none #$4
max_es_cnt=-1
eval_epoch_interval=5
topk_window=30
start_epoch_for_adapter=100
adapter_loss_coef=0.2
retrieval_loss_coef=10
pos_temperature=100
exp_id=train_mad

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONPATH=$PYTHONPATH:. srun -u -t 4-00:00:00 -J mad_train --gres=gpu:4 -c 32 python -m torch.distributed.launch --master_port $RANDOM --nproc_per_node=4 --use_env cone/train.py \
--gumbel_eps 0.3 \
--gumbel_single_proj \
--nms_thd 0.5 \
--seed ${seed} \
--clip_length ${clip_length}  \
--max_es_cnt ${max_es_cnt} \
--topk_window ${topk_window} \
--eval_epoch_interval ${eval_epoch_interval} \
--start_epoch_for_adapter ${start_epoch_for_adapter} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--dset_name ${dset_name} \
--train_path ${train_path} \
--eval_split_name ${eval_split_name} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--t_feat_dir ${text_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--v_appear_feat_dim ${v_appear_feat_dim} \
--v_motion_feat_dim ${v_motion_feat_dim} \
--bsz ${bsz} \
--eval_bsz ${eval_bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--num_queries ${num_queries} \
--num_workers ${num_workers} \
--adapter_module ${adapter_module} \
--adapter_loss_coef ${adapter_loss_coef} \
--qddetr \
--retrieval_loss_coef ${retrieval_loss_coef} \
--enc_layers 2 \
--dec_layers 6 \
--dec_layers_2 2 \
--no_adapter_loss \
--winret \
--no_neg_contrast_loss \
--ret_eval \
--resume_all \
--gumbel \
--gumbel_2 \
--gumbel_3 \
--dabdetr \
--pos_temperature ${pos_temperature} \
--dim_feedforward 2048 \
${@:6}
