# train ego4d without naq
######## Data paths, need to replace them based on your directory
train_path=data/ego_train.jsonl
eval_path=data/ego_val.jsonl
eval_split_name=val
results_root=results
######## Setup video/textual feature path, need to replace them based on your directory
text_feat_dir=data/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_clip_text_features
motion_feat_dir=data/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_video_feature_1.87fps
appearance_feat_dir=data/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_video_feature_1.87fps

########  Feature dimension
v_motion_feat_dim=256
v_appear_feat_dim=256
t_feat_dim=512

######## training
device_id=0
num_queries=5
max_v_l=90
bsz=32
clip_length=0.535 ## we extract video feature every 1.87 FPS, thus a clip duration is 1/1.87 = 0.535 second
max_q_l=20
num_workers=4

######## Hyper-parameter
n_epoch=150
lr_drop=120
max_es_cnt=-1
#10
eval_epoch_interval=1
start_epoch_for_adapter=-1
topk_window=20
dset_name=ego4d
retrieval_loss_coef=1
adapter_module=linear
exp_id=train_ego4d

CUDA_VISIBLE_DEVICES=${device_id} PYTHONPATH=$PYTHONPATH:. python cone/train.py \
--lr_drop 120 \
--n_epoch ${n_epoch} \
--max_es_cnt ${max_es_cnt} \
--eval_epoch_interval ${eval_epoch_interval} \
--start_epoch_for_adapter ${start_epoch_for_adapter} \
--clip_length ${clip_length}  \
--topk_window ${topk_window} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--dset_name ${dset_name} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--motion_feat_dir ${motion_feat_dir} \
--appearance_feat_dir ${appearance_feat_dir} \
--v_motion_feat_dim ${v_motion_feat_dim} \
--t_feat_dir ${text_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--v_appear_feat_dim ${v_appear_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--num_queries ${num_queries} \
--num_workers ${num_workers} \
--adapter_module ${adapter_module} \
--qddetr \
--retrieval_loss_coef ${retrieval_loss_coef} \
--start_epoch 0 \
--lr ${lr} \
--ret_eval \
--gumbel_eps 0.3 \
--nms_thd 0.5 \
--dim_feedforward 2048 \
--gumbel \
--gumbel_2 \
--gumbel_3 \
--dabdetr \
--gumbel_single_proj \
${@:6}
