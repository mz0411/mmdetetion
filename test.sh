export CUDA_VISIBLE_DEVICES=1

python tools/test.py \
       configs/swin/cascade_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_aicity22.py \
       pth/best_model.pth \
       --format-only \
       --options jsonfile_prefix=test-det-2666
