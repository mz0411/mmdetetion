from mmdet.apis import init_detector, inference_detector#, show_result
import mmcv
import os
import pdb
import cv2

config_file = 'configs/swin/cascade_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_aicity22.py'
checkpoint_file = 'pth/best_model.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

imgs_dir = 'visualize_imgs'
imgs = os.listdir(imgs_dir)
save_dir = 'pred_vis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for img_name in imgs:
    img_path = os.path.join(imgs_dir, img_name)
    result = inference_detector(model, img_path)
    model.show_result(img_path, result, score_thr=0.1, thickness=1, font_size=2, out_file=os.path.join(save_dir, img_name))
    
