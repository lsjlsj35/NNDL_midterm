from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import mmcv
import os
from mmdet.visualization.local_visualizer import DetLocalVisualizer

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector
from matplotlib.patches import Rectangle

def visualize_prediction_with_masks(model, image_path, save_path, score_thr=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = inference_detector(model, image_path)
    pred_instances = result.pred_instances.cpu()

    bboxes = pred_instances.bboxes.numpy()
    labels = pred_instances.labels.numpy()
    scores = pred_instances.scores.numpy()
    masks = pred_instances.masks.numpy() if hasattr(pred_instances, 'masks') else None

    class_names = model.dataset_meta['classes']

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')

    height, width = image.shape[:2]

    for i in range(len(bboxes)):
        if scores[i] < score_thr:
            continue

        x1, y1, x2, y2 = bboxes[i]
        label = labels[i]
        caption = f'{class_names[label]}: {scores[i]:.2f}'

        # ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
        #                        edgecolor='lime', facecolor='none', linewidth=2))
        # ax.text(x1, y1 - 2, caption, color='white', fontsize=10,
        #         bbox=dict(facecolor='green', alpha=0.5, pad=1))

        assert masks is not None
        if masks is not None:
            mask = masks[i].astype(bool)
            color = np.random.rand(3) 
            mask_image = np.zeros((height, width, 4))  # RGBA
            for c in range(3):
                mask_image[:, :, c] = color[c]
            mask_image[:, :, 3] = mask.astype(float) 

            ax.imshow(mask_image)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


import torch
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

def load_model(config_path, checkpoint_path, device='cuda:0'):
    register_all_modules()

    cfg = Config.fromfile(config_path)
    if 'pretrained' in cfg.model:
        del cfg.model.pretrained

    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if 'metainfo' in checkpoint:
        model.dataset_meta = checkpoint['metainfo']
    else:
        model.dataset_meta = {
            'classes': (
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            )
        }
    model.cfg = cfg
    return model

# mask_model = load_model('work_dirs/mask/20250527_072056/vis_data/config.py', 'work_dirs/mask/epoch_12.pth')
mask_model = load_model('work_dirs/mask_final/mask_final.py', 'work_dirs/mask_final/epoch_15.pth')
# sparse_model = load_model('configs/voc_custom/sparse-rcnn_r50_fpn_1x_voc.py', 'work_dirs/sparse_rcnn/latest.pth')

voc_test_images = [
    'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000007.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000010.jpg',
]

# 可视化并保存
for img_path in voc_test_images:
    # visualize_and_save(mask_model, img_path, f'../outputs/mask_rcnn/{os.path.basename(img_path)}')
    visualize_prediction_with_masks(mask_model, img_path, f'../outputs/mask_rcnn/{os.path.basename(img_path)}')
    # visualize_and_save(sparse_model, img_path, f'outputs/sparse_rcnn/{os.path.basename(img_path)}')
