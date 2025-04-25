# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
from pathlib import Path
import warnings
import os
import copy
from PIL import Image
import shutil

import cv2
import numpy as np
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from detectron2.structures import PolygonMasks, Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation.cityscapes_evaluation import process_train_id_to_color_img
from detectron2.engine.uda_instance_utils import visulize_color_instances


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def creat_empty_folder(folder_path): 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
        
def organise_evaluate_folder(evaluate_folder):
    " for cityscape evaluate"
    city_names = ['frankfurt', 'lindau', 'munster']
    eval_path = Path(evaluate_folder)
    for city in city_names:
        eval_city_generate = eval_path.rglob(city + "*.png")
        eval_city_list = list(eval_city_generate)
        for i in eval_city_list:
            shutil.move(str(i), evaluate_folder + os.sep + city)
            
def get_error_map(gt_color, visual_pred_color):
    image_a = gt_color
    image_b = visual_pred_color

    if image_a.shape == image_b.shape:
        # 创建一个全零的图像，用于存储差异结果
        difference_image = np.zeros_like(image_a, dtype=np.uint8)

        # 定义白色像素值，表示无数据区域
        white = [255, 255, 255]

        # 创建掩码，标记哪些像素有数据（非白色）
        gt_has_data = np.any(image_a != white, axis=-1)
        pred_has_data = np.any(image_b != white, axis=-1)

        # 两个图像都在该像素位置有数据
        both_have_data = gt_has_data & pred_has_data

        # 检查在有数据的位置，两个图像的颜色是否不同
        colors_different = np.any(image_a != image_b, axis=-1)

        # 定义不同的掩码
        # 1. gt_color有，visual_pred_color没有：橙色
        mask_orange = gt_has_data & (~pred_has_data)

        # 2. gt_color没有，visual_pred_color有：灰色
        mask_gray = (~gt_has_data) & pred_has_data

        # 3. 都有数据，但颜色不同：蓝色
        mask_blue = both_have_data & colors_different

        # 4. 都有数据，且颜色相同：绿色
        mask_green = both_have_data & (~colors_different)

        # 5. 都没有数据：白色
        mask_white = (~gt_has_data) & (~pred_has_data)

        # 分配颜色（OpenCV使用BGR格式）
        difference_image[mask_orange] = [0, 165, 255]   # 橙色
        difference_image[mask_gray] = [128, 128, 128]   # 灰色
        difference_image[mask_blue] = [255, 0, 0]       # 蓝色
        difference_image[mask_green] = [0, 255, 0]      # 绿色
        difference_image[mask_white] = [255, 255, 255]  # 白色

        # 在error_map上添加图例
        error_map = difference_image.copy()
        error_map = add_legend(error_map)

        # 如果需要转换为RGB格式
        error_map_rgb = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

        return error_map_rgb
    else:
        print("输入的图像尺寸不一致。")
        return None

def add_legend(image):
    # 定义图例信息：颜色和对应的文本
    legend_info = [
        ([0, 165, 255], 'GT exit, No Pred'),      # 橙色
        ([128, 128, 128], 'No GT, Pred exit'),    # 灰色
        ([255, 0, 0], 'class error'),            # 蓝色
        ([0, 255, 0], 'correct'),            # 绿色
    ]

    # 图例起始位置
    start_x, start_y = 10, 10
    rect_size = 20  # 色块大小
    spacing = 5     # 间隔

    # 设置文字字体和大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    for i, (color, text) in enumerate(legend_info):
        # 计算位置
        y = start_y + i * (rect_size + spacing)
        # 绘制颜色方块
        cv2.rectangle(image, (start_x, y), (start_x + rect_size, y + rect_size), color, -1)
        # 添加文字（OpenCV的文字颜色是BGR）
        cv2.putText(image, text, (start_x + rect_size + spacing, y + rect_size - 5), 
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image            


def get_error_map_bp(gt_color, visual_pred_color):
    image_a = gt_color
    image_b = visual_pred_color

    if visual_pred_color.shape == gt_color.shape:

        # 创建一个全零的图像，用于存储差异结果
        difference_image = np.zeros_like(image_a, dtype=np.uint8)
        mask = np.all(image_a == [255, 255, 255], axis=-1) & np.all(image_b == [255, 255, 255], axis=-1)


        # 找到两张图像不同的地方
        difference_mask = np.any(image_a != image_b, axis=-1)

        # 在差异的位置使用红色表示
        difference_image[difference_mask] = [33, 33, 255]  # 红色 [B, G, R]

        # # 在相同的位置使用绿色表示
        same_mask = np.logical_not(difference_mask)
        difference_image[same_mask] = [108,238,108]  # 绿色 [B, G, R]
        difference_image[mask] = [255, 255, 255]

        error_map = difference_image
        error_map = cv2.cvtColor(error_map, cv2.COLOR_BGR2RGB)

    return error_map

def cat_pred_gt(visual_pred_color, mask_vis_output, file_name, dataset_name='cityscapes'):

    if dataset_name == 'urbansyn':
        # '/home/yguo/Documents/other/detectron2/datasets/urbansyn/poblenou_terrain/rgb_translated_cityscapes/image_scene_013_beauty_0013.png'
        #/home/yguo/Documents/other/UDA4Inst/datasets/urbansyn_total_label/urbansyn_total_label/poblenou_image_scene_001_objectcolor_0001_gtFine_labelIds.png
        city_name = file_name.split('/')[-3]
        image_name = file_name.split('/')[-1].replace('_beauty_', '_objectcolor_').replace('.png', '_gtFine_labelIds.png')
        train_id_path = './datasets/urbansyn_total_label/urbansyn_total_label/' + city_name + image_name
        
        gt_path = None# no color gt for urbansyn
        train_id_path = file_name.replace('img_urbansyn_instance_category_val', 'label_urbansyn_instance_category_val').replace('.png', '_trainId.png')
        
    if dataset_name == 'cityscapes':
        gt_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_color.png')
        train_id_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        train_id_img = cv2.imread(train_id_path)
        rgb_img = cv2.imread(file_name)

        
    elif dataset_name == 'kitti360':
        # file_name=2013_05_28_drive_0000_sync_image_00_data_rect_0000000386.png
        # datasets/kitti360/KITTI360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic_rgb/0000000250.png
        root_name = 'datasets/kitti360/KITTI360/data_2d_semantics/train'
        sequence = file_name.split('_')[4]
        frame = file_name.split('_')[-1].split('.')[0]
        gt_path = os.path.join(root_name, '2013_05_28_drive_' + sequence + '_sync', 'image_00', 'semantic_rgb', frame + '.png')
        train_id_path = gt_path.replace('semantic_rgb', 'instance')
        # data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000000.png
        rgb_path = gt_path.replace('/data_2d_semantics/train/', '/data_2d_raw/').replace('semantic_rgb', 'data_rect')
        
        instance_img = Image.open(train_id_path)
        instance_array = np.array(instance_img, dtype=np.uint16)
        image_shape = instance_array.shape
        # 获取每个像素的 instanceID
        instance_ids = instance_array
        # 计算 semanticID 和 classInstanceID
        semantic_ids = instance_ids // 1000
        # 定义映射字典
        mapping = {
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18
        }
        # 创建映射函数
        def map_values(value):
            return mapping.get(value, 0)
        # 使用 numpy 的 vectorize 函数创建矢量化的映射函数
        vectorized_map_values = np.vectorize(map_values)
        # 生成新的图
        new_semantic_ids = vectorized_map_values(semantic_ids)

        train_id_img = np.stack([new_semantic_ids] * 3, axis=-1)
        rgb_img = cv2.imread(rgb_path)
        
    gt_img = cv2.imread(gt_path)
    # train_id_img = cv2.imread(train_id_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    human_cycle_vehicle_train_id = [11, 12, 13, 14, 15, 16, 17, 18]
    train_id_img_processed = np.where(np.isin(train_id_img, human_cycle_vehicle_train_id), train_id_img, 255)
    mask = train_id_img_processed==255
    mask_huam_cycle_vehicle = ~mask * 1
    gt_img = gt_img * mask_huam_cycle_vehicle
    black_pixels = np.all(gt_img == [0, 0, 0], axis=-1)
    gt_img[black_pixels] = [255, 255, 255]

    error_map = get_error_map(gt_img.astype("uint8"), visual_pred_color)

    h,w,c = rgb_img.shape
    black_bar_shape = (10, w, c)
    black_bar = np.zeros(black_bar_shape, dtype=np.uint8)
    rgb_blackbar = np.vstack((rgb_img, black_bar))
    rgb_mask = np.vstack((rgb_blackbar, mask_vis_output))
    rgb_mask_blackbar = np.vstack((rgb_mask, black_bar))
    rgb_mask_gt = np.vstack((rgb_mask_blackbar, gt_img))
    rgb_mask_gt_blackbar = np.vstack((rgb_mask_gt, black_bar))
    rgb_mask_gt_pred = np.vstack((rgb_mask_gt_blackbar,visual_pred_color))
    rgb_mask_gt_pred_blackbar = np.vstack((rgb_mask_gt_pred,black_bar))
    rgb_mask_gt_pred_errormap_blackbar = np.vstack((rgb_mask_gt_pred_blackbar,error_map))
    return rgb_mask_gt_pred_errormap_blackbar, error_map



def evaluate_kitti(result_path):
    import kitti360scripts.evaluation.semantic_2d.evalInstanceLevelSemanticLabeling as kitti360_eval

    # set some global states in cityscapes evaluation API, before evaluating
    kitti360_eval.args.predictionPath = os.path.abspath(result_path)
    kitti360_eval.args.predictionWalk = None
    kitti360_eval.args.JSONOutput = False
    kitti360_eval.args.colorized = False
    kitti360_eval.args.gtInstancesFile = os.path.join(result_path, "gtInstances_kitti360.json")
    predictionImgList = []
    groundTruthImgList = []
    # args.groundTruthListFile = os.path.join(args.kitti360Path, 'data_2d_semantics', 'train', '2013_05_28_drive_val_frames.txt')
    groundTruthListFile = '/home/yguo/Documents/other/UDA4Inst/datasets/kitti360/2013_05_28_drive_val_frames_all.txt'
    # use the ground truth search string specified above
    groundTruthImgList = kitti360_eval.getGroundTruth(groundTruthListFile)
    if not groundTruthImgList:
        print("Cannot find any ground truth images to use for evaluation.")
    # get the corresponding prediction for each ground truth imag
    for gt,_ in groundTruthImgList:
        predictionImgList.append( kitti360_eval.getPrediction(kitti360_eval.args, gt) )
    results = kitti360_eval.evaluateImgLists(
        predictionImgList, groundTruthImgList, kitti360_eval.args
    )
    
    
def preparation(output):
    folder = output
    result_save_folder = folder + 'instance_img'
    visul_save_folder = folder  + 'visul_img'
    other_map_save_folder = folder  + 'other_map'
    creat_empty_folder(result_save_folder)
    creat_empty_folder(visul_save_folder)
    creat_empty_folder(other_map_save_folder)
    print('result_save_folder:', result_save_folder)
    return result_save_folder, visul_save_folder, other_map_save_folder

def draw_text(mask, mask_img, classes, score):
    true_positions = torch.where(mask)
    if true_positions[0].numel() > 0:
        mid_0 = int(0.5*(true_positions[0][-1].item() - true_positions[0][0].item())) + true_positions[0][0].item()
        mid_1 = int(0.5*(true_positions[1][-1].item() - true_positions[1][0].item())) + true_positions[1][0].item()
        if mid_1 > 1995:
            mid_1 = 1995
        text_position = (mid_1, mid_0)
        cv2.putText(mask_img, classes + str(round(score.item(), 2)), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return mask_img

def save_error_map(mask_img, visual_pred, error_map_filename, file_name, out_filename, dataset_name):
    visual_semantic_pred_color = process_train_id_to_color_img(visual_pred[:,:,0])
    final_img, error_map = cat_pred_gt(visual_semantic_pred_color, mask_img, file_name, dataset_name)
    cv2.imwrite(out_filename.replace('.png', '_mask_text.png'), mask_img)
    Image.fromarray(error_map.astype("uint8")).save(error_map_filename)
    
def  get_names(path, dataset_name, result_save_folder):
    file_name = path
    if dataset_name == 'cityscapes':
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_txt = os.path.join(result_save_folder, basename + "_pred.txt")
    elif dataset_name == 'kitti360':
        parts = file_name.split('/')
        last_four_parts = parts[-4:]
        file_name = ('_'.join(last_four_parts))
        basename = file_name.split('.')[0]
        pred_txt = os.path.join(result_save_folder, file_name.replace(".png", "_pred.txt"))
    return basename, pred_txt, file_name

def visual_instance_mask(instances, img, visul_save_folder, basename, out_filename):
    visual_pred = 255 * np.ones(img.shape, dtype=np.uint8)
    error_map_filename = os.path.join(visul_save_folder, basename + "_error_map.png")
    color_pseudo_instances = visulize_color_instances(instances)
    color_pseudo_instances_path = os.path.join(visul_save_folder, basename + "_instance.png")
    mask_img = cv2.imread(out_filename.replace('.png', '_mask.png'))
    Image.fromarray(color_pseudo_instances).save(color_pseudo_instances_path)
    return mask_img, visual_pred, error_map_filename

def save_result(num_instances, pred_txt, VISUAL, visual_pred, instances, _metadata, name2label, \
    result_save_folder, basename, file_name, dataset_name, error_map_filename, out_filename, mask_img):
    with open(pred_txt, "w") as fout:
        for i in range(num_instances):
            pred_class = instances.pred_classes[i]
            classes = _metadata.thing_classes[pred_class]
            class_id = name2label[classes].id
            class_train_id = name2label[classes].trainId
            score = instances.scores[i]
            mask = instances.pred_masks[i].bool()
            if VISUAL:
                mask_img = draw_text(mask, mask_img, classes, score)
                visual_pred[mask] = class_train_id
            png_filename = os.path.join(result_save_folder, basename + "_{}_{}.png".format(i, classes))
            Image.fromarray(instances.pred_masks[i].numpy().astype("uint8") * 255).save(png_filename)
            fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))
            
        if VISUAL and num_instances != 0:
            save_error_map(mask_img, visual_pred, error_map_filename, file_name, out_filename, dataset_name)
