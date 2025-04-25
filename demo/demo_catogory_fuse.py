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


from detectron2.data import MetadataCatalog
from cityscapesscripts.helpers.labels import name2label


from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# constants

EVAL = True
VISUAL = False
ONLY_VAL = False


"python demo.py --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="/home/yguo/Documents/other/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_kitti.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # default=['datasets/synscapes/category_img_synscapes_instance_val.txt'],
        default=['datasets/kitti360/2013_05_28_drive_val_frames_image_all.txt'],
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='visual_instance/category_kitti/category_synscapes_500_on_kitti360/',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        # default=['MODEL.WEIGHTS','detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'],
        # default=['MODEL.WEIGHTS','./output/uda_synscapes_clean2_1024_bs3from_coco_huamn_cycle_t2s_s2t_motor_augum/model_best.pth ./output/uda_synscapes_clean2_1024_bs3from_coco_vehicle_t2s_s2t_train_augum/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/instan_seg/uda_urabn_human_cycle_1024_from_pre_coco_bs3_p0.9_t2s_s2t-motor-augu/model_best.pth ./output/instan_seg/uda_urabn_vehicle_1024_from_pre_coco_bs3_p0.9_t2s_s2t-train-source-augu/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/uda_synthia_human_cycle_1024_from_pre_coco_bs3_p0.9_0.25t2s_0.75s2t-motor-augu/model_best.pth ./output/uda_synthia_vehicle_1024_from_pre_coco_bs3_p0.9_0.25t2s_0.75s2t-bus-augu/model_best.pth'],
        default=['MODEL.WEIGHTS','./output/category/synscapes_human_cycle/model_final.pth ./output/category/synscapes_vehicle/model_final.pth'],
        # default=['MODEL.WEIGHTS','./output/category/urbansyn_human_cycle/model_final.pth ./output/category/urbansyn_vehicle/model_final.pth'],
        # default=['MODEL.WEIGHTS','./output/category/synthia_human_cycle/model_final.pth ./output/category/synthia_vehicle/model_final.pth'],
        
        nargs=argparse.REMAINDER,
    )
    return parser

def process_one(path, demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, error_map_save_folder, target, dataset_name='cityscapes'):
    # use PIL, to be consistent with evaluation
    path = str(path)
    print('time:',time.time(), 'processing : ', path, flush=True)
    
    basename, pred_txt, file_name = get_names(path, dataset_name, result_save_folder)
    mask_img, visual_pred, error_map_filename = None, None, None
    # print('path of img is : ', path)
    img = read_image(path, format="BGR")
    img_copy = copy.deepcopy(img)
    out_filename = os.path.join(visul_save_folder, basename+'.png')
    predictions_human_cycle, visualized_output_human_cycle, visualizer = demo_human_cycle.run_on_image_for_instance(img)
    predictions_vehicle, visualized_output_vehicle, visualizer = demo_vehicle.run_on_image_for_instance(img_copy)
    
    if VISUAL:
        visualized_output_human_cycle.save(out_filename.replace('.png', '_human_cyc.png'))
        visualized_output_vehicle.save(out_filename.replace('.png', '_vehicle.png'))

    predictions_fuse = Instances.cat([predictions_human_cycle['instances'], predictions_vehicle['instances']])
    cpu_device = torch.device("cpu")
    instances = predictions_fuse.to(cpu_device)
    num_instances = len(instances)
    
    """ save instances for eval """
    predictions_human_cycle = demo_human_cycle.predictor(img)
    predictions_vehicle = demo_vehicle.predictor(img_copy)

    
    if VISUAL and num_instances != 0:
        mask_img, visual_pred, error_map_filename = visual_instance_mask(instances, img, visul_save_folder, basename, out_filename)

    save_result(num_instances, pred_txt, VISUAL, visual_pred, instances, _metadata, name2label,\
        result_save_folder, basename, file_name, dataset_name, error_map_filename, out_filename, mask_img)


def process_all(inputs, demo_human_cycle, demo_vehicle, result_save_folder, visul_save_folder, error_map_save_folder, target):
    _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
    for path in tqdm.tqdm(inputs):
        process_one(path, demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, error_map_save_folder, target, dataset_name='kitti360')


if __name__ == "__main__":
    if not ONLY_VAL:
        args = get_parser().parse_args()
        result_save_folder, visul_save_folder, other_map_save_folder  = preparation(args.output)

        '''  input multi model, seperate by ' ', run loop'''
        args_copy = copy.deepcopy(args)
        model_weights = args_copy.opts[1].split(' ')
        model_human_cycle = model_weights[0]
        model_vehicle = model_weights[1]

        args.opts[1] = model_human_cycle
        cfg = setup_cfg(args)
        demo_human_cycle = VisualizationDemo(cfg)

        args.opts[1] = model_vehicle
        cfg = setup_cfg(args)
        demo_vehicle = VisualizationDemo(cfg)

        target = 'human_cycle_vehicle'
    
        args_input = args.input
        if len(args_input) == 1:
            if os.path.isdir(args_input[0]):
                inputs = sorted(Path(args_input[0]).glob('*/*.png'))

            elif os.path.isfile(args_input[0]):
                with open(args_input[0], 'r') as file:
                    lines = file.readlines()
                inputs = [line.strip() for line in lines]
                
        # process_all(inputs, demo_human_cycle, demo_vehicle, result_save_folder, visul_save_folder, other_map_save_folder, target)
        
        ''' multi-process '''
        mp.set_start_method("spawn", force=True)
        _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
        paramers = []
        dataset_name = 'kitti360'
        for i in range(len(inputs)):
            paramers.append((inputs[i], demo_human_cycle, demo_vehicle, _metadata, result_save_folder, visul_save_folder, other_map_save_folder, target, dataset_name))
        # input :path, demo, _metadata, result_save_folder, visul_save_folder, error_map_save_folder
        pool = mp.Pool(processes=2)
        pool.starmap(process_one, paramers)
    
    if EVAL:
        if 0:
            # organise_evaluate_folder(result_save_folder)
            # organise_evaluate_folder(visul_save_folder)
            os.environ['CITYSCAPES_RESULTS'] = result_save_folder
            # os.environ['CITYSCAPES_RESULTS'] = 'visual_instance/category/category_urbansyn_1model_final.pth_instance_img'
            # os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py')
            os.system('python /home/yguo/Documents/cityscapesScripts/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling_urbansyn.py')
        if 1:
            _temp_dir = result_save_folder
            evaluate_kitti(_temp_dir)