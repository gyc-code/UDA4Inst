# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import torch.multiprocessing as mp
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
from detectron2.engine.uda_instance_utils import visulize_color_instances # correct_label_by_CLIP, \
#correct_label_by_GT, remove_wrong_label_instance_by_GT,remove_empty_instance_by_GT, keep_stuff_label_instance_by_GT


from detectron2.data import MetadataCatalog
from cityscapesscripts.helpers.labels import name2label

from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# constants

EVAL=True
VISUAL = False
ONLY_VAL = False


"python demo.py --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"




def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        # default="/home/yguo/Documents/other/detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        # default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_uda.yaml",
        default="configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k_kitti.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        # default=['/home/yguo/Documents/other/detectron2/demo/b.jpg'],
        # default=['/home/yguo/Documents/other/UDA4Inst/debug_cindy'],
        # default=['/datafast/120-1/Datasets/segmentation/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val'],
        # default=['datasets/synscapes/category_img_synscapes_instance_val.txt'],
        default=['datasets/kitti360/2013_05_28_drive_val_frames_image_all.txt'],
        
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='visual_instance/category_kitti/test/',
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
        # default=['MODEL.WEIGHTS','./output/smartmix/urbansyn_random_small_fix_20kbs3/model_best.pth'],
        # default=['MODEL.WEIGHTS','./output/smartmix/urbansyn_only_source_range_5_10/model_final.pth'],
        # default=['MODEL.WEIGHTS','./output/category/synscapes_full/model_final.pth'],
        # default=['MODEL.WEIGHTS','./output/category/urbansyn_full/model_final.pth'],
        default=['MODEL.WEIGHTS','./output/category/synthia_full/model_final.pth'],
        
        
        nargs=argparse.REMAINDER,
    )
    return parser


def process_one(path, demo_human_cycle, _metadata, result_save_folder, visul_save_folder, other_map_save_folder, dataset_name='cityscapes'):
    # use PIL, to be consistent with evaluation
    # print('-')
    path = str(path)
    print('time:',time.time(), 'processing : ', path, flush=True)
    basename, pred_txt, file_name = get_names(path, dataset_name, result_save_folder)
    mask_img, visual_pred, error_map_filename = None, None, None
    
    # print('path of img is : ', path)
    img = read_image(path, format="BGR")
    out_filename = os.path.join(visul_save_folder, basename+'.png')
    predictions_fuse, visualized_output, visualizer = demo_human_cycle.run_on_image_for_instance(img)
    if VISUAL:
        visualized_output.save(out_filename.replace('.png', '_mask.png'))

    cpu_device = torch.device("cpu")
    instances = predictions_fuse['instances'].to(cpu_device)
    """ save instances for eval """
    num_instances = len(instances)

            
    if VISUAL and num_instances != 0:
        mask_img, visual_pred, error_map_filename = visual_instance_mask(instances, img, visul_save_folder, basename, out_filename)

    save_result(num_instances, pred_txt, VISUAL, visual_pred, instances, _metadata, name2label,\
        result_save_folder, basename, file_name, dataset_name, error_map_filename, out_filename, mask_img)


def process_all(inputs, demo, result_save_folder, visul_save_folder, error_map_save_folder):
    _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
    for path in tqdm.tqdm(inputs):
        process_one(path, demo, _metadata, result_save_folder, visul_save_folder, error_map_save_folder, dataset_name='kitti360')


if __name__ == "__main__":
    if not ONLY_VAL:
        args = get_parser().parse_args()
        result_save_folder, visul_save_folder, other_map_save_folder  = preparation(args.output)

        '''  input multi model, seperate by ' ', run loop'''
        args_copy = copy.deepcopy(args)
        model_weights = args_copy.opts[1].split(' ')
        model = model_weights[0]
        args.opts[1] = model
        cfg = setup_cfg(args)
        demo = VisualizationDemo(cfg)
        target = '-'
        args_input = args.input
        if len(args_input) == 1:
            if os.path.isdir(args_input[0]):
                inputs = sorted(Path(args_input[0]).glob('*/*.png'))
            elif os.path.isfile(args_input[0]):
                with open(args_input[0], 'r') as file:
                    lines = file.readlines()
                inputs = [line.strip() for line in lines]

        # process_all(inputs, demo, result_save_folder, visul_save_folder, other_map_save_folder)
        ''' multi-process '''
        mp.set_start_method("spawn", force=True)
        _metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val")
        paramers = []
        dataset_name = 'kitti360'
        for i in range(len(inputs)):
            paramers.append((inputs[i], demo, _metadata, result_save_folder, visul_save_folder, other_map_save_folder, dataset_name))
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
            # eval on kitti
            _temp_dir = result_save_folder
            evaluate_kitti(_temp_dir)