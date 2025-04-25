# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import time
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from cityscapesscripts.helpers.labels import id2label, name2label
from shapely.geometry import MultiPolygon, Polygon


try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

VISUALIZE_POLYGON = False

logger = logging.getLogger(__name__)

def _get_syn_real_uda_files_from_filelist(source_image_dir, source_gt_dir, target_image_dir):
    files = []
    with open(source_image_dir,'r') as f:
        source_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(source_gt_dir,'r') as f:
        source_labels = [line.rstrip().split(' ') for line in f.readlines()]
    with open(target_image_dir,'r') as f:
        target_images = [line.rstrip().split(' ') for line in f.readlines()]

    for idx, source_image in enumerate(source_images):
        source_label_file = source_labels[idx][0]

        # idx_target = np.random.choice(range(len(target_images)))
        # another option : get target by index
        idx_target = idx if (idx < len(target_images)) else (idx % len(target_images))
        # print('target index ',idx,  idx_target)
        target_image = target_images[idx_target]
        if "_gtFine_labelTrainIds.png" in source_label_file:
            source_label_file = source_label_file.replace("_gtFine_labelTrainIds.png", "_gtFine_labelIds.png")

        source_instance_file = source_label_file.replace("_gtFine_labelIds.png", "_gtFine_instanceIds.png")
        source_json_file = source_label_file.replace("_gtFine_labelIds.png", "_gtFine_polygons.json")

        source = (source_image[0], source_instance_file, source_label_file, source_json_file)
        target = (target_image[0])
        files.append({'source' : source, 'target':target})
    
    assert len(files), "No images found in {}".format(source_image_dir)
    for f in files[0]['source']:
        assert PathManager.isfile(f), f
    assert PathManager.isfile(files[0]['target']), files[0]['target']
    return files

def _get_files(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True):
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    if os.path.isfile(source_image_dir) and  os.path.isfile(source_gt_dir) and  os.path.isfile(target_image_dir):
        files = _get_syn_real_uda_files_from_filelist(source_image_dir, source_gt_dir, target_image_dir)
        return files
    else:
        print('please give file path for datasets')
        return None
    
def _process_ret_dataset_id_to_contiguous_id(ret):
    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels
    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image['source']["annotations"]:
            try:
                anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
            except:
                dict_per_image['source']["annotations"].remove(anno)#  cindy 
                continue
    return ret

def load_syn_real_uda_instances(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        category: cindy: human_cycle_vehicle/ vehicle / human_cycle
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = _get_files(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True)
    if files is None:
        return

    logger.info("Preprocessing uda annotations ...")
    with PathManager.open(files[0]['target'], "rb") as f:
        inst_image = np.asarray(Image.open(f), order="F")
        target_h, target_w, c = inst_image.shape
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 8, 4))
    ret = pool.map(
        functools.partial(_uda_files_to_dict, h=target_h, w=target_w, category='human_cycle_vehicle', from_json=from_json, to_polygons=to_polygons),
        files,
    )
    ##### use in debug
    # ret = []
    # for i in range (len(files)):
    #     uda_ret = _uda_files_to_dict(files[i], target_h, target_w, category='human_cycle_vehicle', from_json=from_json, to_polygons=to_polygons)
    #     ret.append(uda_ret)


    logger.info("Loaded {} images from {}".format(len(ret), source_image_dir))
    ret = _process_ret_dataset_id_to_contiguous_id(ret)
    return ret

def load_syn_real_uda_human_cycle_instances(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        category: cindy: human_cycle_vehicle/ vehicle / human_cycle
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = _get_files(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True)
    if files is None:
        return

    logger.info("Preprocessing uda human_cycle annotations ...")
    with PathManager.open(files[0]['target'], "rb") as f:
        inst_image = np.asarray(Image.open(f), order="F")
        target_h, target_w, c = inst_image.shape
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 8, 4))
    ret = pool.map(
        functools.partial(_uda_files_to_dict, h=target_h, w=target_w, category='human_cycle', from_json=from_json, to_polygons=to_polygons),
        files,
    )
    ##### use in debug
    # ret = []
    # for i in range (len(files)):
    #     uda_ret = _uda_files_to_dict(files[i], target_h, target_w, category=category, from_json=from_json, to_polygons=to_polygons)
    #     ret.append(uda_ret)
    logger.info("Loaded {} images from {}".format(len(ret), source_image_dir))
    ret = _process_ret_dataset_id_to_contiguous_id(ret)
    return ret

def load_syn_real_uda_vehicle_instances(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        category: cindy: human_cycle_vehicle/ vehicle / human_cycle
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = _get_files(source_image_dir, source_gt_dir, target_image_dir, from_json=True, to_polygons=True)
    if files is None:
        return

    logger.info("Preprocessing uda vehicle annotations ...")
    with PathManager.open(files[0]['target'], "rb") as f:
        inst_image = np.asarray(Image.open(f), order="F")
        target_h, target_w, c = inst_image.shape
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 8, 4))
    ret = pool.map(
        functools.partial(_uda_files_to_dict, h=target_h, w=target_w, category='vehicle', from_json=from_json, to_polygons=to_polygons),
        files,
    )
    ##### use in debug
    # ret = []
    # for i in range (len(files)):
    #     uda_ret = _uda_files_to_dict(files[i], target_h, target_w, category=category, from_json=from_json, to_polygons=to_polygons)
    #     ret.append(uda_ret)
    logger.info("Loaded {} images from {}".format(len(ret), source_image_dir))
    ret = _process_ret_dataset_id_to_contiguous_id(ret)
    return ret


def _uda_files_to_dict(uda_file, h=1024, w=2048, category='human_cycle_vehicle', from_json=True, to_polygons=True):
    source_ret = _source_cityscapes_files_to_dict(uda_file['source'], category=category, from_json=from_json, to_polygons=to_polygons)
    target_ret = _target_files_to_dict(uda_file['target'], h, w)
    ret_uda = {'source':source_ret, 'target':target_ret}
    return ret_uda

def _source_cityscapes_files_to_dict(files, category='human_cycle_vehicle', from_json=True, to_polygons=True):

    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    # from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files
    annos = []

    if from_json:
        # from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https/home/yguo/Documents/other/detectron2/datasets/synscapes/img://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        if VISUALIZE_POLYGON:
            img_clone = cv2.imread(image_file) ####  cindy add, visulize polygon

        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            if category == 'vehicle': ## cindy 
                if label_name not in ['car', 'truck', 'bus', 'train']:
                    continue
            elif category == 'human_cycle':
                if label_name not in ['person', 'rider', 'motorcycle', 'bicycle']:
                    continue
            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format # cindy add , remove bike and motor
                continue

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id   

            # if "segmentation" in obj: # cindy add,  use coco logic for synscapes
            segm = obj.get("segmentation", None)
            if len(segm) == 0:
                continue  # ignore this instance
            # filter out invalid polygons (< 3 points)
            poly_coord = []
            for poly in segm:
                # if len(poly) % 2 == 0 and len(poly) >= 6:  # TODO : len(poly) is point number, why 2 times
                # print(len(poly))
                # if len(poly) % 2 == 0 and len(poly) >= 6: # default : 6
                if len(poly) >= 6: # default : 6
                    poly_xy = np.flip(np.array(poly), axis=1)
                    # print(poly_xy.shape)
                    x, y = poly_xy.shape
                    poly_xy_reshape = np.reshape(poly_xy, (1, x*y))
                    # print(poly_xy_reshape.shape)
                    poly_coord.append((poly_xy_reshape).tolist()[0])

                    if VISUALIZE_POLYGON:
                        cv2.polylines(img_clone,[poly_xy],True,(255,0,255), 1) ####  cindy add, visulize polygon

            anno["segmentation"] = poly_coord
            anno["bbox"] = obj["bbox"]
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            annos.append(anno)
        
        if VISUALIZE_POLYGON:
            cv2.imwrite(image_file.split('/')[-1].replace('.png', '_polygon.png') , img_clone) ####  cindy add, visulize polygon
        
    ret["annotations"] = annos
    return ret

def _target_files_to_dict(target_img_file, h, w):

    ret = {
        "file_name": target_img_file,
        "image_id": os.path.basename(target_img_file),
        "height": h,
        "width": w,
    }
    return ret
