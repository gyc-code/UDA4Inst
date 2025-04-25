# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import time
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

VISUALIZE_POLYGON = False

logger = logging.getLogger(__name__)


def _get_cityscapes_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = basename[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def _get_cityscapes_files_from_filelist(image_dir, gt_dir, data_name):
    files = []
    with open(image_dir,'r') as f:
        images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(gt_dir,'r') as f:
        labels = [line.rstrip().split(' ') for line in f.readlines()]
    for idx, image in enumerate(images):
        instance_file = labels[idx][0]
        # if "_gtFine_instanceIds.png" in label_file:
        if 'kitti360' in data_name:
            # KITTI360/data_2d_semantics/train/2013_05_28_drive_0005_sync/image_00/instance/0000006449.png
            # kitti360/polygon/2013_05_28_drive_0005_sync_image_00_instance_0000004472_gtFine_polygons.json
            parts = instance_file.split('/')
            last_four_parts = parts[-4:]
            json_file = (('/home/yguo/Documents/other/UDA4Inst/datasets/kitti360/polygon/' + '_'.join(last_four_parts)).replace('.png', '_gtFine_polygons.json')).replace('_semantic_', '_instance_')
            files.append((image[0], None, None, json_file))   
        else:
            # instance_file = label_file
            # label_file = instance_file.replace("_gtFine_instanceIds.png", "_gtFine_labelIds.png")
            json_file = instance_file.split('_gtFine_')[0] + "_gtFine_polygons.json"
            # json_file = instance_file.replace("_gtFine_labelIds.png", "_gtFine_polygons.json")
            files.append((image[0], None, None, json_file))
    
    assert len(files), "No images found in {}".format(image_dir)
    # for f in files[0]:
    #     if 'kitti360' in data_name:
    #         continue
    #     assert PathManager.isfile(f), f
    return files



def _get_files(image_dir, gt_dir, data_name, from_json=True, to_polygons=True):
    if from_json:
        assert to_polygons, (
            "urbansyn's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    if os.path.isfile(image_dir) and  os.path.isfile(gt_dir):
        files = _get_cityscapes_files_from_filelist(image_dir, gt_dir, data_name)
        return files
    elif os.path.isdir(image_dir) and  os.path.isdir(gt_dir):
        files = _get_cityscapes_files(image_dir, gt_dir) # original
        return files
    else:
        print('please give file path or dir path for images_dir and gt_dir ')
        return None

def _process_ret_dataset_id_to_contiguous_id(ret):
    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret

def load_urbansyn_instances(image_dir, gt_dir, data_name, category='human_cycle_vehicle', from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        category : human_cycle_vehicle, or human_cycle or vehicle
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = _get_files(image_dir, gt_dir, data_name, from_json=True, to_polygons=True)
    if files is None:
        return
    logger.info("Preprocessing annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    # debug
    _urbansyn_files_to_dict(files[0], category, from_json=from_json, to_polygons=to_polygons)
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    ret = pool.map(
        functools.partial(_urbansyn_files_to_dict, category=category, from_json=from_json, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    ret = _process_ret_dataset_id_to_contiguous_id(ret)
    return ret


def load_cityscapes_instances(image_dir, gt_dir, category='human_cycle_vehicle', from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if from_json:
        assert to_polygons, (
            "Cityscapes's json annotations are in polygon format. "
            "Converting to mask format is not supported now."
        )
    if os.path.isfile(image_dir) and  os.path.isfile(gt_dir):
        files = _get_cityscapes_files_from_filelist(image_dir, gt_dir)
    elif os.path.isdir(image_dir) and  os.path.isdir(gt_dir):
        files = _get_cityscapes_files(image_dir, gt_dir) # original
    else:
        print('please give file path or dir path for images_dir and gt_dir ')
        return

    logger.info("Preprocessing cityscapes annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    
    # ret = [_cityscapes_files_to_dict(files[0], category, from_json=from_json, to_polygons=to_polygons)]
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    partial_func = functools.partial(
    _cityscapes_files_to_dict,
    category,
    from_json=from_json,
    to_polygons=to_polygons
    )
    ret = pool.map(partial_func, files)
    
    # ret = pool.map(functools.partial(_cityscapes_files_to_dict,  from_json=from_json, to_polygons=to_polygons),files,category)
    
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


def load_cityscapes_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, _, label_file, json_file in _get_cityscapes_files(image_dir, gt_dir):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


def _cityscapes_files_to_dict(category, files,  from_json, to_polygons):
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
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # `polygons_union` contains the union of all valid polygons.
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            if category == 'vehicle': ## cindy 
                if label_name not in ['car', 'truck', 'bus', 'train']:
                # if label_name not in ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']: # a test
                # if label_name not in ['car', 'truck', 'train', 'bicycle']: # a test
                # if label_name not in [ 'truck', 'bus', 'train', 'motorcycle']: # a test
                    continue
            elif category == 'human_cycle':
                if label_name not in ['person', 'rider', 'motorcycle', 'bicycle']:
                # if label_name not in ['person', 'rider']: #, 'motorcycle', 'bicycle' # a test
                # if label_name not in ['person', 'rider', 'motorcycle', 'bus']: # a test
                # if label_name not in ['person', 'rider', 'car', 'bicycle']: #, 'motorcycle', 'bicycle' # a test
                    continue

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id

            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            poly_coord = []
            for poly_el in poly_list:
                # COCO API can work only with exterior boundaries now, hence we store only them.
                # TODO: store both exterior and interior boundaries once other parts of the
                # codebase support holes in polygons.
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            annos.append(anno)
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    return ret

def _urbansyn_files_to_dict(files, category, from_json, to_polygons):
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
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, _, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

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
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
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
                # if label_name not in ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']: # a test
                # if label_name not in ['car', 'truck', 'train', 'bicycle']: # a test
                # if label_name not in [ 'truck', 'bus', 'train', 'motorcycle']: # a test
                    continue
            elif category == 'human_cycle':
                if label_name not in ['person', 'rider', 'motorcycle', 'bicycle']:
                # if label_name not in ['person', 'rider']: #, 'motorcycle', 'bicycle' # a test
                # if label_name not in ['person', 'rider', 'motorcycle', 'bus']: # a test
                # if label_name not in ['person', 'rider', 'car', 'bicycle']: #, 'motorcycle', 'bicycle' # a test
                    continue
            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
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



if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("gt_dir")
    parser.add_argument("--type", choices=["instance", "semantic"], default="instance")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer
    from cityscapesscripts.helpers.labels import labels

    logger = setup_logger(name=__name__)

    dirname = "cityscapes-data-vis"
    os.makedirs(dirname, exist_ok=True)

    if args.type == "instance":
        dicts = load_cityscapes_instances(
            args.image_dir, args.gt_dir, from_json=True, to_polygons=True
        )
        logger.info("Done loading {} samples.".format(len(dicts)))

        thing_classes = [k.name for k in labels if k.hasInstances and not k.ignoreInEval]
        meta = Metadata().set(thing_classes=thing_classes)

    else:
        dicts = load_cityscapes_semantic(args.image_dir, args.gt_dir)
        logger.info("Done loading {} samples.".format(len(dicts)))

        stuff_classes = [k.name for k in labels if k.trainId != 255]
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)

    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
