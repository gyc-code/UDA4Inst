# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic, load_urbansyn_instances
from .udadataset_cityscapes import load_syn_real_uda_instances, load_syn_real_uda_human_cycle_instances, load_syn_real_uda_vehicle_instances
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    ######### key for eval on cityscapes
    "urbansyn_{task}_train_eval_cityscapes": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "synscapes_{task}_train_eval_cityscapes": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    "synthia_{task}_train_eval_cityscapes": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "urbansyn_human_cycle_{task}_train_eval_cityscapes": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "urbansyn_vehicle_{task}_train_eval_cityscapes": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "synthia_human_cycle_{task}_train_eval_cityscapes": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "synthia_vehicle_{task}_train_eval_cityscapes": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "synscapes_human_cycle_{task}_train_eval_cityscapes": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    "synscapes_vehicle_{task}_train_eval_cityscapes": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    ######### key for eval on cityscapes

    ######### key for eval on kitti360
    "urbansyn_{task}_train_eval_kitti360": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "synscapes_{task}_train_eval_kitti360": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    "synthia_{task}_train_eval_kitti360": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "urbansyn_human_cycle_{task}_train_eval_kitti360": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "urbansyn_vehicle_{task}_train_eval_kitti360": ("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"),
    "synthia_human_cycle_{task}_train_eval_kitti360": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "synthia_vehicle_{task}_train_eval_kitti360": ("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
    "synscapes_human_cycle_{task}_train_eval_kitti360": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    "synscapes_vehicle_{task}_train_eval_kitti360": ("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"),
    ######### key for eval on kitti360
    
    
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    # "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val_void255/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
    
    "kitti360_{task}_train": ("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt"),
    "kitti360_{task}_val": ("kitti360/2013_05_28_drive_val_frames_image_all.txt", "kitti360/2013_05_28_drive_val_frames_label_all.txt"),
    
    
    
    ######### for category experiments, keep
    # "urbansyn_category_vehicle_{task}_train": ("urbansyn_total_label/img_urbansyn_instance_category_train.txt", "urbansyn_total_label/label_urbansyn_instance_category_train.txt"),
    # "urbansyn_category_human_cycle_{task}_train": ("urbansyn_total_label/img_urbansyn_instance_category_train.txt", "urbansyn_total_label/label_urbansyn_instance_category_train.txt"),
    # "urbansyn_category_vehicle_{task}_val": ("urbansyn_total_label/img_urbansyn_instance_category_val.txt", "urbansyn_total_label/label_urbansyn_instance_category_val.txt"),
    # "urbansyn_category_human_cycle_{task}_val": ("urbansyn_total_label/img_urbansyn_instance_category_val.txt", "urbansyn_total_label/label_urbansyn_instance_category_val.txt"),   
    # "urbansyn_full_category_{task}_train": ("urbansyn_total_label/img_urbansyn_instance_category_train.txt", "urbansyn_total_label/label_urbansyn_instance_category_train.txt"),
    # "synscapes_category_human_cycle_{task}_train": ("synscapes/category_img_synscapes_instance_train.txt", "synscapes/category_label_synscapes_instance_train.txt"),
    # "synscapes_category_vehicle_{task}_train": ("synscapes/category_img_synscapes_instance_train.txt", "synscapes/category_label_synscapes_instance_train.txt"),
    # "synscapes_full_category_{task}_train": ("synscapes/category_img_synscapes_instance_train.txt", "synscapes/category_label_synscapes_instance_train.txt"),
    # "synthia_category_human_cycle_{task}_train": ("synthia/category_img_synthia_instance_train.txt", "synthia/category_label_synthia_instance_train.txt"),
    # "synthia_category_vehicle_{task}_train": ("synthia/category_img_synthia_instance_train.txt", "synthia/category_label_synthia_instance_train.txt"),
    # "synthia_full_category_{task}_train": ("synthia/category_img_synthia_instance_train.txt", "synthia/category_label_synthia_instance_train.txt"),
    # "cityscapes_fine_category_vehicle_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    # "cityscapes_fine_category_human_cycle_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    ######### for category experiments
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        data_name = key.split('_')[0]
        if 'cityscapes_fine' in  inst_key:
            if '_human_cycle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                        x, y, category='human_cycle', from_json=True, to_polygons=True
                    ),
                )
            elif '_vehicle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                        x, y, category='vehicle', from_json=True, to_polygons=True
                    ),
                )
            else:                
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                        x, y, category='human_cycle_vehicle', from_json=True, to_polygons=True
                    ),
                )
                
        elif 'urbansyn' in  inst_key or 'synscapes' in  inst_key or 'synthia' in inst_key or 'kitti360_' in inst_key:
            if '_human_cycle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir, data_name=data_name: load_urbansyn_instances(
                        x, y, data_name=data_name, category='human_cycle', from_json=True, to_polygons=True
                    ),
                )
            elif '_vehicle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir, data_name=data_name: load_urbansyn_instances(
                        x, y, data_name=data_name,category='vehicle', from_json=True, to_polygons=True
                    ),
                )
            else:
                DatasetCatalog.register(
                    inst_key,
                    lambda x=image_dir, y=gt_dir, data_name=data_name: load_urbansyn_instances(
                        x, y, data_name=data_name, category='human_cycle_vehicle', from_json=True, to_polygons=True
                    ),
                )

        if 'kitti360' in  inst_key:
            MetadataCatalog.get(inst_key).set(
                image_dir=image_dir, gt_dir=gt_dir, evaluator_type="kitti360_instance", **meta  #"cityscapes_instance"  cityscapes_instance2sem_seg
            )
        else:
            MetadataCatalog.get(inst_key).set(
                image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta 
            )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )

_RAW_UDA_CITYSCAPES_SPLITS = {
    #########  for eval on cityscapes
    "uda_syn_real_{task}_train_eval_cityscapes": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        # 'source':("img_3time_urban_synscapes_instance.txt", "label_3time_urban_synscapes_instance.txt"), 
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        # 'target':("cityscapes/leftimage8bit_train_small.txt", "cityscapes/gtFine_train_small.txt")

        },
    "uda_urbansyn_syn_real_human_cycle_{task}_train_eval_cityscapes": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },
    "uda_urbansyn_syn_real_vehicle_{task}_train_eval_cityscapes": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },

    "uda_synscapes_syn_real_human_cycle_{task}_train_eval_cityscapes": {
        'source':("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"), 
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },
    "uda_synscapes_syn_real_vehicle_{task}_train_eval_cityscapes": {
        'source':("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"), 
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },
    
    "uda_synthia_syn_real_human_cycle_{task}_train_eval_cityscapes": {
        'source':("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },
    "uda_synthia_syn_real_vehicle_{task}_train_eval_cityscapes": {
        'source':("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
        'target':("cityscapes/leftimage8bit_train.txt", "cityscapes/gtFine_train.txt")
        },

    #########  for eval on kitti360
    "uda_syn_real_{task}_train_eval_kitti360": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")

        },
    "uda_urbansyn_syn_real_human_cycle_{task}_train_eval_kitti360": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },
    "uda_urbansyn_syn_real_vehicle_{task}_train_eval_kitti360": {
        'source':("urbansyn_total_label/img_urbansyn_instance.txt", "urbansyn_total_label/label_urbansyn_instance.txt"), 
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },

    "uda_synscapes_syn_real_human_cycle_{task}_train_eval_kitti360": {
        'source':("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"), 
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },
    "uda_synscapes_syn_real_vehicle_{task}_train_eval_kitti360": {
        'source':("synscapes/img_synscapes_instance.txt", "synscapes/label_synscapes_instance.txt"), 
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },
    
    "uda_synthia_syn_real_human_cycle_{task}_train_eval_kitti360": {
        'source':("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },
    "uda_synthia_syn_real_vehicle_{task}_train_eval_kitti360": {
        'source':("synthia/img_synthia_instance.txt", "synthia/label_synthia_instance.txt"),
        'target':("kitti360/2013_05_28_drive_train_frames_image.txt", "kitti360/2013_05_28_drive_train_frames_label.txt")
        },
}

def register_all_uda_cityscapes(root): # design by cindy , only for instance so far,2023.9.28
    for key, value in _RAW_UDA_CITYSCAPES_SPLITS.items():
        if '_train' in key:
            source_image_dir, source_gt_dir = value['source']
            target_image_dir, target_gt_dir = value['target']
            meta = _get_builtin_metadata("cityscapes")
            source_image_dir = os.path.join(root, source_image_dir)
            source_gt_dir = os.path.join(root, source_gt_dir)
            target_image_dir = os.path.join(root, target_image_dir)
            target_gt_dir = os.path.join(root, target_gt_dir)
            inst_key = key.format(task="instance_seg")

            if '_human_cycle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    # cindy : only register the load data function and parammeters, when DatasetCatalog.get run the load data function
                    lambda x=source_image_dir, y=source_gt_dir, p=target_image_dir: load_syn_real_uda_human_cycle_instances(
                    x, y, p, from_json=True, to_polygons=True
                    ),
                )
            elif '_vehicle_' in inst_key:
                DatasetCatalog.register(
                    inst_key,
                    # cindy : only register the load data function and parammeters, when DatasetCatalog.get run the load data function
                    lambda x=source_image_dir, y=source_gt_dir, p=target_image_dir: load_syn_real_uda_vehicle_instances(
                    x, y, p, from_json=True, to_polygons=True
                    ),
                )
            else:
                DatasetCatalog.register(
                    inst_key,
                    # cindy : only register the load data function and parammeters, when DatasetCatalog.get run the load data function
                    lambda x=source_image_dir, y=source_gt_dir, p=target_image_dir: load_syn_real_uda_instances(
                    x, y, p, from_json=True, to_polygons=True
                    ),
                )

        if 'kitti360' in  inst_key:
            MetadataCatalog.get(inst_key).set(
                #cindy add :cityscapes_instance2sem_seg  , origin : cityscapes_instance
                source_image_dir=source_image_dir, source_gt_dir=source_gt_dir, target_image_dir=target_image_dir, evaluator_type="kitti360_instance", **meta) #cityscapes_instance

        else:
            MetadataCatalog.get(inst_key).set(
                #cindy add :cityscapes_instance2sem_seg  , origin : cityscapes_instance
                source_image_dir=source_image_dir, source_gt_dir=source_gt_dir, target_image_dir=target_image_dir, evaluator_type="cityscapes_instance", **meta) #cityscapes_instance

    # print('finish register')
            

# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_uda_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)