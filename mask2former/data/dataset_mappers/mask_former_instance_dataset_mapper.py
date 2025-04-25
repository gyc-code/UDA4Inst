# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import cv2
import os

import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F


from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask

__all__ = ["MaskFormerInstanceDatasetMapper"]

FAR_REGION_RANDOM = True

def cv_read_exr(exrpath):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    depth = cv2.imread(exrpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # return (depth*100000).astype(int)
    return (depth*100000).astype(float)


class MaskFormerInstanceDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for instance segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __transform_annotation__(self, transforms, dataset_dict, data_key, image):
        assert "annotations" in dataset_dict[data_key]
        for anno in dataset_dict[data_key]["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict[data_key].pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "segmentation" in annos[0]
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(mask_util.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )
        return annos, masks
                        
    def __mask_to_instance__(self, image, annos, masks):
        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # Prepare per-category binary masks
        instances = Instances(image_shape)
        instances.gt_classes = classes
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks = BitMasks(torch.stack(masks))
            instances.gt_masks = masks.tensor
        return  instances

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if "source" in dataset_dict:
            for data_key in dataset_dict.keys():
                image = utils.read_image(dataset_dict[data_key]["file_name"], format=self.img_format)
                image_for_far = copy.deepcopy(image)
                utils.check_image_size(dataset_dict[data_key], image)
                aug_input = T.AugInput(image)
                aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
                image = aug_input.image
                # introduce depth label
                # depth_map = None # TODO make depth for target
                # # TODO: temply using source depth for poth
                # depth_file_name = dataset_dict["source"]["file_name"].replace('/rgb_translated_cityscapes/', '/depth/').replace('_beauty_', '_depth_').replace('.png','.exr')
                # depth_map = cv_read_exr(depth_file_name)

                if data_key == "source":
                    depth_file_name = dataset_dict[data_key]["file_name"].replace('/rgb_translated_cityscapes/', '/depth/').replace('_beauty_', '_depth_').replace('.png','.exr')
                    depth_map = cv_read_exr(depth_file_name)
                    # depth_map = (np.round(depth_map, decimals=0)).astype(int)
                    far_region = depth_map > 70

                elif data_key == "target":
                    ''' depth map from depth anything, it is a relative depth , near depth is about 200 and far away is close to 0'''
                    depth_path = '/home/yguo/Documents/other/UDA4Inst/datasets/cityscapes/cityscapes_depth_depth_anything/'
                    image_id = dataset_dict[data_key]["image_id"].replace('leftImg8bit.png','leftImg8bit_depth.npy')
                    # TODO , THIS IS relative depth, change to use metric depth, by meter
                    depth_file_name = depth_path + image_id
                    depth_map = np.load(depth_file_name)
                    # depth_map = (1 / (depth_map+0.000001)) * 1000
                    ''' we use 20/30 as the threshold of far away region'''
                    far_region = depth_map < 30 # TODO : use real depth
                
                # to find max_row_index, mean_col_index
                true_indices = np.where(far_region)
                far_region_transforms = None
                if len(true_indices[0]) > 0:
                    max_row_index = np.max(true_indices[0])

                    if data_key == "target":
                        mean_col_index = true_indices[1][np.argmax(true_indices[0])] # pseudo depth is not regular, use lowest point
                    else:
                        mean_col_index = np.max(true_indices[1]) - np.min(true_indices[1]) # synthetic depth is regular

                    ''' design a crop box with 300*300, and center is the lowest row of the far region'''
                    crop_x0 = max(mean_col_index - 150, 0)
                    crop_y0 = max(max_row_index - 150, 0)
                    crop_x1 = min(mean_col_index + 150, dataset_dict[data_key]["width"])
                    crop_y1 = max(max_row_index + 150, dataset_dict[data_key]["height"])
                    # cv2.imwrite('source_' + dataset_dict[data_key]["image_id"] + '_far.png',((far_region*1)*255))
                    far_region_transforms = copy.deepcopy(transforms)

                    for index, t in enumerate(far_region_transforms):
                        # print(t.__class__.__name__)
                        if 'Crop' in t.__class__.__name__:
                            t.y0 = crop_y0
                            t.x0 = crop_x0
                            t.w = crop_x1 - crop_x0
                            t.y = crop_y1 - crop_y0
                        if 'Resize' in t.__class__.__name__:
                            continue
                        far_region_image = t.apply_image(image_for_far) # if true_indices[0] is none, t is not changed, so far_region_image is the same as image processed.
                else:
                    far_region_image = image
                    far_region_transforms = transforms
                for index, t in enumerate(transforms):
                    # print(t.__class__.__name__)
                    if 'Color' in t.__class__.__name__:
                        continue
                    depth_map = t.apply_image(depth_map)

                # Pad image here!
                image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                try:
                    far_region_image = torch.as_tensor(np.ascontiguousarray(far_region_image.transpose(2, 0, 1)))
                except:
                    print('...')
                
                if self.size_divisibility > 0:
                    image_size = (image.shape[-2], image.shape[-1])
                    padding_size = [
                        0,
                        self.size_divisibility - image_size[1],
                        0,
                        self.size_divisibility - image_size[0],
                    ]
                    # pad image
                    image = F.pad(image, padding_size, value=128).contiguous()  
                    far_region_image = F.pad(far_region_image, padding_size, value=128).contiguous()
                    depth_map = F.pad(depth_map, padding_size, value=128).contiguous()


                if data_key == "target":
                    template_mask = utils.read_image('template/cityscapes_ego_car_template.png', format=self.img_format)
                    for index, t in enumerate(transforms):
                        # print(t.__class__.__name__)
                        if 'Color' in t.__class__.__name__:
                            continue
                        template_mask = t.apply_image(template_mask)
                    template_mask = torch.as_tensor(np.ascontiguousarray(template_mask.transpose(2, 0, 1)))
                    if self.size_divisibility > 0:
                        template_mask = F.pad(template_mask, padding_size, value=128).contiguous()
                    dataset_dict["target"]['template_img'] = template_mask
                else:
                    # transform instance masks
                    dataset_dict_bp = copy.deepcopy(dataset_dict) # TODO : MAY BE REMOVED
                    annos, masks = self.__transform_annotation__(transforms, dataset_dict, data_key, image)
                    # if far_region_transforms is not None:
                    far_region_annos, far_region_masks = self.__transform_annotation__(far_region_transforms, dataset_dict_bp, data_key, far_region_image)
                    far_region_masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in far_region_masks]

                    # Pad image and segmentation label here!
                    # image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
                    masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

                    if self.size_divisibility > 0:
                        # pad mask
                        masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]
                        if far_region_transforms is not None:
                            far_region_masks = [F.pad(x, padding_size, value=0).contiguous() for x in far_region_masks]

                    instances = self.__mask_to_instance__(image, annos, masks)
                    dataset_dict[data_key]["instances"] = instances
                    # if far_region_transforms is not None:
                    far_region_instances = self.__mask_to_instance__(far_region_image, far_region_annos, far_region_masks)
                    dataset_dict[data_key]["far_region_instances"] = far_region_instances

                # print('image shape : ', image.shape)
                dataset_dict[data_key]["image"] = image
                dataset_dict[data_key]["far_region_image"] = far_region_image #  TODO   DEBUG
                dataset_dict[data_key]['depth'] = depth_map
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image

            # transform instnace masks
            assert "annotations" in dataset_dict
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            if len(annos):
                assert "segmentation" in annos[0]
            segms = [obj["segmentation"] for obj in annos]
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

            classes = [int(obj["category_id"]) for obj in annos]
            classes = torch.tensor(classes, dtype=torch.int64)

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                # pad image
                image = F.pad(image, padding_size, value=128).contiguous()
                # pad mask
                masks = [F.pad(x, padding_size, value=0).contiguous() for x in masks]

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image

            # Prepare per-category binary masks
            instances = Instances(image_shape)
            instances.gt_classes = classes
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances
        # print(dataset_dict['source']['image'].shape, dataset_dict['target']['image'].shape)
        return dataset_dict
