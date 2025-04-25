# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import cv2
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.utils import decode_seg_map_sequence
from .evaluator import DatasetEvaluator


def process_train_id_to_color_img(img_train_id, dataset='cityscapes'):
    h, w = img_train_id.shape
    img_train_id = img_train_id.reshape(1, h, w)
    img_train_id = np.repeat(img_train_id, 3, axis=0)
    rgb_masks_final = decode_seg_map_sequence(img_train_id, dataset)
    # rgb_masks_final = decode_seg_map_sequence(img_train_id, dataset)

    rgb_masks_np_final = rgb_masks_final.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    # img_color = cv2.cvtColor(rgb_masks_np_final[0], cv2.COLOR_BGR2RGB)
    return rgb_masks_np_final[0]

def cat_pred_gt(visual_pred_color, file_name):
    gt_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_color.png')
    train_id_path = file_name.replace('/leftImg8bit', '/gtFine').replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
    gt_img = cv2.imread(gt_path)
    rgb_img = cv2.imread(file_name)
    train_id_img = cv2.imread(train_id_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    human_cycle_vehicle_train_id = [11, 12, 13, 14, 15, 16, 17, 18]
    train_id_img_processed = np.where(np.isin(train_id_img, human_cycle_vehicle_train_id), train_id_img, 255)
    mask = train_id_img_processed==255
    mask_huam_cycle_behicle = ~mask * 1
    gt_img = gt_img * mask_huam_cycle_behicle
    black_pixels = np.all(gt_img == [0, 0, 0], axis=-1)
    gt_img[black_pixels] = [255, 255, 255]
    h,w,c = rgb_img.shape
    black_bar_shape = (10, w, c)
    black_bar = np.zeros(black_bar_shape, dtype=np.uint8)
    rgb_blackbar = np.vstack((rgb_img, black_bar))
    rgb_gt = np.vstack((rgb_blackbar, gt_img))
    rgb_gt_blackbar = np.vstack((rgb_gt, black_bar))
    rgb_gt_pred = np.vstack((rgb_gt_blackbar,visual_pred_color))

    return rgb_gt_pred, gt_img


class CityscapesEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using cityscapes API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._visual_dir = tempfile.TemporaryDirectory(prefix="cityscapes_visual_")
        self._temp_dir = self._working_dir.name
        self._temp_visual_dir = self._visual_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )

        self._temp_visual_dir = comm.all_gather(self._temp_visual_dir)[0]
        if self._temp_visual_dir != self._visual_dir.name:
            self._visual_dir.cleanup()
        self._logger.info(
            "Writing cityscapes visual results to temporary directory {} ...".format(self._temp_visual_dir)
        )


class CityscapesInstanceEvaluator(CityscapesEvaluator):
    """
    Evaluate instance segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                # output = output[output.scores > 0.1]  ##  cindy add , filter will decrease ap
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        score = output.scores[i]
                        mask = output.pred_masks[i].numpy().astype("uint8")
                        png_filename = os.path.join(
                            self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                        )
                        # vs_png_filename = os.path.join(
                        #     self._temp_dir, basename + "_{}_{}_{}.png".format(i, classes, str(score.item()))
                        # )

                        Image.fromarray(mask * 255).save(png_filename)
                        # Image.fromarray(mask * 255).save(vs_png_filename)

                        fout.write(
                            "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                        )
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))


        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        try:
            results = cityscapes_eval.evaluateImgLists(
                predictionImgList, groundTruthImgList, cityscapes_eval.args, self._logger
            )["averages"]

            ret = OrderedDict()
            ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
            self._working_dir.cleanup() 
            return ret
        except:
            print('------------ error happen in eval')
            self._working_dir.cleanup() 
            return None


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        self._working_dir.cleanup()
        return ret


class CityscapesInstance2SemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label
        self.process_instance2semantic(inputs, outputs, name2label)
    
    
    def process_instance2semantic(self, inputs, outputs, name2label):
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")
            if "instances" in output:
                output = output["instances"].to(self._cpu_device)
                output = output[output.scores > 0.5]
                # filter score < 0.5
                # pred = 255 * np.ones(output._image_size, dtype=np.uint8) # comment by cindy, for catogory, a lot of background is 255 ???
                pred = 0 * np.ones(output._image_size, dtype=np.uint8)# cindy
                visual_pred = 255 * np.ones(output._image_size, dtype=np.uint8)# cindy
                pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")
                visual_pred_filename = os.path.join(self._temp_visual_dir, basename + "_visual_pred.png")
                
                num_instances = len(output)
                with open(pred_txt, "w") as fout:
                    for i in range(num_instances):
                        pred_class = output.pred_classes[i]
                        classes = self._metadata.thing_classes[pred_class]
                        class_id = name2label[classes].id
                        class_train_id = name2label[classes].trainId
                        score = output.scores[i]

                        if classes == 'person' and  score < 0.9:
                            continue
                        elif classes == 'rider' and score < 0.8:
                            continue
                        elif classes == 'truck' and score < 0.9:
                            continue
                        elif classes == 'bus' and score < 0.9:
                            continue
                        #### mask = output.pred_masks[i].numpy().astype("uint8")
                        mask = output.pred_masks[i].bool()
                        pred[mask] = class_id
                        visual_pred[mask] = class_train_id
                        fout.write(
                            "{} {} {}\n".format(os.path.basename(pred_filename), class_id, score)
                        )   
                    Image.fromarray(pred).save(pred_filename)

                    visual_pred_color = process_train_id_to_color_img(visual_pred)
                    pred_gt, gt_color = cat_pred_gt(visual_pred_color, file_name)
                    print('save', visual_pred_filename)
                    Image.fromarray(pred_gt.astype("uint8")).save(visual_pred_filename)
                    
            else:
                # Cityscapes requires a prediction file for every ground truth image.
                with open(pred_txt, "w") as fout:
                    pass


    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        ret_instance2semaic = self.evaluate_instance2semaic()
        # self._working_dir.cleanup()

        return ret_instance2semaic

    def evaluate_instance2semaic(self):
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval
        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir) # origin
        # gt_dir = PathManager.get_local_path('/home/yguo/Documents/other/detectron2_maskRCNN/datasets/cityscapes/gtFine/val/') # cindy add
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
        assert len(groundTruthImgList), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(cityscapes_eval.args.groundTruthSearch)
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(cityscapes_eval.args, gt))
        results = cityscapes_eval.evaluateImgLists(predictionImgList, groundTruthImgList, cityscapes_eval.args)
        ret = OrderedDict()
        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
            "iIoU": 100.0 * results["averageScoreInstClasses"],
            "IoU_sup": 100.0 * results["averageScoreCategories"],
            "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        }
        return ret
    
