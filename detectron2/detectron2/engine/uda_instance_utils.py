import torch
import cv2
import numpy as np
import torch.nn.functional as F
import copy
import random
import matplotlib.pyplot as plt


from skimage import measure
from PIL import Image

from detectron2.structures.masks import polygons_to_bitmask

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import _log_api_usage
from detectron2.structures import PolygonMasks, Instances

__all__ = [
"instance_poly2color_semantic",
"generate_class_mask",
"source_target_mix",
"instance_poly2color_semantic",
"apply_black_instance_on_target",
"source_data_augmentation",
"fillhole",
"transform_instance_annotations",
"filter_pseudo_instance",
]

DEBUG_IMG_FLAG = False
VISUALIZE_POLYGON=False
visual_iter = 1
Target_coefficients = None
Source_coefficients = None

RARE_CLASS_NAMES = [5, 6] # 3 for truck ,bus is 4, train is 5,  motor is 6, bike is 7

def translated_obj_mask(obj_mask,image, dx=50,dy=50):
    ''' dx control col, dy control row,dy > 0, move down, dx > 0, move right'''
    # 获取mask的形状
    rows, cols = obj_mask.shape
    # 创建平移后的mask
    translated_mask = torch.zeros_like(obj_mask, dtype=torch.bool)
    translated_img = copy.deepcopy(image)
    # 确定新的位置
    x_start = max(dx, 0)
    x_end = min(cols, cols + dx)
    y_start = max(dy, 0)
    y_end = min(rows, rows + dy)
    
    orig_x_start = max(-dx, 0)
    orig_x_end = min(cols, cols - dx)
    orig_y_start = max(-dy, 0)
    orig_y_end = min(rows, rows - dy)
    
    translated_mask[y_start:y_end, x_start:x_end] = obj_mask[orig_y_start:orig_y_end, orig_x_start:orig_x_end]
    for c in range(3):
        translated_img[c, y_start:y_end, x_start:x_end] = image[c, orig_y_start:orig_y_end, orig_x_start:orig_x_end]

    return translated_mask, translated_img

def get_cityscapes_labels():
    return [
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)]

def remove_occlussion(base_instances, pasted_instances):
    ''' remove the parts of base_instances coverd by pasted_instances '''
    # creat a mask for all instances for occlusion
    all_pasted_masks = pasted_instances.gt_masks.any(dim=0)
    # use ~ to get the un coverd region
    not_pasted_mask = ~all_pasted_masks
    # only keep uncovered region
    base_instances.gt_masks &= not_pasted_mask

def remove_ego_car_logo(pseudo_instance, template):
    ''' design for cityscapes, crop 1024*1024, remove pseudo label which is ego car head and logo'''
    if len(pseudo_instance._fields) == 0:
        return
    white_mask = template >= 20
    white_mask = (white_mask*1)[0, :, :]
    pred_masks = pseudo_instance.pred_masks
    if len(pred_masks) == 0:
        return
    ''' process every mask , if one mask is covered totally by template, remove it and its score and label'''
    indices_to_remove = []
    for i, mask in enumerate(pred_masks):
        mask_area = mask.sum().item()
        if mask_area == 0:
            continue
        ''' mask:1 is area of instance, white_mask 1 is not ego car log, 0 is logo, 
        if mask is in white_mask, multipy makes it 0.'''
        template_apply_mask = mask * white_mask.cuda()
        template_apply_mask_area = template_apply_mask.sum().item()
        if (template_apply_mask_area/mask_area) < 0.2:
            indices_to_remove.append(i)
    new_pseudo_instance = Instances((pseudo_instance.image_size[0], pseudo_instance.image_size[1]))
    for i in range(len(pseudo_instance)):
        if i not in indices_to_remove:
            if len(new_pseudo_instance._fields) == 0:
                new_pseudo_instance = Instances.cat([pseudo_instance[i], pseudo_instance[i]])
            else:
                new_pseudo_instance = Instances.cat([new_pseudo_instance, pseudo_instance[i]])
    del pseudo_instance
    return new_pseudo_instance[1:]

def rare_class_balance(rare_class_samples, img_to_paste, instance_to_add):
    ''' 
    rare_class_samples : a list , element is map, {'img': img, 'instance': instance}
    img_to_paste: to paste rare instance to the img
    instance_to_add :  to add rare instance to the instance
    '''
    pick_num = int(len(rare_class_samples)/2) if len(rare_class_samples) > 1 else 1
    pick_samples = random.sample(rare_class_samples, pick_num)
    for i, sample in enumerate(pick_samples):
        img = sample['img']
        instance = sample['instance']
        mask = instance.gt_masks[0]
        for c in range(3):
            img_to_paste[c,:][mask] = img[c,:][mask]
        remove_occlussion(instance_to_add, instance)
        instance_to_add = Instances.cat([instance_to_add, instance])
    if len(rare_class_samples) > 10:# control the canditate number 
        del rare_class_samples[:4]
    return instance_to_add, rare_class_samples

def visulize_color_instances(instances):
    height = instances._image_size[0]
    width = instances._image_size[1]
    color_instances = np.zeros((height, width,3), dtype=np.uint8)
    try:
        instance_mask = instances.gt_masks
    except:
        instance_mask = (instances.pred_masks).cpu().bool()
    for i in range(len(instances)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_instances[instance_mask[i,:,:]] = [r,g,b]

    return color_instances

def polyfit(pseudo_instances):
    center_row_list = []
    height_list = []
    for i, pred_mask in enumerate(pseudo_instances.pred_masks):
        foreground_indices = pred_mask.nonzero(as_tuple=False)
        if foreground_indices.size(0) == 0:
            continue
            # raise ValueError("The pred_mask contains no foreground pixels.")
        # 计算中心坐标
        center_row = foreground_indices.float().mean(dim=0)[0].item()
        # print('center : ', foreground_indices.float().mean(dim=0))
        # 计算前景高度
        min_row = foreground_indices[:, 0].min().item()
        max_row = foreground_indices[:, 0].max().item()
        height = max_row - min_row + 1
        center_row_list.append(center_row)
        height_list.append(height)

    if len(center_row_list) > 2:
        # 示例数据：目标高度数组和最低点数组
        height_list = np.array(height_list)
        center_row_list = np.array(center_row_list)
        # 使用NumPy的polyfit函数拟合二次多项式
        Target_coefficients = np.polyfit(height_list, center_row_list, 1)
    else:
        print('use last target_coefficients')
        # 计算前景高度
    min_row = foreground_indices[:, 0].min().item()
    max_row = foreground_indices[:, 0].max().item()
    height = max_row - min_row + 1


def get_object_shift_by_depth_map(obj_mask, obj_depth_map, depth_map_to_paste):
    depths_array = obj_depth_map[obj_mask]
    counts = np.bincount(depths_array)
    obj_depth = np.argmax(counts)
    region_in_paste_img = depth_map_to_paste == obj_depth
    foreground_coords = np.column_stack(np.where(region_in_paste_img))
    if foreground_coords.size == 0:
        print('no this depth in image to paste')
        return 0,0
    depth_center_y_to_paste, depth_center_x_to_paste = foreground_coords.mean(axis=0)

    obj_foreground_coords = np.column_stack(np.where(obj_mask))
    obj_center_y, obj_center_x = obj_foreground_coords.mean(axis=0)
    return int(depth_center_x_to_paste - obj_center_x), int(depth_center_y_to_paste - obj_center_y)

def source_instance_paste_to_target_mix(one_data, local_iter, folder_name, source_rare_class_samples):
    global Target_coefficients
    depth_map_source = one_data['source']['depth'].astype(int)
    depth_map_target = one_data['target']['depth'].astype(int)
    gt_instance = one_data['source']['instances']
    gt_classes = gt_instance.gt_classes
    # gt_polygons = gt_syn.gt_masks
    gt_masks = gt_instance.gt_masks
    source_img = one_data['source']['image']
    _, hs, ws = source_img.shape

    target_img = one_data['target']['image']
    pseudo_instances = one_data['target']['instances']
    # pseudo_instances = pseudo_label['instances']
    file_id = one_data['target']['image_id'].split('.')[0]

    if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
        target_img_vis = target_img.cpu().permute(1,2,0).numpy()
        target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
        cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_target_ori.jpg', target_img_vis)
        instances_img = 255 * np.ones(target_img.shape, dtype=np.uint8)
        gt_color_instances = visulize_color_instances(gt_instance)

    gt_instance_select = Instances((hs, ws))
    THIS_FRAME_HAS_RARE_CLASSES = False

    for i, obj_mask in enumerate(gt_masks):
        instance_size = (obj_mask*1).sum().item()
        if instance_size == 0:
            continue 
        x_shift, y_shift = get_object_shift_by_depth_map(obj_mask, depth_map_source, depth_map_target)
        ''' shift obj_mask'''
        x_shift = random.randint(-150, 150)
        shift_obj_mask, shift_source_image = translated_obj_mask(obj_mask,source_img, dx=x_shift,dy=y_shift)        
        ins = Instances((hs, ws))
        ins.gt_classes = gt_classes[i].view(1)
        ins.gt_masks = shift_obj_mask.view(1, hs, ws)
        ''' gather all big source instances'''
        if len(gt_instance_select._fields) == 0:
            gt_instance_select = Instances.cat([ins, ins]) # first one is redandence
        else:
            gt_instance_select = Instances.cat([gt_instance_select, ins])
        ''' mix the image'''
        if i == 0:
            source_img = torch.from_numpy(cv2.GaussianBlur(source_img.permute(1,2,0).to(torch.uint8).numpy(), (5, 5),0)).permute(2,0,1)
        for c in range(3):
            target_img[c,:][shift_obj_mask] = shift_source_image[c,:][shift_obj_mask]
            if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
                instances_img[c,:][shift_obj_mask] = shift_source_image[c,:][shift_obj_mask]

        # for rare class balance
        if gt_classes[i].item() in RARE_CLASS_NAMES:
            source_rare_class_samples.append({'img':source_img, 'instance':ins})
            THIS_FRAME_HAS_RARE_CLASSES = True

    if len(gt_instance_select._fields) != 0: # if gt_instance_selecthas nothing, no labels mix
        # do class balance
        if not THIS_FRAME_HAS_RARE_CLASSES and len(source_rare_class_samples):
            gt_instance_select, source_rare_class_samples = rare_class_balance(source_rare_class_samples, target_img, gt_instance_select)
    
        pseudo_instances.gt_masks = pseudo_instances.pred_masks.bool().cpu()
        pseudo_instances.gt_classes = pseudo_instances.pred_classes.cpu()
        del pseudo_instances._fields['pred_masks']
        del pseudo_instances._fields['pred_classes']
        del pseudo_instances._fields['pred_boxes']
        del pseudo_instances._fields['scores']
        remove_occlussion(pseudo_instances, gt_instance_select[1:]) # modify pseudo_instances, remove parts of coverd by gt_instance_select
        one_data['source']['instances'] = Instances.cat([pseudo_instances, gt_instance_select[1:]])
        one_data['source']['image'] = target_img
        if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
            color_instances = visulize_color_instances(one_data['source']['instances'])
            target_img_vis = target_img.cpu().permute(1,2,0).numpy()
            source_img_vis = source_img.cpu().permute(1,2,0).numpy()
            target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis = cv2.cvtColor(source_img_vis,cv2.COLOR_BGR2RGB)
            target_img_vis_cp = copy.deepcopy(target_img_vis)
            if VISUALIZE_POLYGON:   
                color_map = get_cityscapes_labels()
                for i in range(one_data['source']['instances'].__len__()):
                    mask = one_data['source']['instances'].gt_masks[i].to(torch.uint8).numpy()
                    class_id = one_data['source']['instances'].gt_classes[i].item()
                    # classes = self._metadata.thing_classes[pred_class]
                    #     class_id = name2label[classes].id
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.drawContours(target_img_vis_cp, [contour], -1, (color_map[class_id][2],color_map[class_id][1], color_map[class_id][0]), 2)  # 绿色轮廓
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_s2t_mix.jpg', target_img_vis_cp)
            instances_img = instances_img.transpose((1,2,0))
            instances_img = cv2.cvtColor(instances_img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_s2t_instance.jpg', instances_img)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_s2t_color_instance.jpg', color_instances)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_source_gt_color_instance.jpg', gt_color_instances)

            city = one_data['source']['file_name'].split('/')[8]
            s_img_name = one_data['source']['file_name'].split('/')[-1]
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_gt_img_'  + city + '_' + s_img_name, source_img_vis)
            del target_img_vis_cp
    return one_data, source_rare_class_samples


def target_instance_paste_to_source_mix(one_data, local_iter, folder_name, target_rare_class_samples=[]):
    gt_instance = one_data['source']['instances']
    gt_classes = gt_instance.gt_classes
    depth_map_source = one_data['source']['depth'].astype(int)
    depth_map_target = one_data['target']['depth'].astype(int)
    gt_masks = gt_instance.gt_masks
    source_img = one_data['source']['image']
    _, hs, ws = source_img.shape

    target_img = one_data['target']['image']
    pseudo_instances = one_data['target']['instances']
    pred_masks = pseudo_instances.pred_masks
    pred_classes = pseudo_instances.pred_classes
    file_id = one_data['target']['image_id'].split('.')[0]

    if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
        t_instances_img = 255 * np.ones(target_img.shape, dtype=np.uint8)

    pred_instance_select = Instances((hs, ws))
    THIS_FRAME_HAS_RARE_CLASSES = False
    for i, obj_mask in enumerate(pred_masks.cpu()):
    # for i in range(gt_masks.shape[0]):
        instance_size = (obj_mask).sum().item()
        if instance_size == 0:
            continue 
        obj_mask = obj_mask.bool()
        x_shift, y_shift = get_object_shift_by_depth_map(obj_mask, depth_map_target, depth_map_source)
        
        ''' shift obj_mask'''
        x_shift = random.randint(-150, 150)
        shift_obj_mask, shift_target_image = translated_obj_mask(obj_mask,target_img, dx=x_shift,dy=y_shift)    

        ins = Instances((hs, ws))
        ins.gt_classes = pred_classes[i].cpu().view(1)
        ins.gt_masks = shift_obj_mask.cpu().view(1, hs, ws)
        ''' gather all big source instances'''
        if len(pred_instance_select._fields) == 0:
            pred_instance_select = Instances.cat([ins, ins]) # first one is redandence
        else:
            pred_instance_select = Instances.cat([pred_instance_select, ins])
        ''' mix the image'''
        for c in range(3):
            source_img[c,:][shift_obj_mask] = shift_target_image[c,:][shift_obj_mask]
            if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
                t_instances_img[c,:][shift_obj_mask] = shift_target_image[c,:][shift_obj_mask]

    if len(pred_instance_select._fields) != 0:
        remove_occlussion(gt_instance, pred_instance_select[1:])
        one_data['source']['instances'] = Instances.cat([gt_instance, pred_instance_select[1:]])
        one_data['source']['image'] = source_img
        if DEBUG_IMG_FLAG or local_iter % visual_iter ==0:
            color_instances = visulize_color_instances(one_data['source']['instances'])
            color_pseudo_instances = visulize_color_instances(pseudo_instances)
            
            target_img_vis = target_img.cpu().permute(1,2,0).numpy()
            source_img_vis = source_img.cpu().permute(1,2,0).numpy()
            target_img_vis = cv2.cvtColor(target_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis = cv2.cvtColor(source_img_vis,cv2.COLOR_BGR2RGB)
            source_img_vis_cp = copy.deepcopy(source_img_vis)
            if VISUALIZE_POLYGON:   
                color_map = get_cityscapes_labels()
                for i in range(one_data['source']['instances'].__len__()):
                    mask = one_data['source']['instances'].gt_masks[i].to(torch.uint8).numpy()
                    class_id = one_data['source']['instances'].gt_classes[i].item()
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.drawContours(source_img_vis_cp, [contour], -1, (color_map[class_id][2],color_map[class_id][1], color_map[class_id][0]), 2)  # 轮廓颜色根据CityScapes 的颜色
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter) + '_t2s_mix.jpg', source_img_vis_cp)
            t_instances_img = t_instances_img.transpose((1,2,0))
            t_instances_img = cv2.cvtColor(t_instances_img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_t2s_instance.jpg' , t_instances_img)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_t2s_color_instance.jpg', color_instances)
            cv2.imwrite(folder_name + file_id + '_' + str(local_iter)+ '_pseudo_color_instance.jpg', color_pseudo_instances)
            del source_img_vis_cp
        del pseudo_instances
    return one_data

