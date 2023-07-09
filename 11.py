
import torch

# # 定义张量
# bboxes = torch.tensor([[21.857, 83.327, 39.952, 106.61]])

# # 修改张量中的某个元素
# bboxes[0, 2] = 50.0

# # 取得张量中的某个元素
# element = bboxes[0, 1]

# # 对张量进行切片操作
# sliced_tensor = bboxes[:, :2]

# # 打印结果
# print(bboxes)
# print(element)
# print(sliced_tensor)

# # 假设有 scores, boxes, labels 张量
# scores = torch.tensor([0.8, 0.6, 0.9, 0.5])
# boxes = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45], [5, 10, 20, 30], [20, 30, 40, 50]])
# labels = torch.tensor([0, 1, 1, 0])

# # 定义阈值
# threshold = 0.7

# # 通过比较 scores 张量和阈值，获取符合条件的索引
# indices = torch.where(scores > threshold)[0]

# # 根据符合条件的索引，选择对应的 boxes 张量
# filtered_boxes = boxes[indices]

# # 打印结果
# print(filtered_boxes)

# from mmengine.structures import InstanceData
# import torch
# import numpy as np

# img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
# instance_data = InstanceData(metainfo=img_meta)
# instance_data.det_labels = torch.LongTensor([2, 3,5])
# instance_data.det_scores = torch.Tensor([0.8, 0.7,0.5])
# instance_data.bboxes = torch.rand((3, 4))
# print('The length of instance_data is', len(instance_data))  # 2

# #instance_data.bboxes = torch.rand((3, 4))

file='config.py'
old_str="""   test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))"""
new_str="""   test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))"""
import os
def alter(file,old_str,new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)
 
alter(file, old_str, new_str)