from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
import torch#张量要用
# 使用测试函数进行目标检测
def predict(params):
    config_file = params['config_path']
    checkpoint_file = params['epochs']
    img_path = params['img_path']
    img_name = params['filename']
    confidence = params['confidence']
    cfg_options = {'model.test_cfg.rcnn.score_thr':confidence}
    model = init_detector(config_file, checkpoint_file, device='cpu',cfg_options=cfg_options)
    result = inference_detector(model, img_path)

    print(result)
    #reviseConfd(confidence,result)#改阈值 tensor从零开始

    img = mmcv.imread(img_path,
                      channel_order='rgb')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    # show the results
    res = './result/'+img_name
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file=res
    )
    return res


def test(config_file, checkpoint_file, img_path, img_name):
    model = init_detector(config_file, checkpoint_file, device='cpu')
    result = inference_detector(model, img_path)
    img = mmcv.imread(img_path,
                      channel_order='rgb')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    # show the results
    res = './result/'+img_name
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0
    )
    visualizer.show()
    
def reviseConfd(confidence,result):
    # filtered_bboxes = torch.tensor([[]])

    print("========================================")
    scores = result.pred_instances.scores
    bboxes = result.pred_instances.bboxes
    masks = result.pred_instances.masks
    labels = result.pred_instances.labels
    #result.pred_instances.scores = scores*2#可以运算，但是数量更改不行
    # 通过比较 scores 张量和阈值，获取符合条件的索引  
    indices = torch.where(scores > confidence)[0]
    # 根据符合条件的索引，选择对应的 boxes 张量
    #filtered_boxes = bboxes[indices]
    filtered_bboxes = bboxes[indices]

    filtered_labels = labels[indices]
    filtered_masks = masks[indices]
    filtered_scores = scores[indices]
    #print(result.pred_instances.scores)
    result.pred_instances.bboxes = filtered_bboxes
    result.pred_instances.labels = filtered_labels
    result.pred_instances.scores = filtered_scores
    result.pred_instances.masks = filtered_masks
    # result.pred_instances=len(filtered_bboxes)
    print(len(result.pred_instances))#56
    print(len(filtered_bboxes))#24 0.5时 最后一张图
    # del result.pred_instances.bboxes
    # result.pred_instances.bboxes = filtered_bboxes
    #print("========================================gt_instances")
    
    #print(result.gt_instances)
    #print("========================================ignored_instances")
    #print(result.ignored_instances)
    # print("========================================bboxes")
    # print(filtered_bboxes)
    # print("========================================labels")
    # print(filtered_labels)
    # print("========================================masks")
    # print(filtered_masks)
    # print("========================================scores")
    # print(filtered_scores)
    #result.pred_instances.bboxes=torch.rand((56, 4))#可以修改不能改变维数
    print(result.pred_instances.bboxes)
    #return 0
    """result里包括
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of model predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        python tools/test.py ./config_model/config.py ./work_dirs/epoch_12.pth --show-score-thr
    """

def main():
    config_file = 'config_model\config.py'
    checkpoint_file = 'work_dirs\epoch_12.pth'
    img_path = 'data\\coco\\train2017\\Uterus_2613.jpg'
    img_name = 'test_result.png'
    test(config_file, checkpoint_file, img_path, img_name)


if __name__ == '__main__':
    main()
