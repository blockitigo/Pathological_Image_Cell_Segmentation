from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv


def test(config_file, checkpoint_file, img_path, img_name):
    model = init_detector(config_file, checkpoint_file, device='cpu')
    result = inference_detector(model, img_path)
    img = mmcv.imread(img_path,
                      channel_order='rgb')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0
    )
    visualizer.show()


def main():
    config_file = 'config_model\config.py'
    checkpoint_file = 'work_dirs\epoch_12.pth'
    img_path = 'data\\coco\\train2017\\Uterus_2613.jpg'
    img_name = 'asd.png'
    model = init_detector(config_file, checkpoint_file, device='cpu')
    result = inference_detector(model, img_path)
    print(result.pred_instances.scores)
    print(result.pred_instances.bboxes)
    print(result.pred_instances.labels)
    # for pred in result:
    #     bbox = pred[:4]  # 边界框坐标 (x_min, y_min, x_max, y_max)
    #     score = pred[4]  # 置信度
    #     label = pred[5]  # 类别标签
    #     print('Bounding Box:', bbox)
    #     print('Confidence Score:', score)
    #     print('Label:', label)

    # print(model)
    # show_result(img_path, result, model.CLASSES)
    # test(config_file, checkpoint_file, img_path, img_name)


if __name__ == '__main__':
    main()
