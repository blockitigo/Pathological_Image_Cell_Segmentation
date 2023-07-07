from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
import mmcv
# 使用测试函数进行目标检测
def predict(params):
    config_file = params['config_path']
    checkpoint_file = params['epochs']
    img_path = params['img_path']
    img_name = params['filename']
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
    

def main():
    config_file = 'config_model\config.py'
    checkpoint_file = 'work_dirs\epoch_12.pth'
    img_path = 'data\\coco\\train2017\\Uterus_2613.jpg'
    img_name = 'test_result.png'
    test(config_file, checkpoint_file, img_path, img_name)


if __name__ == '__main__':
    main()
