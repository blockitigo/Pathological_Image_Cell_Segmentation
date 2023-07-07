# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import logging
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
import view_streamlit as vs
import view_streamlit

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# 使用该函数来覆盖原配置文件中的参数
def parse_diy(data):
    view_streamlit.st.write('开始配置自定义参数')
    # 使用传来的参数覆盖配置文件中参数
    cfg = Config.fromfile(data['config']) # 读取config文件
    # 设置数据集信息
    # cfg.data_root=data['data_root']
    # cfg.metainfo=data['metainfo']
    # cfg.train_dataloader.dataset.data_root=data['data_root']
    # cfg.val_dataloader.dataset.data_root=data['data_root']
    # cfg.train_dataloader.dataset.metainfo = data['metainfo']
    # cfg.val_dataloader.dataset.metainfo = data['metainfo']
    # cfg.test_dataloader = cfg.val_dataloader
    # 设置类别数目
    # cfg.model.roi_head.bbox_head.num_classes = len(data['metainfo'])
    # cfg.model.roi_head.mask_head.num_classes = len(data['metainfo'])
    # 不清楚这个参数
    # cfg.val_evaluator.ann_file = cfg.data_root+'/'+'var//dataset.json'
    # cfg.test_evaluator = cfg.val_evaluator
    # 加载与训练模型，先试试不写会不会自动下载⭐
    # cfg.load_from = 'D:\计算机可视化\\bighomework\SEGmentation\mmdetection-main\mmdetection-main/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

    #评价时间间隔
    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.max_epochs=data['epochs']
    cfg.train_dataloader.batch_size=data['batch_size']
    # cfg.train_cfg.val_interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    # cfg.default_hooks.checkpoint.interval = 1
    print(data['learning_rate'])
    cfg.optim_wrapper.optimizer.lr = data['learning_rate'] #/ 8
    # cfg.default_hooks.logger.interval = 10
    # # 
    # cfg.log_processor.window_size=50

    # 图片分辨率
    # cfg.train_pipeline.scale=(1333, 800)

    return cfg

def run_model(data):
    args = parse_args()
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg=parse_diy(data)
    
    cfg.launcher = args.launcher
    # vs.st.write(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    return runner.train()
    # return 
