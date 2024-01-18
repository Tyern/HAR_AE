
import os
import random
import argparse
from omegaconf import OmegaConf
import numpy as np
from itertools import cycle
import torch
from models import get_model
from trainers import get_trainer, get_logger
from loaders import get_dataloader
from optimizers import get_optimizer
from datetime import datetime
from tensorboardX import SummaryWriter
from utils import save_yaml, search_params_intp, eprint, parse_unknown_args, parse_nested_args
import time

def run(cfg, writer):
    start = time.time()
    """main training function"""
    # Setup seeds
    seed = cfg.get('seed', 1)
    print(f'running with random seed : {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Setup device
    device = cfg.device

    # Setup Dataloader
    d_dataloaders = {}
    for key, dataloader_cfg in cfg['data'].items(): # keyはz32.ymlのindist_trainなど，dataloder_cfgはそれに対応する辞書
        #if 'holdout' in cfg:
            # dataloader_cfg = process_holdout(dataloader_cfg, int(cfg['holdout']))
        print(dataloader_cfg)
        d_dataloaders[key] = get_dataloader(dataloader_cfg)

    # Setup Model
    model = get_model(cfg).to(device)
    trainer = get_trainer(cfg)
    logger = get_logger(cfg, writer)

    # Setup optimizer
    if hasattr(model, 'own_optimizer') and model.own_optimizer: #hasattrでmodelの属性にown_optimizerがあればTrue
        optimizer, sch = model.get_optimizer(cfg['training']['optimizer'])
    elif 'optimizer' not in cfg['training']:
        optimizer = None
        sch = None
    else:
        optimizer, sch = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    
    print(round(time.time() - start,0),'秒')
    
    model, train_result = trainer.train(model, optimizer, d_dataloaders, logger=logger,
                                   logdir=writer.file_writer.get_logdir(), scheduler=sch,
                                   clip_grad=cfg['training'].get('clip_grad', None))
    print(round(time.time() - start,0),'秒')


def process_holdout(dataloader_cfg, holdout):
    """udpate config if holdout option is present in config"""
    if 'LeaveOut' in dataloader_cfg['dataset'] and 'out_class' in dataloader_cfg:
        print('LeaveOut, out_classがcfgに設定できている')
        if len(dataloader_cfg['out_class'] ) == 1:  # indist
            dataloader_cfg['out_class'] = [holdout]
        else:  # ood
            dataloader_cfg['out_class'] = [i for i in range(10) if i != holdout]
    print(dataloader_cfg)
    return dataloader_cfg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='results/')
    parser.add_argument('--run', default=None, help='unique run id of the experiment')
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    # モデルの構造はあらかじめymlファイルに記述しておく
    cfg = OmegaConf.load(args.config)
    if args.device == 'cpu':
        cfg['device'] = f'cpu'
    else:
        cfg['device'] = f'cuda:{args.device}'

    if args.run is None:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        run_id = args.run
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    config_basename = os.path.basename(args.config).split('.')[0] #config_basename = z32
    logdir = os.path.join(args.logdir, config_basename, str(run_id))
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    print(f'config saved as {copied_yml}')

    run(cfg, writer)


