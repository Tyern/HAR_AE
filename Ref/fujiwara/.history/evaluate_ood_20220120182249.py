"""
evaluate OOD detection performance through AUROC score

Example:
    python evaluate_cifar_ood.py --dataset FashionMNIST_OOD \
            --ood MNIST_OOD,ConstantGray_OOD \
            --resultdir results/fmnist_ood_vqvae/Z7K512/e300 \
            --ckpt model_epoch_280.pkl \
            --config Z7K512.yml \
            --device 1
"""
import os
import yaml
import argparse
import copy
import torch
import numpy as np
from torch.utils import data
from models import get_model, load_pretrained
from loaders import get_dataloader

from utils import roc_btw_arr, batch_run, search_params_intp, parse_unknown_args, parse_nested_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultdir', type=str, help='result dir. results/... or pretrained/...')
    parser.add_argument('--config', type=str, help='config file name')
    parser.add_argument('--ckpt', type=str, help='checkpoint file name to load. default', default=None)
    parser.add_argument('--ood', type=str, help='list of OOD datasets, separated by comma')
    parser.add_argument('--device', type=str, help='device')
    parser.add_argument('--dataset', type=str, choices=['MNIST_OOD', 'CIFAR10_OOD', 'ImageNet32', 'FashionMNIST_OOD',
                                                        'FashionMNISTpad_OOD'],
                        default='MNIST', help='inlier dataset dataset')
    parser.add_argument('--aug', type=str, help='pre-defiend data augmentation', choices=[None, 'CIFAR10', 'CIFAR10-OE'])
    parser.add_argument('--method', type=str, choices=[None, 'likelihood_regret', 'input_complexity', 'outlier_exposure'])
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)


    # load config file
    cfg = yaml.load(open(os.path.join(args.resultdir, args.config)), Loader=yaml.FullLoader)
    result_dir = args.resultdir
    if args.ckpt is not None:
        ckpt_file = os.path.join(result_dir, args.ckpt)
    else:
        raise ValueError(f'ckpt file not specified')

    l_ood = [s.strip() for s in args.ood.split(',')]
    device = f'cuda:{args.device}'

    print(f'loading from : {ckpt_file}')


    def evaluate(m, in_dl, out_dl, device):
        """computes OOD detection score"""
        in_pred = batch_run(m, in_dl, device, method='predict')
        out_pred = batch_run(m, out_dl, device, method='predict')
        auc = roc_btw_arr(out_pred, in_pred)
        return auc


    # load dataset
    print('ood datasets')
    print(l_ood)

#データセットに応じた画像サイズ・チャネル数を指定する
    if args.dataset in {'MNIST_OOD', 'FashionMNIST_OOD'}:
        size = 28
        channel = 1
    else:
        size = 32
        channel = 3
    data_dict = {'path': 'datasets',
                'size': size,
                'channel': channel,
                'batch_size': 64,
                'n_workers': 0, #元は4
                'split': 'evaluation',
                'path': 'datasets'}


# Datasetはtransformsを制御して
# DataLoaderはDatasetを制御する

# DataLoderはデータセットかデータをバッチに固める
# それぞれの「バッチ」には「各画像の集合」とそれに対応する「ラベルをtensor」で保持している
#   batch -> [画像の集合tensor, ラベルtensor[3, 8, 8, 0, 6, 6, 1, 6, 3, 1, 0]]
# それぞれの画像の集合にはRGBの画像が複数枚ある


#引数 --dataset　からDataLoderを得る (1つ)
    data_dict_ = copy.copy(data_dict)
    data_dict_['dataset'] = args.dataset
    in_dl = get_dataloader(data_dict_)
    for batch in in_dl:
        print(batch[0].shape)
        break

#引数 --ood　からDataLoderを得る(複数)
    l_ood_dl = []
    for ood_name in l_ood:
        data_dict_ = copy.copy(data_dict)
        data_dict_['dataset'] = ood_name 
        dl = get_dataloader(data_dict_)
        l_ood_dl.append(dl)

    model = get_model(cfg).to(device)
    gpu_id = 0
    ckpt_data = torch.load(ckpt_file, map_location='cuda:'+str(gpu_id)) #学習済みモデルをGPUに配置
    if 'model_state' in ckpt_data:
        model.load_state_dict(ckpt_data['model_state'])
    else:
        model.load_state_dict(torch.load(ckpt_file, map_location='cuda:'+str(gpu_id)))

    model.eval() #評価モードに設定
    model.to(device)

    in_pred = batch_run(model, in_dl, device=device, no_grad=False)

    l_ood_pred = []
    for dl in l_ood_dl:
        out_pred = batch_run(model, dl, device=device, no_grad=False)
        l_ood_pred.append(out_pred)

    l_ood_auc = []
    for pred in l_ood_pred:
        l_ood_auc.append(roc_btw_arr(pred, in_pred))

    print('OOD Detection Results in AUC')
    for ds, auc in zip(l_ood, l_ood_auc):
        print(f'{ds}:{auc:.4f}')



