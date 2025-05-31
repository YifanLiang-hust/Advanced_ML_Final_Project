import os
import signal
import argparse
import torch
import torchvision
import numpy as np
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import dataset.imagenet

import trainer.locoop

from torch.utils.data import DataLoader
from torchvision import transforms
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):

    if args.model_path:
        cfg.model_path = args.model_path

    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.shot:
        cfg.DATASET.NUM_SHOTS = args.shot

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.topk:
        cfg.topk = args.topk


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    # cfg.TRAINER.COOP.CSC = True  # class-specific context
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'    
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def set_ood_loader(args, out_dataset, preprocess):

    from torchvision import datasets

    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, "OOD", "iNaturalist"), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, "OOD", "SUN"), transform=preprocess)
    elif out_dataset == 'Places':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, "OOD", "Places"), transform=preprocess)
    elif out_dataset == 'texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, "OOD", "Texture", "images"), transform=preprocess)

    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=16)
    return testloaderOut


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    trainer = build_trainer(cfg)

    import clip_w_local
    _, preprocess = clip_w_local.load(args.model_path)

    if args.train:
        print(f"Start training")
        trainer.train()
    if args.load:
        print(f"Load model from {args.output_dir}")
        trainer.load_model(args.output_dir, epoch=args.load_epoch)
    if args.test:
        print(f"Start testing")
        trainer.test()
    if args.ood:
        print(f"Start OOD detection")
        if cfg.DATASET.NAME == 'ImageNet':
            out_datasets = ['iNaturalist', 'SUN', 'Places', 'texture']
            testset_dir  = os.path.join(args.root, "ID", "ImageNet", "val")

        test_set = torchvision.datasets.ImageFolder(testset_dir, transform=preprocess)
        id_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=16, 
                                    drop_last=False, pin_memory=True)

        label_pre_mcm, label_pre_gl, in_score_mcm, in_score_gl = trainer.test_ood(id_data_loader, args.T)
    
        auroc_list_mcm, aupr_in_list_mcm, aupr_out_list_mcm, fpr_list_mcm = [], [], [], []
        auroc_list_glmcm, aupr_in_list_glmcm, aupr_out_list_glmcm, fpr_list_glmcm = [], [], [], []

        for out_dataset in out_datasets:
            print(f"Evaluting OOD dataset {out_dataset}")
            ood_loader = set_ood_loader(args, out_dataset, preprocess)
            _, _, out_score_mcm, out_score_gl = trainer.test_ood(ood_loader, args.T)

            print("MCM score")
            get_and_print_results(args.output_dir, in_score_mcm, out_score_mcm,
                                auroc_list_mcm, aupr_in_list_mcm, 
                                aupr_out_list_mcm, fpr_list_mcm)

            print("GL-MCM score")
            get_and_print_results(args.output_dir, in_score_gl, out_score_gl,
                                auroc_list_glmcm, aupr_in_list_glmcm, 
                                aupr_out_list_glmcm, fpr_list_glmcm)

            # plot_distribution(args, in_score_mcm, out_score_mcm, out_dataset, score='MCM')
            # plot_distribution(args, in_score_gl, out_score_gl, out_dataset, score='GLMCM')

        print("MCM avg. AUROC:{:.4f}, FPR:{:.4f}, AUPR_IN:{:.4f}, AUPR_OUT:{:.4f}".format(np.mean(auroc_list_mcm), np.mean(fpr_list_mcm), np.mean(aupr_in_list_mcm), np.mean(aupr_out_list_mcm)))
        print("GL-MCM avg. AUROC:{:.4f}, FPR:{:.4f}, AUPR_IN:{:.4f}, AUPR_OUT:{:.4f}".format(np.mean(auroc_list_glmcm), np.mean(fpr_list_glmcm),  np.mean(aupr_in_list_glmcm), np.mean(aupr_out_list_glmcm)))
    print('over')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="perform training")
    parser.add_argument("--load", action="store_true", help="load model")
    parser.add_argument("--test", action="store_true", help="perform testing")
    parser.add_argument("--ood", action="store_true", help="perform ood detection")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--model_path", type=str, 
                        default="./checkpoint/ViT-B-16.pt", 
                        help="path to CLIP checkpoint")
    parser.add_argument("--root", type=str, default="./data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="./output/LoCoOp", help="output directory")
    parser.add_argument("--load_epoch", type=int, default=50, help="load model weights at this epoch for evaluation")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--shot", type=int, default=16, help="number of training samples per class"
    )
    parser.add_argument(
        "--config-file", 
        type=str, 
        default="./config/trainer/ImageNet_b16.yaml", 
        help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="./config/dataset/imagenet.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="LoCoOp", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # parameters for LoCoOp
    parser.add_argument('--lambda_value', type=float, default=0.25,
                        help='weight for regulization loss')
    parser.add_argument('--topk', type=int, default=200,
                        help='topk for extracted OOD regions')
    parser.add_argument('--T', type=float, default=1,
                        help='temperature parameter')

    args = parser.parse_args()
    main(args)
