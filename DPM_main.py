import argparse
import torch
import os
import signal
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import warnings
import torchvision
from torchvision import transforms
warnings.filterwarnings("ignore")

import datasets.Benchmark_RGB_RGB
import datasets.Benchmark_RGB_IR
import datasets.Benchmark_RGB_MS
import datasets.Benchmark_RGB_A
import datasets.Benchmark_RGB_Re

import datasets.Benchmark_IR_IR
import datasets.Benchmark_MS_MS
import datasets.Benchmark_A_A
import datasets.Benchmark_Re_Re

import datasets.Benchmark_RGB_RGB_ALL
import datasets.Benchmark_IR_IR_ALL
import datasets.Benchmark_MS_MS_ALL
import datasets.Benchmark_A_A_ALL
import datasets.Benchmark_Re_Re_ALL

import datasets.imagenet

import trainer.DPM_man
import trainer.DPM_F
import trainer.DPM
import trainer.DPM_loss
import trainer.DPM_loss1
import trainer.DPM_loss2
import trainer.DPM_loss3
import trainer.DPM_loss12
import trainer.DPM_loss13
import trainer.DPM_loss123

import trainer.DPM_a
import trainer.DPM_noa
import trainer.DPM_noa_local
import trainer.DPM_arc
import trainer.DPM_focal
# import trainer.DPM_entropy
# import trainer.DPM_cutmix
import trainer.DPM_cutmixood
# import trainer.DPM_cutmixoodbinary
# import trainer.DPM_gridmix
import trainer.DPM_gridmixood
# import trainer.DPM_gridmixoodbinary
import trainer.DPM_linemixood
import trainer.DPM_circlemixood
import trainer.DPM_cornermixood
import trainer.DPM_patchmixood
import trainer.DPM_mixupood
import trainer.DPM_randomcutood
import trainer.DPM_randomcutood_local
import trainer.DPM_randomcutood2
# import trainer.DPM_smooth
# import trainer.DPM_energy
# import trainer.DPM_other
# import trainer.DPM_flood
# import trainer.DPM_hdm
import trainer.DPM_locoop
# import trainer.DPM_cos


from time import sleep

import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path, transform):
        self.transform = transform
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f) 
    def __getitem__(self, idx):

        x = self.data['train'][idx].impath
        y = self.data['train'][idx].label
        img = Image.open(x)
        img = self.transform(img)
        return img, y
    def __len__(self):

        return len(self.data['train'])


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

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.shots:
        cfg.DATASET.NUM_SHOTS = args.shots

    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.dff:
        cfg.dff = args.dff

    if args.num_heads:
        cfg.num_heads = args.num_heads
    

    cfg.loss_en = args.loss_en
    cfg.topk = args.topk
    cfg.loss1 = args.loss1
    cfg.loss2 = args.loss2
    cfg.loss3 = args.loss3
    cfg.loss4 = args.loss4
    cfg.loss1_ood = args.loss1_ood
    cfg.loss2_ood = args.loss2_ood
    cfg.loss3_ood = args.loss3_ood

    cfg.scale = args.scale
    cfg.margin = args.margin
    cfg.gamma = args.gamma


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

    cfg.TRAINER.DPM = CN()
    cfg.TRAINER.DPM.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP_MLC = CN()
    cfg.TRAINER.COOP_MLC.N_CTX_POS = 16
    cfg.TRAINER.COOP_MLC.N_CTX_NEG = 16
    cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = None  # 全部使用可学习的
    # cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT = "a photo of"
    cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT = None
    cfg.TRAINER.COOP_MLC.CSC  = True

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
    # cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    return cfg

def set_ood_loader(args, out_dataset, preprocess):

    from torchvision import datasets

    if out_dataset == 'ImageNetr':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'imagenet-r'), transform=preprocess)
    elif out_dataset == 'cifar10':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'cifar10','test'), transform=preprocess)
    elif out_dataset == 'LSUN':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN_C'), transform=preprocess)
    elif out_dataset == 'LSUN_R':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'LSUN_R'), transform=preprocess)

    elif out_dataset == 'OOD_Easy':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/OOD/OOD_Easy/test", transform=preprocess)
    elif out_dataset == 'OOD_Hard':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/OOD/OOD_Hard/test", transform=preprocess)
    
    elif out_dataset == 'OOD_IR':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/CS-OOD/IR/OOD/test", transform=preprocess)
    elif out_dataset == 'OOD_MS':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/CS-OOD/MSRGB/OOD/test", transform=preprocess)
    elif out_dataset == 'OOD_Re':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/CS-OOD/Resampling_bias/OOD/test", transform=preprocess)
    elif out_dataset == 'OOD_A':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/Benchmark/CS-OOD/Aerial/OOD/test", transform=preprocess)
    
    elif out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/xz_OOD/iNaturalist", transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/xz_OOD/SUN", transform=preprocess)
    elif out_dataset == 'Places':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/xz_OOD/places365", transform=preprocess)
    elif out_dataset == 'texture':
        testsetout = datasets.ImageFolder(root="/home/data/datasets/xz_OOD/dtd/images", transform=preprocess)
    elif out_dataset == 'ImageNet-O':
        testsetout = datasets.ImageFolder(root="/new_data/datasets/OOD/imagenet-o", transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batchsize, shuffle=False, num_workers=16)
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
    
    import clip
    _, preprocess = clip.load(args.model_path)
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    #     transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    # ])

    if cfg.DATASET.NAME == 'Benchmark_RGB_RGB':
        out_datasets = ['OOD_Easy', 'OOD_Hard', 'SUN']
        trainset_dir = "/home/data/datasets/Benchmark/OOD/ID/train"
        trainset =  torchvision.datasets.ImageFolder(trainset_dir, transform=preprocess)
        testset_dir  = "/home/data/datasets/Benchmark/OOD/ID/test"
    elif cfg.DATASET.NAME == 'Benchmark_RGB_IR':   
        out_datasets = ['OOD_IR', 'SUN'] 
        trainset_dir = "/home/data/datasets/Benchmark/OOD/ID/train"
        trainset =  torchvision.datasets.ImageFolder(trainset_dir, transform=preprocess)
        testset_dir  = "/home/data/datasets/Benchmark/CS-OOD/IR/ID/test"
    elif cfg.DATASET.NAME == 'Benchmark_RGB_MS':
        out_datasets = ['OOD_MS', 'SUN']
        trainset_dir = "/home/data/datasets/Benchmark/OOD/ID/train"
        trainset =  torchvision.datasets.ImageFolder(trainset_dir, transform=preprocess)
        testset_dir  = "/home/data/datasets/Benchmark/CS-OOD/MSRGB/ID/test"
    elif cfg.DATASET.NAME == 'Benchmark_RGB_Re':
        out_datasets = ['OOD_Re', 'OOD_Easy', 'OOD_Hard', 'SUN']
        trainset_dir = "/home/data/datasets/Benchmark/OOD/ID/train"
        trainset =  torchvision.datasets.ImageFolder(trainset_dir, transform=preprocess)
        testset_dir  = "/home/data/datasets/Benchmark/CS-OOD/Resampling_bias/ID/test"
    elif cfg.DATASET.NAME == 'Benchmark_RGB_A':
        out_datasets = ['OOD_A', 'SUN']
        trainset_dir = "/home/data/datasets/Benchmark/OOD/ID/train"
        trainset =  torchvision.datasets.ImageFolder(trainset_dir, transform=preprocess)
        testset_dir  = "/home/data/datasets/Benchmark/CS-OOD/Aerial/ID/test"
        
        
    elif cfg.DATASET.NAME == 'ImageNet':
        out_datasets = ['iNaturalist', 'SUN', 'Places', 'texture']
        trainset_dir = "/home/data/xz2002/DPM/data/ImageNet/split_fewshot/shot_16-seed_1.pkl"
        trainset_dir = args.pkl_dir
        trainset = MyDataset(trainset_dir, preprocess)
        testset_dir  = "/home/data/datasets/ImageNet/val"

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    testset = torchvision.datasets.ImageFolder(testset_dir, transform=preprocess)
    id_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    ood_loader_list = []
    for out_dataset in out_datasets:
        ood_loader = set_ood_loader(args, out_dataset, preprocess)
        ood_loader_list.append(ood_loader)
    
    if args.train:
        print(f"Start training")
        # trainer.train(train_loader, id_loader, ood_loader_list, out_datasets)
        trainer.train()
    if args.load:
        print(f"Load model from {args.output_dir}")
        trainer.load_model(args.output_dir, epoch=args.load_epoch)
    if args.test:
        print(f"Start testing")
        trainer.test()
    if args.ood:
        print(f"Start OOD detection")
        trainer.test_ood(train_loader, id_loader, ood_loader_list, out_datasets)
    if args.plot:
        import os
        print(f"Start OOD plot")
        plot_param = {
            'fea1_mls': {'alpha': args.plot_fea1_mls_alpha, 'beta': args.plot_fea1_mls_beta},
            'fea2_mls': {'alpha': args.plot_fea2_mls_alpha, 'beta': args.plot_fea2_mls_beta},
            'fea3_mls': {'alpha': args.plot_fea3_mls_alpha, 'beta': args.plot_fea3_mls_beta},
            'fea1_mcm': {'alpha': args.plot_fea1_mcm_alpha, 'beta': args.plot_fea1_mcm_beta},
            'fea2_mcm': {'alpha': args.plot_fea2_mcm_alpha, 'beta': args.plot_fea2_mcm_beta},
            'fea3_mcm': {'alpha': args.plot_fea3_mcm_alpha, 'beta': args.plot_fea3_mcm_beta}
        }
        plot_output_dir = os.path.join(args.output_dir, 'distribution')
        trainer.plot_ood(train_loader, id_loader, ood_loader_list, out_datasets, optimal_params=plot_param, t=100, output_dir=plot_output_dir)
    if args.tsne:
        import os
        print(f"Start t-SNE plot")
        plot_output_dir = os.path.join(args.output_dir, 'tsne')
        rgb_dir = "/home/data/datasets/Benchmark/OOD/ID/test"
        rgb = torchvision.datasets.ImageFolder(rgb_dir, transform=preprocess)
        rgb_loader = DataLoader(rgb, batch_size=args.batch_size, shuffle=False, num_workers=16)
        # trainer.draw_tsne(id_loader, ood_loader_list, out_datasets, output_dir=plot_output_dir)
        trainer.draw_tsne_prototype(plot_output_dir)
        # trainer.draw_tsne_combined(id_loader, ood_loader_list, out_datasets, output_dir=plot_output_dir)
        # trainer.draw_tsne_multi(rgb_loader, id_loader, ood_loader_list, out_datasets, output_dir=plot_output_dir)
    if args.vis:
        print(f"Start visualization")
        import glob
        import os
        from tqdm import tqdm
        
        # 要处理的数据集目录列表
        image_dirs = [
            # "/home/data/datasets/Benchmark/OOD/ID/test",
            # "/home/data/datasets/Benchmark/OOD/OOD_Easy/test",
            # "/home/data/datasets/Benchmark/OOD/OOD_Hard/test",
            # "/home/data/datasets/xz_OOD/SUN"
            "/home/data/datasets/Benchmark/CS-OOD/Aerial/ID/test",
            "/home/data/datasets/Benchmark/CS-OOD/Aerial/OOD/test",
            "/home/data/datasets/Benchmark/CS-OOD/MSRGB/ID/test",
            "/home/data/datasets/Benchmark/CS-OOD/MSRGB/OOD/test",
            "/home/data/datasets/Benchmark/CS-OOD/IR/ID/test",
            "/home/data/datasets/Benchmark/CS-OOD/IR/OOD/test",
        ]
        
        # 对应的输出目录名称
        output_subdirs = [
            # "ID",
            # "OOD-Easy",
            # "OOD-Hard",
            # "SUN"
            "Aerial-ID",
            "Aerial-OOD",
            "MSRGB-ID",
            "MSRGB-OOD",
            "IR-ID",
            "IR-OOD",
        ]
        
        # 基本输出路径
        base_output_dir = os.path.join(args.output_dir, 'vis')
        
        # 批处理大小
        batch_size = 512  # 根据您的GPU内存调整
        
        # 遍历每个数据集目录
        for img_dir, out_subdir in zip(image_dirs, output_subdirs):
            # 创建该数据集的输出目录
            vis_output_dir = os.path.join(base_output_dir, out_subdir)
            os.makedirs(vis_output_dir, exist_ok=True)
            
            print(f"Processing directory: {img_dir} → {vis_output_dir}")
            
            # 收集此目录中的所有图像文件
            image_files = []
            
            # 遍历目录树
            for root, dirs, files in os.walk(img_dir):
                # 保留目录结构信息
                rel_path = os.path.relpath(root, img_dir)
                curr_out_dir = os.path.join(vis_output_dir, rel_path) if rel_path != '.' else vis_output_dir
                os.makedirs(curr_out_dir, exist_ok=True)
                
                # 获取图像文件
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    matches = glob.glob(os.path.join(root, ext))
                    matches.extend(glob.glob(os.path.join(root, ext.upper())))
                    
                    for img_path in matches:
                        image_files.append((img_path, curr_out_dir))
            
            if not image_files:
                print(f"No images found in {img_dir}")
                continue
                
            print(f"Found {len(image_files)} images in {img_dir}")
            
            # 批量处理图像，提高效率
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i+batch_size]
                batch_paths = [item[0] for item in batch]
                batch_dirs = [item[1] for item in batch]
                
                # 显示当前处理的批次和路径
                current_folder = os.path.basename(os.path.dirname(batch_paths[0]))
                print(f"Processing batch {i//batch_size + 1}/{(len(image_files)+batch_size-1)//batch_size} from {current_folder}")
                
                # 批量处理
                trainer.visualize(batch_paths, batch_dirs, batch_size=batch_size)
            
            print(f"Completed processing {len(image_files)} images for {out_subdir}")
            
        print("All visualizations complete!")
    print('over')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", action="store_true", help="use res connection")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--ood_batchsize", type=int, default=512, help="ood batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--dff", type=int, default=2048, help="hidden dimension of the feedforward network")
    parser.add_argument("--pkl_dir", type=str, default="/home/data/xz2002/DPM/data/ImageNet/split_fewshot/shot_16-seed_1.pkl", help="pkl file path")
    parser.add_argument("--train", action="store_true", help="perform training, when last epoch finished then test")
    parser.add_argument("--load", action="store_true", help="load model")
    parser.add_argument("--test", action="store_true", help="perform testing")
    parser.add_argument("--ood", action="store_true", help="perform ood testing")
    parser.add_argument("--plot", action="store_true", help="perform ood plot")
    parser.add_argument("--tsne", action="store_true", help="perform tsne plot")
    parser.add_argument("--vis", action="store_true", help="perform visualization")
    parser.add_argument("--shots", type=int, default=16, help="number of shots")
    parser.add_argument("--model_path", type=str, default="/new_data/xz/clip_checkpoint/OriginCLIP/ViT-B-16.pt", help="pretrained model path")
    parser.add_argument("--root", type=str, default='/home/data/datasets', help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--load_epoch", type=int, default=20, help="load epoch")
    parser.add_argument(
        "--seed", type=int, default=1555, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--loss1", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss2", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss3", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss4", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss1_ood", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss2_ood", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss3_ood", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--loss_en", type=float, default=0.25, help=""
    )
    parser.add_argument(
        "--topk", type=int, default=20, help="top k"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="DPM_OOD",help="name of trainer")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--scale", type=float, default=0.0, help="scale factor for arcloss"
    )
    parser.add_argument(
        "--margin", type=float, default=0.0, help="margin for arcloss"
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="gamma for arcloss"
    )
    # 添加这些参数
    # parser.add_argument("--plot_fea1_mls_alpha", type=float, default =  5.6, help="alpha value for fea1_mls")
    # parser.add_argument("--plot_fea1_mls_beta",  type=float, default = -4.3, help="beta value for fea1_mls")
    # parser.add_argument("--plot_fea2_mls_alpha", type=float, default =  7.5, help="alpha value for fea2_mls")
    # parser.add_argument("--plot_fea2_mls_beta",  type=float, default = -5.6, help="beta value for fea2_mls")
    # parser.add_argument("--plot_fea3_mls_alpha", type=float, default =  9.2, help="alpha value for fea3_mls")
    # parser.add_argument("--plot_fea3_mls_beta",  type=float, default = -8.1, help="beta value for fea3_mls")
    # parser.add_argument("--plot_fea1_mcm_alpha", type=float, default =  9.3, help="alpha value for fea1_mcm")
    # parser.add_argument("--plot_fea1_mcm_beta",  type=float, default = -4.4, help="beta value for fea1_mcm")
    # parser.add_argument("--plot_fea2_mcm_alpha", type=float, default =  6.7, help="alpha value for fea2_mcm")
    # parser.add_argument("--plot_fea2_mcm_beta",  type=float, default = -2.2, help="beta value for fea2_mcm")
    # parser.add_argument("--plot_fea3_mcm_alpha", type=float, default =  10.0, help="alpha value for fea3_mcm")
    # parser.add_argument("--plot_fea3_mcm_beta",  type=float, default = -9.1, help="beta value for fea3_mcm")

    parser.add_argument("--plot_fea1_mls_alpha", type=float, default =  0, help="alpha value for fea1_mls")
    parser.add_argument("--plot_fea1_mls_beta",  type=float, default =  -10, help="beta value for fea1_mls")
    parser.add_argument("--plot_fea2_mls_alpha", type=float, default =  0, help="alpha value for fea2_mls")
    parser.add_argument("--plot_fea2_mls_beta",  type=float, default =  -10, help="beta value for fea2_mls")
    parser.add_argument("--plot_fea3_mls_alpha", type=float, default =  0, help="alpha value for fea3_mls")
    parser.add_argument("--plot_fea3_mls_beta",  type=float, default =  -10, help="beta value for fea3_mls")
    parser.add_argument("--plot_fea1_mcm_alpha", type=float, default =  0, help="alpha value for fea1_mcm")
    parser.add_argument("--plot_fea1_mcm_beta",  type=float, default =  -10, help="beta value for fea1_mcm")
    parser.add_argument("--plot_fea2_mcm_alpha", type=float, default =  0, help="alpha value for fea2_mcm")
    parser.add_argument("--plot_fea2_mcm_beta",  type=float, default =  -10, help="beta value for fea2_mcm")
    parser.add_argument("--plot_fea3_mcm_alpha", type=float, default =  0, help="alpha value for fea3_mcm")
    parser.add_argument("--plot_fea3_mcm_beta",  type=float, default =  -10, help="beta value for fea3_mcm")
    args = parser.parse_args()

    main(args)