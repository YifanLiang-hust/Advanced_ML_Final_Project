CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot1-seed1 --train --seed 1 --shot 1
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot1-seed2 --train --seed 2 --shot 1
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot1-seed3 --train --seed 3 --shot 1

CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot2-seed1 --train --seed 1 --shot 2
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot2-seed2 --train --seed 2 --shot 2
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot2-seed3 --train --seed 3 --shot 2

CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot4-seed1 --train --seed 1 --shot 4
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot4-seed2 --train --seed 2 --shot 4
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot4-seed3 --train --seed 3 --shot 4

CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot8-seed1 --train --seed 1 --shot 8
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot8-seed2 --train --seed 2 --shot 8
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot8-seed3 --train --seed 3 --shot 8

CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot16-seed1 --train --seed 1 --shot 16
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot16-seed2 --train --seed 2 --shot 16
CUDA_VISIBLE_DEVICES=0 python CoOp_main.py --trainer DPSCT --config-file ./config/trainer/ImageNet_b16.yaml --model_path ./checkpoint/ViT-B-16.pt --output_dir ./output/DPSCT/b16/shot16-seed3 --train --seed 3 --shot 16











