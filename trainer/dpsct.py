import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils.detection_util import get_and_print_results

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()

def entropy_select_topk(p, top_k, label, num_of_local_feature, nat_probs):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
    true_probs_repeat = true_probs.repeat_interleave(num_of_local_feature)

    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(label_repeat.clone().detach().unsqueeze(1)).any(dim=1)
    selected_p = p[~contains_label]
    selected_true_probs = true_probs_repeat[~contains_label]

    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()

    return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1) * (0.0000001 + selected_true_probs) )


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = cfg.model_path

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        # x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features, local_image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.t()
        logits_local = logit_scale * local_image_features @ text_features.T

        return logits, logits_local


@TRAINER_REGISTRY.register()
class DPSCT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.lambda_value = cfg.lambda_value
        self.top_k = cfg.topk

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model_path = cfg.model_path
        self.output_dir = cfg.OUTPUT_DIR

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output, output_local = self.model(image)
                # calculate CoOp loss
                label_onehot = F.one_hot(label, num_classes=self.num_classes)
                probs = F.softmax(output, dim=1)
                true_probs = torch.gather(probs, 1, (label.unsqueeze(1)).long()).squeeze()
                loss = -torch.sum(label_onehot * torch.log(probs + 1e-12), dim=1) * (1.0000001 - true_probs)
                loss_id = loss.mean()

                # calculate OOD regularization loss
                batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                output_local = output_local.view(batch_size * num_of_local_feature, -1)
                loss_en = - entropy_select_topk(output_local, self.top_k, label, num_of_local_feature, probs)

                # calculate total loss
                loss = loss_id + self.lambda_value * loss_en

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output_local = self.model(image)

            # calculate CoOp loss
            label_onehot = F.one_hot(label, num_classes=self.num_classes)
            probs = F.softmax(output, dim=1)
            true_probs = torch.gather(probs, 1, (label.unsqueeze(1)).long()).squeeze()
            loss = -torch.sum(label_onehot * F.log_softmax(output, dim=1), dim=1) * (1.0000001 - true_probs)
            loss_id = loss.mean()

            # calculate OOD regularization loss
            batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
            output_local = output_local.view(batch_size * num_of_local_feature, -1)     
            loss_en = - entropy_select_topk(output_local, self.top_k, label, num_of_local_feature, probs)

            # calculate total loss 
            loss = loss_id + self.lambda_value * loss_en

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_en": self.lambda_value * loss_en.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def train(self, train_loader, id_loader, ood_loader_list, out_datasets):
        self.before_train()
        self.best_result = 0
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.save_model(self.epoch, self.output_dir)
            self.test()
            self.test_ood(train_loader, id_loader, ood_loader_list, out_datasets)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")


        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            if len(output) == 2:
                output = output[0]
            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_ood(self, train_loader, id_loader, ood_loader_list, out_datasets, t=10, f=0.5):
        self.model.eval()
        
        # NumPy版本的softmax函数
        def softmax_numpy(x, axis=1):
            x_max = np.max(x, axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            return e_x / np.sum(e_x, axis=axis, keepdims=True)

        # 辅助函数：特征缩放
        def scale_features(kl_id_norm, kl_ood_norm, in_fea, out_fea):
            x_max = max(kl_id_norm.max(), kl_ood_norm.max())
            x_min = min(kl_id_norm.min(), kl_ood_norm.min())
            target_max = max(np.max(in_fea), np.max(out_fea))
            target_min = min(np.min(in_fea), np.min(out_fea))
            # 添加小常数避免除零
            epsilon = 1e-5  
            # 线性缩放
            kl_id_norm_scaled = (kl_id_norm - x_min) / (x_max - x_min + epsilon) * (target_max - target_min) + target_min
            kl_ood_norm_scaled = (kl_ood_norm - x_min) / (x_max - x_min + epsilon) * (target_max - target_min) + target_min
            
            return kl_id_norm_scaled, kl_ood_norm_scaled
        
        # 单批次计算KL散度
        def compute_kl_efficiently_batch(batch_softmax, class_sim_tensor):
            batch_size = batch_softmax.size(0)
            num_classes = class_sim_tensor.size(0)
            batch_kl = torch.zeros(batch_size, num_classes, device=batch_softmax.device)
            kl_div_fn = torch.nn.KLDivLoss(reduction='none')
            
            for c in range(num_classes):
                class_dist = class_sim_tensor[c].unsqueeze(0).expand(batch_size, -1)
                kl_values = kl_div_fn(torch.log(batch_softmax + 1e-10), class_dist).sum(dim=1)
                batch_kl[:, c] = kl_values
            
            return batch_kl.cpu().numpy()

        # 高效提取特征函数
        def extract_features_efficiently(loader, compute_kl=False, class_sim_tensor=None):
            """高效提取特征，减少内存使用"""
            logits_list = []
            local_softmax_max_list = []
            kl_list = []
            
            for batch_idx, (images, _) in enumerate(tqdm(loader)):
                images = images.cuda()
                global_logits, local_logits = self.model(images)
                
                # 处理全局特征
                batch_logits = global_logits.cpu().numpy()
                logits_list.append(batch_logits)
                
                # 对softmax后的logits计算局部特征最大值
                local_softmax = F.softmax(local_logits / t, dim=-1)
                local_softmax_max = torch.max(local_softmax, dim=2)[0]
                local_softmax_max = torch.max(local_softmax_max, dim=1)[0].cpu().numpy()
                local_softmax_max_list.append(local_softmax_max)
                
                # 如果需要计算KL散度
                if compute_kl and class_sim_tensor is not None:
                    batch_softmax = F.softmax(global_logits / t, dim=-1)
                    batch_kl = compute_kl_efficiently_batch(batch_softmax, class_sim_tensor)
                    kl_list.append(batch_kl)
                
                # 手动清理内存
                del global_logits, local_logits, local_softmax
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 合并结果
            logits = np.concatenate(logits_list, axis=0)
            local_softmax_max = np.concatenate(local_softmax_max_list, axis=0)
            
            if compute_kl and len(kl_list) > 0:
                kl = np.concatenate(kl_list, axis=0)
                return logits, local_softmax_max, kl
            else:
                return logits, local_softmax_max
        
        # 获取训练集类均值
        print('获取训练集类均值...')
        total_logit, labels = [], []

        with torch.no_grad():
            for batch_idx, (images, label) in enumerate(tqdm(train_loader)):
                images = images.cuda()
                global_logits, _ = self.model(images)
                labels.append(label)
                total_logit.append(global_logits.cpu())

        total_logit = torch.cat(total_logit, dim=0)
        labels = torch.cat(labels, dim=0).cpu().numpy()
        num_classes = self.dm.num_classes

        # 应用softmax，计算类均值
        total_logit_softmax = F.softmax(total_logit / t, dim=-1).numpy()
        class_sim_mean = np.array([total_logit_softmax[labels == i].mean(axis=0) for i in range(num_classes)])
        class_sim_tensor = torch.from_numpy(class_sim_mean).float().cuda()
        
        # 获取ID数据集的特征 - 高效方式
        print('获取ID数据集特征...')
        id_logits, id_local_softmax_max, id_kl = extract_features_efficiently(
            id_loader, compute_kl=True, class_sim_tensor=class_sim_tensor
        )
        
        # 计算ID数据集的GLMCM分数
        id_logits_softmax = softmax_numpy(id_logits / t)
        factor = f
        id_glmcm_softmax = np.max(id_logits_softmax, axis=1) + factor * id_local_softmax_max
        
        # 初始化评估指标和结果存储
        results = {
            'glmcm_softmax': {'auroc': [], 'aupr_in': [], 'aupr_out': [], 'fpr': []},
            'glmcm_softmax_kl': {'auroc': [], 'aupr_in': [], 'aupr_out': [], 'fpr': []},
        }
        
        # 存储所有beta值下的性能
        best_params = {
            'glmcm_softmax_kl': {'beta': 0, 'auroc': 0, 'fpr': 1, 'avg_score': -float('inf')},
        }
        
        # 存储每个数据集的最佳参数
        dataset_scores = {ood_name: {
            'glmcm_softmax': {'auroc': 0, 'fpr': 1, 'aupr_in': 0, 'aupr_out': 0},
            'glmcm_softmax_kl': {'beta': 0, 'auroc': 0, 'fpr': 1, 'aupr_in': 0, 'aupr_out': 0},
        } for ood_name in out_datasets}
        
        # 存储beta性能
        beta_performance = {
            'glmcm_softmax_kl': {}
        }
        
        # 对每个OOD数据集进行评估
        for i, out_dataset in enumerate(out_datasets):
            print(f"\n评估OOD数据集 {out_dataset}")
            ood_loader = ood_loader_list[i]
            
            # 获取OOD数据集的特征 - 高效方式
            print('获取OOD数据集特征...')
            ood_logits, ood_local_softmax_max, ood_kl = extract_features_efficiently(
                ood_loader, compute_kl=True, class_sim_tensor=class_sim_tensor
            )
            
            # 计算OOD数据集的GLMCM分数 - 使用预计算的最大值
            ood_logits_softmax = softmax_numpy(ood_logits / t)
            factor = f
            ood_glmcm_softmax = np.max(ood_logits_softmax, axis=1) + factor * ood_local_softmax_max
            
            # 计算GLMCM (softmax)的性能
            print("\nGLMCM (softmax后)的性能:")
            auroc, fpr, aupr_in, aupr_out = get_and_print_results(
                None, id_glmcm_softmax, ood_glmcm_softmax, 
                results['glmcm_softmax']['auroc'], 
                results['glmcm_softmax']['aupr_in'], 
                results['glmcm_softmax']['aupr_out'], 
                results['glmcm_softmax']['fpr']
            )
            dataset_scores[out_dataset]['glmcm_softmax'] = {
                'auroc': auroc, 'fpr': fpr, 'aupr_in': aupr_in, 'aupr_out': aupr_out
            }
            
            # 缩放KL散度特征
            kl_id_norm = np.min(id_kl, axis=1)
            kl_ood_norm = np.min(ood_kl, axis=1)
            
            kl_id_norm_softmax_scaled, kl_ood_norm_softmax_scaled = scale_features(
                kl_id_norm, kl_ood_norm, 
                id_glmcm_softmax, ood_glmcm_softmax
            )
            
            # 初始化当前数据集的最佳参数
            dataset_best = {
                'glmcm_softmax_kl': {'beta': 0, 'auroc': 0, 'fpr': 1, 'score': -float('inf')},
            }
            
            # 搜索beta参数
            print("\n开始beta参数搜索...")
            for beta in tqdm(np.arange(-1.0, 0.1, 0.1), desc="Beta搜索", leave=False):
                beta = round(beta, 1)  # 确保精度
                
                if beta not in beta_performance['glmcm_softmax_kl']:
                    beta_performance['glmcm_softmax_kl'][beta] = {}
                
                # 计算GLMCM(softmax)+KL的性能
                id_score_softmax_kl = 1.0 * id_glmcm_softmax + beta * kl_id_norm_softmax_scaled
                ood_score_softmax_kl = 1.0 * ood_glmcm_softmax + beta * kl_ood_norm_softmax_scaled
                
                auroc, fpr, aupr_in, aupr_out = get_and_print_results(
                    None, id_score_softmax_kl, ood_score_softmax_kl, [], [], [], []
                )
                score = auroc - fpr
                
                beta_performance['glmcm_softmax_kl'][beta][out_dataset] = {
                    'auroc': auroc, 'fpr': fpr, 'aupr_in': aupr_in, 'aupr_out': aupr_out
                }
                
                if score > dataset_best['glmcm_softmax_kl']['score']:
                    dataset_best['glmcm_softmax_kl'] = {
                        'beta': beta, 'auroc': auroc, 'fpr': fpr, 
                        'aupr_in': aupr_in, 'aupr_out': aupr_out, 'score': score
                    }
            
            # 存储当前数据集的最佳参数
            dataset_scores[out_dataset]['glmcm_softmax_kl'] = dataset_best['glmcm_softmax_kl']
            
            # 打印当前数据集的最佳参数
            print(f"\n{out_dataset}的最佳参数:")
            print(f"GLMCM(softmax): AUROC={dataset_scores[out_dataset]['glmcm_softmax']['auroc']:.4f}, FPR={dataset_scores[out_dataset]['glmcm_softmax']['fpr']:.4f}")
            print(f"GLMCM(softmax)+KL: beta={dataset_best['glmcm_softmax_kl']['beta']}, AUROC={dataset_best['glmcm_softmax_kl']['auroc']:.4f}, FPR={dataset_best['glmcm_softmax_kl']['fpr']:.4f}")
        
        # 计算全局最佳参数
        print("\n计算全局最佳参数...")
        for beta in beta_performance['glmcm_softmax_kl'].keys():
            if len(beta_performance['glmcm_softmax_kl'][beta]) == len(out_datasets):
                avg_auroc = np.mean([beta_performance['glmcm_softmax_kl'][beta][ds]['auroc'] for ds in out_datasets])
                avg_fpr = np.mean([beta_performance['glmcm_softmax_kl'][beta][ds]['fpr'] for ds in out_datasets])
                avg_score = avg_auroc - avg_fpr
                
                if avg_score > best_params['glmcm_softmax_kl']['avg_score']:
                    best_params['glmcm_softmax_kl'] = {
                        'beta': beta, 'auroc': avg_auroc, 'fpr': avg_fpr, 'avg_score': avg_score
                    }
        
        # 打印平均性能
        print("\nGLMCM (softmax后)的平均性能:")
        avg_softmax_auroc = np.mean([dataset_scores[ds]['glmcm_softmax']['auroc'] for ds in out_datasets])
        avg_softmax_fpr = np.mean([dataset_scores[ds]['glmcm_softmax']['fpr'] for ds in out_datasets])
        print(f"AUROC: {avg_softmax_auroc:.4f}, FPR: {avg_softmax_fpr:.4f}")
        
        # 打印全局最佳参数
        print("\n全局最佳参数:")
        print(f"GLMCM(softmax)+KL: beta={best_params['glmcm_softmax_kl']['beta']}, AUROC={best_params['glmcm_softmax_kl']['auroc']:.4f}, FPR={best_params['glmcm_softmax_kl']['fpr']:.4f}")
        
        # 打印全局最佳参数下各数据集的性能
        print("\n使用全局最佳beta参数的各数据集性能:")
        for out_dataset in out_datasets:
            global_beta = best_params['glmcm_softmax_kl']['beta']
            if global_beta in beta_performance['glmcm_softmax_kl'] and out_dataset in beta_performance['glmcm_softmax_kl'][global_beta]:
                perf = beta_performance['glmcm_softmax_kl'][global_beta][out_dataset]
                print(f"{out_dataset} - GLMCM(softmax)+KL - beta={global_beta}: AUROC={perf['auroc']:.4f}, FPR={perf['fpr']:.4f}")

        # 计算和打印每个数据集各自最佳性能的平均值
        print("\n各数据集各自最佳性能平均值:")
        best_auroc_avg = np.mean([dataset_scores[ds]['glmcm_softmax_kl']['auroc'] for ds in out_datasets])
        best_fpr_avg = np.mean([dataset_scores[ds]['glmcm_softmax_kl']['fpr'] for ds in out_datasets])
        best_aupr_in_avg = np.mean([dataset_scores[ds]['glmcm_softmax_kl']['aupr_in'] for ds in out_datasets])
        best_aupr_out_avg = np.mean([dataset_scores[ds]['glmcm_softmax_kl']['aupr_out'] for ds in out_datasets])

        print(f"GLMCM(softmax)+KL(各自最佳beta):")
        print(f"平均 AUROC: {best_auroc_avg:.4f}, 平均FPR: {best_fpr_avg:.4f}")      

    @torch.no_grad()
    def test_visualize(self, img_path, label):
        """code for visualization results"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.model_path, device=device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        output, output_local = self.model_inference(image)

        num_regions = output_local.shape[1]
        label = torch.tensor(label).cuda()
        label_repeat = label.repeat_interleave(num_regions)
        output_local = F.softmax(output_local, dim=-1)

        output_local = output_local.view(num_regions, -1)

        # -----top k--------
        pred_topk = torch.topk(output_local, k=self.top_k, dim=1)[1]
        contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

        return contains_label

    @torch.no_grad()
    def plot_ood(self, train_loader, id_loader, ood_loader_list, out_datasets, optimal_params=None, output_dir='./distributions', t=10, f=0.5):
        """绘制OOD检测分布图 - 使用传入的最佳参数"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import numpy as np
        from tqdm import tqdm
        
        self.model.eval()
        
        # NumPy版本的softmax函数
        def softmax_numpy(x, axis=1):
            x_max = np.max(x, axis=axis, keepdims=True)
            e_x = np.exp(x - x_max)
            return e_x / np.sum(e_x, axis=axis, keepdims=True)

        # 辅助函数：特征缩放
        def scale_features(kl_id_norm, kl_ood_norm, in_fea, out_fea):
            x_max = max(kl_id_norm.max(), kl_ood_norm.max())
            x_min = min(kl_id_norm.min(), kl_ood_norm.min())
            target_max = max(np.max(in_fea), np.max(out_fea))
            target_min = min(np.min(in_fea), np.min(out_fea))
            # 添加小常数避免除零
            epsilon = 1e-5  
            # 线性缩放
            kl_id_norm_scaled = (kl_id_norm - x_min) / (x_max - x_min + epsilon) * (target_max - target_min) + target_min
            kl_ood_norm_scaled = (kl_ood_norm - x_min) / (x_max - x_min + epsilon) * (target_max - target_min) + target_min
            
            return kl_id_norm_scaled, kl_ood_norm_scaled
        
        # 单批次计算KL散度
        def compute_kl_efficiently_batch(batch_softmax, class_sim_tensor):
            batch_size = batch_softmax.size(0)
            num_classes = class_sim_tensor.size(0)
            batch_kl = torch.zeros(batch_size, num_classes, device=batch_softmax.device)
            kl_div_fn = torch.nn.KLDivLoss(reduction='none')
            
            for c in range(num_classes):
                class_dist = class_sim_tensor[c].unsqueeze(0).expand(batch_size, -1)
                kl_values = kl_div_fn(torch.log(batch_softmax + 1e-10), class_dist).sum(dim=1)
                batch_kl[:, c] = kl_values
            
            return batch_kl.cpu().numpy()

        # 高效提取特征函数
        def extract_features_efficiently(loader, compute_kl=False, class_sim_tensor=None):
            """高效提取特征，减少内存使用"""
            logits_list = []
            local_softmax_max_list = []
            kl_list = []
            
            for batch_idx, (images, _) in enumerate(tqdm(loader)):
                images = images.cuda()
                global_logits, local_logits = self.model(images)
                
                # 处理全局特征
                batch_logits = global_logits.cpu().numpy()
                logits_list.append(batch_logits)
                
                # 对softmax后的logits计算局部特征最大值
                local_softmax = F.softmax(local_logits / t, dim=-1)
                local_softmax_max = torch.max(local_softmax, dim=2)[0]
                local_softmax_max = torch.max(local_softmax_max, dim=1)[0].cpu().numpy()
                local_softmax_max_list.append(local_softmax_max)
                
                # 如果需要计算KL散度
                if compute_kl and class_sim_tensor is not None:
                    batch_softmax = F.softmax(global_logits / t, dim=-1)
                    batch_kl = compute_kl_efficiently_batch(batch_softmax, class_sim_tensor)
                    kl_list.append(batch_kl)
                
                # 手动清理内存
                del global_logits, local_logits, local_softmax
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 合并结果
            logits = np.concatenate(logits_list, axis=0) if logits_list else np.array([])
            local_softmax_max = np.concatenate(local_softmax_max_list, axis=0) if local_softmax_max_list else np.array([])
            
            if compute_kl and len(kl_list) > 0:
                kl = np.concatenate(kl_list, axis=0)
                return logits, local_softmax_max, kl
            else:
                return logits, local_softmax_max
        
        # 检查参数
        if optimal_params is None:
            optimal_params = {'glmcm_softmax_kl': {'beta': -0.5, 'alpha': 1.0}}
        
        # 获取训练集类均值
        print('获取训练集类均值...')
        total_logit, labels = [], []

        with torch.no_grad():
            for batch_idx, (images, label) in enumerate(tqdm(train_loader)):
                images = images.cuda()
                global_logits, _ = self.model(images)
                labels.append(label)
                total_logit.append(global_logits.cpu())
                
                # 每5个批次清理内存
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

        total_logit = torch.cat(total_logit, dim=0)
        labels = torch.cat(labels, dim=0).cpu().numpy()
        num_classes = self.dm.num_classes

        # 应用softmax，计算类均值
        total_logit_softmax = F.softmax(total_logit / t, dim=-1).numpy()
        class_sim_mean = np.array([total_logit_softmax[labels == i].mean(axis=0) for i in range(num_classes)])
        class_sim_tensor = torch.from_numpy(class_sim_mean).float().cuda()
        
        # 释放不再需要的内存
        del total_logit, total_logit_softmax
        torch.cuda.empty_cache()
        
        # 获取ID数据集的特征
        print('获取ID数据集特征...')
        id_logits, id_local_softmax_max, id_kl = extract_features_efficiently(
            id_loader, compute_kl=True, class_sim_tensor=class_sim_tensor
        )
        
        # 计算ID数据集的GLMCM分数
        id_logits_softmax = softmax_numpy(id_logits / t)
        id_glmcm_softmax = np.max(id_logits_softmax, axis=1) + f * id_local_softmax_max
        
        # 清理内存
        del id_logits_softmax
        torch.cuda.empty_cache()
        
        # 处理KL散度
        kl_id_norm = np.min(id_kl, axis=1)
        
        # 对每个OOD数据集绘制分布图
        for i, out_dataset in enumerate(out_datasets):
            if i > 0:
                break
            print(f"\n处理OOD数据集 {out_dataset}")
            ood_loader = ood_loader_list[i]
            
            # 获取OOD数据集的特征
            print('获取OOD数据集特征...')
            ood_logits, ood_local_softmax_max, ood_kl = extract_features_efficiently(
                ood_loader, compute_kl=True, class_sim_tensor=class_sim_tensor
            )
            
            # 计算OOD数据集的GLMCM分数
            ood_logits_softmax = softmax_numpy(ood_logits / t)
            ood_glmcm_softmax = np.max(ood_logits_softmax, axis=1) + f * ood_local_softmax_max
            
            # 清理内存
            del ood_logits_softmax
            torch.cuda.empty_cache()
            
            # 缩放KL散度特征
            kl_ood_norm = np.min(ood_kl, axis=1)
            kl_id_norm_softmax_scaled, kl_ood_norm_softmax_scaled = scale_features(
                kl_id_norm, kl_ood_norm, 
                id_glmcm_softmax, ood_glmcm_softmax
            )
            
            # 直接使用传入的optimal_params计算分数和绘制图表
            for method_name, params in optimal_params.items():
                alpha = params.get('alpha', 1.0)
                beta = params.get('beta', -0.5)
                
                print(f"使用参数: method={method_name}, alpha={alpha}, beta={beta} 绘制分布图")
                
                # 计算最终分数
                id_scores = alpha * id_glmcm_softmax + beta * kl_id_norm_softmax_scaled
                ood_scores = alpha * ood_glmcm_softmax + beta * kl_ood_norm_softmax_scaled
                
                # 计算性能指标
                # FPR@95%TPR的计算
                threshold = np.percentile(id_scores, 5)  # 5%分位点，使95%的ID样本正确分类
                fpr = (ood_scores >= threshold).mean()  # OOD样本被错分为ID的比例
                
                # 计算AUROC
                from sklearn.metrics import roc_auc_score
                labels = np.zeros(len(id_scores) + len(ood_scores), dtype=np.int32)
                labels[:len(id_scores)] = 1  # ID标签为1
                scores = np.concatenate((id_scores, ood_scores))
                auroc = roc_auc_score(labels, scores)
                
                # 创建输出目录
                method_dir = os.path.join(self.output_dir, output_dir, out_dataset)
                os.makedirs(method_dir, exist_ok=True)
                
                # 文件名
                file_name = f"{method_name}_alpha{alpha}_beta{beta}_auroc{auroc:.4f}_fpr{fpr:.4f}"
                
                # 绘制分布图
                self.plot_distribution(method_dir, id_scores, ood_scores, file_name, fpr)

            # 处理完一个数据集后，清理内存
            del ood_logits, ood_local_softmax_max, ood_kl, ood_glmcm_softmax
            torch.cuda.empty_cache()
        
        # 全部处理完成后，清理剩余内存
        del id_logits, id_local_softmax_max, id_kl, id_glmcm_softmax
        torch.cuda.empty_cache()
                

    def plot_distribution(self, output_dir, id_scores, ood_scores, file_name, fpr):
        """绘制ID和OOD分数的分布"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import numpy as np
        
        sns.set(style="white", palette="muted")
        palette = ['#A8BAE3', '#55AB83']
        
        id_scores = id_scores.flatten()
        ood_scores = ood_scores.flatten()
        # 计算ID分数的5%分位线（使得95%的ID分数大于该阈值）
        threshold = np.percentile(id_scores, 5)
        
        data = {
            "ID": id_scores.tolist(),
            "OOD": ood_scores.tolist()
        }
        
        # 创建画布
        plt.figure(figsize=(10, 6))
        
        # 使用seaborn绘制分布
        g = sns.displot(data, kind="kde", palette=palette, fill=True, alpha=0.8)
        
        for ax in g.axes.flat:
            # 添加阈值线和FPR文本
            x_min, x_max = ax.get_xlim()
            data_range = x_max - x_min
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
            text_offset = data_range * 0.05
            text_x = threshold + text_offset
            ax.text(text_x, ax.get_ylim()[1]*0.9, 
                    f'FPR@95%: {fpr:.4f}', 
                    bbox=dict(facecolor='white', alpha=0))

        # 设置标签
        g.set_axis_labels("Scores", "Density")
        
        # 调整布局
        g.fig.tight_layout()
        
        # 保存图像
        file_path = os.path.join(output_dir, f"{file_name}.png")
        print(f"保存图像到: {file_path}")
        g.savefig(file_path, bbox_inches='tight')
        plt.close()
