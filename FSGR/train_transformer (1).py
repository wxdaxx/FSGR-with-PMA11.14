import os
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider

from models.fsgr import TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from models.fsgr.transformer import Transformer
from models.fsgr.optim_entry import build_optimizer, SupConLoss
from torch.cuda.amp import autocast, GradScaler

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import warnings
import logging

# 仅保留错误
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

# === 快速跑通的可选“限步”开关（通过环境变量控制；0 表示不限制） ===
LIMIT_TRAIN = int(os.getenv('LIMIT_TRAIN', '0'))
LIMIT_VAL   = int(os.getenv('LIMIT_VAL',   '0'))
LIMIT_EVAL  = int(os.getenv('LIMIT_EVAL',  '0'))

# 固定性
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Fast eval envs ---
FAST_EVAL = os.getenv("FSGR_FAST_EVAL", "0") != "0"
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

PRINT_INTERVAL = 50


def evaluate_loss(model, dataloader, loss_fn, text_field, e, device, loss_contrast, beta=0.25):
    """验证阶段：仅计算 CE loss；支持限步（FSGR_FAST_EVAL/FSGR_VAL_STEPS 或 LIMIT_VAL）。"""
    model.eval()
    running_loss = 0.0
    step_count = 0

    # 快速评估开关 + 限制步数
    fast_eval = os.getenv("FSGR_FAST_EVAL", "0") != "0"
    max_val_steps = int(os.getenv("FSGR_VAL_STEPS", "0")) if fast_eval else 0

    from tqdm import tqdm
    with tqdm(desc=f'Epoch {e} - validation', unit='it', total=len(dataloader), mininterval=1, disable=False) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                # 统一 4 元素 batch：image_ids, images, placeholder, captions
                if len(batch) != 4:
                    raise ValueError(f"期望4元素batch，但收到{len(batch)}元素")
                _, images, _, captions = batch
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)

                out = model(images, captions)          # (B, L, |V|)
                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()

                ce_loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                running_loss += float(ce_loss.item())
                step_count += 1

                if (it + 1) % PRINT_INTERVAL == 0 or it == len(dataloader) - 1:
                    pbar.set_postfix(loss=round(running_loss / max(1, step_count), 4))
                pbar.update()

                # 两种限步都生效
                if LIMIT_VAL and (it + 1) >= LIMIT_VAL:
                    break
                if max_val_steps and (it + 1) >= max_val_steps:
                    break

    return running_loss / max(1, step_count)
def evaluate_metrics(model, dataloader, text_field, e, device):
    """生成 caption 并计算指标；支持限步（FSGR_FAST_EVAL/FSGR_EVAL_STEPS 或 LIMIT_EVAL）。"""
    model.eval()
    gen, gts = {}, {}

    fast_eval = os.getenv("FSGR_FAST_EVAL", "0") != "0"
    try:
        max_eval_steps = int(os.getenv("FSGR_EVAL_STEPS", "0")) if fast_eval else 0
    except Exception:
        max_eval_steps = 0
    total_steps = max_eval_steps if max_eval_steps > 0 else len(dataloader)

    def _pick_image_tensor(img_pack):
        """从 img_pack 中挑选 [B,3,H,W] 的图像张量"""
        if torch.is_tensor(img_pack):
            if img_pack.ndim == 4 and img_pack.shape[1] in (1, 3):
                return img_pack
        if isinstance(img_pack, (list, tuple)):
            for x in img_pack:
                if torch.is_tensor(x) and x.ndim == 4 and x.shape[1] in (1, 3):
                    return x
        raise ValueError(f"[evaluate_metrics] 在 {type(img_pack)} 中找不到图像张量")

    from tqdm import tqdm
    with torch.no_grad():
        with tqdm(desc=f'Epoch {e} - evaluation', unit='it', total=total_steps, mininterval=1, disable=False) as pbar:
            for it, batch in enumerate(dataloader):
                # 典型结构：((detections, _), caps_gt) 或 (detections, caps_gt)
                if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                    raise ValueError(f"[evaluate_metrics] 非预期batch结构: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

                left, caps_gt_batch = batch
                detections = _pick_image_tensor(left).to(device, non_blocking=True)

                out, _ = model.beam_search(
                    detections, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1
                )

                caps_gen = text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(caps_gt_batch, caps_gen)):
                    # 去重空格，保持与原代码一致
                    gen_i = ' '.join([k for k, _ in itertools.groupby(gen_i)])
                    key = f'{it}_{i}'
                    gen[key] = [gen_i]
                    gts[key] = gts_i

                pbar.update()

                if LIMIT_EVAL and (it + 1) >= LIMIT_EVAL:
                    break
                if max_eval_steps and (it + 1) >= max_eval_steps:
                    break

    # 统一分词 & 计算指标（默认禁用 METEOR，除非设置 FSGR_USE_METEOR=1）
    gts_tok = PTBTokenizer.tokenize(gts)
    gen_tok = PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts_tok, gen_tok)
    return scores
def train_xe(model, dataloader, optim, text_field, device, loss_contrast, e, beta=0.25):
    model.train()

    # 打印当前真实 lr
    if len(optim.param_groups) > 1:
        print(f'Backbone lr = {optim.param_groups[0]["lr"]:.6f}, Dec lr = {optim.param_groups[1]["lr"]:.6f}')
    else:
        print(f'lr = {optim.param_groups[0]["lr"]:.6f}')

    running_loss = 0.0
    with tqdm(desc=f'Epoch {e} - train', unit='it', total=len(dataloader), mininterval=1, disable=False) as pbar:
        for it, batch in enumerate(dataloader):
            # 统一4元素batch
            if len(batch) != 4:
                raise ValueError(f"期望4元素batch，但收到{len(batch)}元素")
            image_ids, images, _, captions = batch
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(images, captions)  # [B, seq, vocab]
                # 输出数值有效性检查
                if (not torch.isfinite(out).all()) or torch.isnan(out).any():
                    print(f"[WARN] step {it}: 模型输出存在 NaN/Inf，跳过该batch")
                    optim.zero_grad(set_to_none=True)
                    pbar.update()
                    continue

                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()

                ce_loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                loss = ce_loss  # 无标签时不加 supcon

                # loss 有效性检查
                if (not torch.isfinite(loss)) or torch.isnan(loss):
                    print(f"[WARN] step {it}: loss 非法 (loss={loss.detach().item():.4f})，跳过该batch")
                    optim.zero_grad(set_to_none=True)
                    pbar.update()
                    continue

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # 先反缩放再裁剪，配合 AMP
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optim)
            scaler.update()

            running_loss += float(loss.detach().item())
            if (it + 1) % 50 == 0 or it == len(dataloader) - 1:
                pbar.set_postfix(loss=round(running_loss / (it + 1), 4))
            pbar.update()
            if LIMIT_TRAIN and (it + 1) >= LIMIT_TRAIN: break

    scheduler.step()
    return running_loss / max(1, len(dataloader))


def train_scst(model, dataloader, optim, cider, text_field, e, device):
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()

    print(f'RL lr = {optim.state_dict()["param_groups"][0]["lr"]:.6f}')
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc=f'Epoch {e} - train', unit='it', total=len(dataloader), mininterval=1, disable=False) as pbar:
        for it, ((detections, _), caps_gt) in enumerate(dataloader):
            detections = detections.to(device)

            with autocast():
                outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'], beam_size, out_size=beam_size)

                caps_gen = text_field.decode(outs.view(-1, seq_len))
                caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
                caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
                reward_baseline = torch.mean(reward, -1, keepdim=True)
                loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
                loss = loss.mean()

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()

            if (it + 1) % PRINT_INTERVAL == 0 or it == len(dataloader) - 1:
                avg_loss = running_loss / (it + 1)
                avg_reward = running_reward / (it + 1)
                avg_baseline = running_reward_baseline / (it + 1)
                pbar.set_postfix(
                    loss=round(avg_loss, 4),
                    reward=round(avg_reward, 4),
                    reward_baseline=round(avg_baseline, 4)
                )
                pbar.update(PRINT_INTERVAL if (it + 1) % PRINT_INTERVAL == 0 else len(dataloader) % PRINT_INTERVAL)
            else:
                pbar.update()

    scheduler_rl.step()
    tokenizer_pool.close()
    tokenizer_pool.join()
    return running_loss / len(dataloader), running_reward / len(dataloader), running_reward_baseline / len(dataloader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if not torch.cuda.is_available():
        print("警告：未检测到CUDA设备，将使用CPU训练（速度极慢）")

    parser = argparse.ArgumentParser(description='Transformer Training')
    parser.add_argument('--exp_name', type=str, default='fsgr_fix')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--return_index', action='store_true')
    parser.add_argument('--adapter_b', type=int, default=6)
    parser.add_argument('--adapter_e', type=int, default=11)
    parser.add_argument('--beta', type=float, default=0.25)

    parser.add_argument('--features_path', type=str, default='./datasets/coco/images/')
    parser.add_argument('--labels_path', type=str, default='../local_text_label_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./m2_annotations')
    parser.add_argument('--text_embed_path', type=str, default='./text_embeddings/ram_ViT16_clip_text.pth')
    parser.add_argument('--pre_vs_path', type=str, default='./.cache/clip/ViT-B-16.pt')
    parser.add_argument("--pre_name", type=str, default='ViT-B/16')
    parser.add_argument('--logs_folder', type=str, default='./tensorboard_logs')
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=100)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)

    parser.add_argument('--xe_base_lr', type=float, default=2e-4)
    parser.add_argument('--rl_base_lr', type=float, default=1e-5)
    args = parser.parse_args()

    print(f"实验参数: exp_name={args.exp_name}, batch_size={args.batch_size}, xe_base_lr={args.xe_base_lr}")

    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # 数据
    image_field = ImageDetectionsField(
        detections_path=args.features_path,
        labels_path=args.labels_path if args.return_index else None,
        max_detections=49,
        load_in_tmp=False
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True,
        tokenize='spacy', remove_punctuation=True, nopoints=False
    )
    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    # 词表
    os.makedirs('./vocab_language', exist_ok=True)
    if not os.path.isfile('./vocab_language/vocab.pkl'):
        print("构建词汇表...")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('./vocab_language/vocab.pkl', 'wb'))
    else:
        print('加载已有词汇表...')
        text_field.vocab = pickle.load(open('./vocab_language/vocab.pkl', 'rb'))
        print(f"词汇表大小: {len(text_field.vocab)}")

    # 模型
    adapter_layer_list = [args.adapter_b, args.adapter_e]
    encoder = TransformerEncoder(
        2, 0, text=args.text,
        attention_module=ScaledDotProductAttention,
        attention_module_kwargs={'m': args.m}
    )
    decoder = TransformerDecoderLayer(
        len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>']
    )
    model = Transformer(
        text_field.vocab.stoi['<bos>'], encoder, decoder,
        adapter_layer_list, pre_vs_path=args.pre_vs_path,
        text_emb_path=args.text_embed_path, pre_name=args.pre_name,
        text=args.text, return_index=args.return_index
    ).to(device)

    # 字典数据集（用于评估）
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    cider_train = None  # 如需SCST+CIDEr，需要Java环境
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    # 学习率调度（注意：LambdaLR 的返回是“乘法因子”）
    def lambda_lr(epoch):
        # 用 epoch+1 使得第1个epoch就有0.25倍率
        t = epoch + 1
        if t <= 3:
            return t / 4.0         # 0.25, 0.5, 0.75
        elif t <= 6:
            return 1.0             # 1.0
        elif t <= 12:
            return 0.2
        else:
            return 0.04

    def lambda_lr_rl(epoch):
        t = epoch + 1
        refine_epoch = args.refine_epoch_rl
        if t <= refine_epoch:
            return 1.0
        elif t <= refine_epoch + 3:
            return 0.2
        elif t <= refine_epoch + 6:
            return 0.04
        else:
            return 0.008

    # 优化器&损失
    optim = build_optimizer(model)              # 分组：backbone/decoder
    # === 强制缩放 param group 学习率，避免过大导致 NaN ===
    if len(optim.param_groups) >= 1:
        optim.param_groups[0]['lr'] = args.xe_base_lr * 0.1   # backbone
    if len(optim.param_groups) >= 2:
        optim.param_groups[1]['lr'] = args.xe_base_lr         # decoder

    scheduler = LambdaLR(optim, lambda_lr)
    optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_contrast = SupConLoss(temperature=0.07)
    scaler = GradScaler()

    # 训练状态
    use_rl = False
    best_cider = .0
    best_test_cider = 0.
    patience = 0
    start_epoch = 0

    # 可选：断点恢复
    if args.resume_last or args.resume_best:
        fname = f'./save_models/{args.exp_name}_last.pth' if args.resume_last else './save_models/batch100_25.pth'
        if os.path.exists(fname):
            data = torch.load(fname, map_location=device)
            torch.set_rng_state(data['torch_rng_state'])
            if torch.cuda.is_available() and data['cuda_rng_state'] is not None:
                torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider = data.get('best_cider', .0)
            best_test_cider = data.get('best_test_cider', .0)
            patience = data.get('patience', 0)
            use_rl = data.get('use_rl', False)

            if use_rl:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])
            else:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])

            print(f"从epoch {data['epoch']} 恢复训练，最佳CIDEr: {best_cider:.4f}")

    os.makedirs('./save_models', exist_ok=True)
    print("开始训练...")

    # 训练主循环
    for e in range(start_epoch, start_epoch + 100):
        print(f"\n===== Epoch {e} =====")
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True, num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, device, loss_contrast, e, beta=args.beta)
            writer.add_scalar('data/train_loss', train_loss, e)
            print(f"训练损失: {train_loss:.4f}")
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field, e, device)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
            print(f"训练损失: {train_loss:.4f}, 平均奖励: {reward:.4f}")

        # 验证
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, e, device, loss_contrast, beta=args.beta)
        writer.add_scalar('data/val_loss', val_loss, e)
        print(f"验证损失: {val_loss:.4f}")

        # 指标（需Java环境才能完全跑CIDEr，没装也不影响训练）
        scores_val = evaluate_metrics(model, dict_dataloader_val, text_field, e, device)
        print("验证集指标:", {k: round(v, 4) for k, v in scores_val.items() if k != 'BLEU'})
        print(f"验证集BLEU: {[round(x, 4) for x in scores_val['BLEU']]}")

        scores_test = evaluate_metrics(model, dict_dataloader_test, text_field, e, device)
        print("测试集指标:", {k: round(v, 4) for k, v in scores_test.items() if k != 'BLEU'})
        print(f"测试集BLEU: {[round(x, 4) for x in scores_test['BLEU']]}")

        # 早停/切换RL（保持原逻辑）
        best = val_loss >= best_cider
        if best:
            best_cider = scores_val['CIDEr']
            patience = 0
        else:
            patience += 1

        best_test = scores_test['CIDEr'] >= best_test_cider
        if best_test:
            best_test_cider = scores_test['CIDEr']

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:
                print(f"强制继续XE训练（epoch {e} < {args.xe_least}）")
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
                for _ in range(e - 1):
                    scheduler_rl.step()
                print("切换到RL训练")
            else:
                print("早停触发，结束训练")
                exit_train = True

        if e == args.xe_most and not use_rl:
            use_rl = True
            switch_to_rl = True
            patience = 0
            optim_rl = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
            scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
            for _ in range(e - 1):
                scheduler_rl.step()
            print(f"达到最大XE epoch {args.xe_most}，切换RL")

        if switch_to_rl and not best:
            best_path = f'./save_models/{args.exp_name}_best.pth'
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device)['state_dict'])
                print(f"从最佳模型 {best_path} 恢复")

        # 保存
        save_data = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': scores_val['CIDEr'],
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }
        os.makedirs('./save_models', exist_ok=True)
        torch.save(save_data, f'./save_models/{args.exp_name}_last.pth')

        if best:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_best.pth')
        if best_test:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_best_test.pth')
        if e >= 55:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_{e}.pth')

        if exit_train:
            writer.close()
            break

    writer.close()
    print("训练结束")
