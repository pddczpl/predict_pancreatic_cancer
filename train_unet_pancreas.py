"""
train_unet_pancreas.py

# Dice + BCE (cân bằng tổng thể)
python tcia/train_unet_pancreas.py --train_csv ./tcia/cropped/train.csv --val_csv ./tcia/cropped/val.csv `
  --in_channels 1 --epochs 20 --batch_size 4 --lr 1e-3 `
  --out_dir ./checkpointsdicebce --loss dicebce --bce_weight 0.3 --amp --num_workers 4 --patience 8

# Weighted Dice + BCE (cân bằng tổng thể)
python tcia/train_unet_pancreas.py --train_csv ./tcia/cropped/train.csv --val_csv ./tcia/cropped/val.csv `
  --in_channels 1 --epochs 20 --batch_size 4 --lr 1e-3 `
  --out_dir ./checkpointsweighted --loss weighted_dicebce --bce_weight 0.3 --pos_weight 15.0 --amp --num_workers 4 --patience 8

# Tversky thuần (ưu tiên recall nhẹ)
python tcia/train_unet_pancreas.py --train_csv ./tcia/cropped/train.csv --val_csv ./tcia/cropped/val.csv `
  --in_channels 1 --epochs 20 --batch_size 4 --lr 1e-3 `
  --out_dir ./checkpointstversky --amp --num_workers 4 --patience 8 `
  --loss tversky --alpha 0.7 --beta 0.3

# Focal Tversky (tập trung mạnh hơn vào vùng khó/nhỏ)
python tcia/train_unet_pancreas.py --train_csv ./tcia/cropped/train.csv --val_csv ./tcia/cropped/val.csv `
  --in_channels 1 --epochs 20 --batch_size 4 --lr 1e-3 `
  --out_dir ./checkpointsfocal --amp --num_workers 4 --patience 8 `
  --loss focal_tversky --alpha 0.7 --beta 0.3 --gamma 0.75
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import the Dataset util
from pancreas_split_and_dataset import PancreasSliceDataset
def set_seed(seed=42):
    """Thiết lập random seed cho tất cả thư viện"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Cho multi-GPU
    
    # Đảm bảo deterministic behavior trên GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Worker init function cho DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# -----------------------------
# UNet building blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
         )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.bottom = Down(base_ch * 8, base_ch * 16)
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottom(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# -----------------------------
# Losses & Metrics
# -----------------------------
class WeightedDiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.4, pos_weight=15.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Weighted BCE
        pos_weight = torch.tensor([self.pos_weight]).to(logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        
        # Dice Loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        inter = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        dice_loss = 1.0 - dice
        
        total_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        return total_loss

class DiceBCELoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        inter = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        dice_loss = 1.0 - dice

        loss = self.w * bce + (1 - self.w) * dice_loss
        if not torch.isfinite(loss):
            # fallback nhỏ để không nổ loop
            return bce
        return loss

class TverskyLoss(torch.nn.Module):
    """
    Tversky = TP / (TP + alpha*FP + beta*FN)
    alpha > beta -> phạt FP nhiều hơn; beta > alpha -> ưu tiên recall.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - tversky.mean()
        return loss

class TverskyBCELoss(torch.nn.Module):
    """ Kết hợp Tversky + BCE để ổn định giai đoạn đầu """
    def __init__(self, alpha=0.7, beta=0.3, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.w = bce_weight
    def forward(self, logits, targets):
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.tversky(logits, targets)

class FocalTverskyLoss(torch.nn.Module):
    """
    Focal Tversky: (1 - Tversky)^gamma
    gamma > 1 tập trung vào vùng khó/nhỏ (foreground hiếm).
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 0.0, 1.0)
        targets = torch.clamp(targets, 0.0, 1.0)

        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1.0 - t)).sum(dim=1)
        fn = ((1.0 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal = torch.pow(1.0 - tversky, self.gamma).mean()
        return focal

@torch.no_grad()
def eval_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    # Dice
    inter = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = ((2*inter + eps) / (union + eps)).mean().item()
    # IoU
    inter_iou = (preds * targets).sum(dim=(1,2,3))
    union_iou = (preds + targets - preds*targets).sum(dim=(1,2,3))
    iou = ((inter_iou + eps) / (union_iou + eps)).mean().item()
    return dice, iou

@torch.no_grad()
def soft_dice_metric(logits, targets, eps: float = 1e-7):
    """Soft-Dice dùng xác suất (không threshold)"""
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, 0.0, 1.0)
    targets = torch.clamp(targets, 0.0, 1.0)

    p = probs.view(probs.size(0), -1)
    t = targets.view(targets.size(0), -1)

    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = ((2.0 * inter + eps) / (denom + eps)).mean().item()
    return dice  # float

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, amp: bool):
    model.train()
    run_loss, n = 0.0, 0

    use_amp = bool(amp and scaler is not None)
    autocast_dtype = 'cuda' if device.type == 'cuda' else 'cpu'

    for x, y, _ in loader:
        # dữ liệu an toàn
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # 1) forward (autocast), rồi check logits
            with torch.amp.autocast(autocast_dtype):
                logits = model(x)
            if not torch.isfinite(logits).all():
                # không tính loss để tránh propagate NaN
                continue

            # 2) loss (autocast), rồi check loss
            with torch.amp.autocast(autocast_dtype):
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                continue

            # 3) backward/step với GradScaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # để clip grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            # 1) forward, check logits
            logits = model(x)
            if not torch.isfinite(logits).all():
                continue

            # 2) loss, check loss
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                continue

            # 3) backward/step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        run_loss += float(loss.detach()) * x.size(0)
        n += x.size(0)

    return run_loss / max(n, 1)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    total_soft_dice = 0.0
    n = 0
    skip_in = skip_logits = skip_loss = used = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            skip_in += 1
            continue

        logits = model(x)
        if not torch.isfinite(logits).all():
            skip_logits += 1
            continue

        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            skip_loss += 1
            continue

        # Metrics
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        soft_dice = soft_dice_metric(logits, y)

        eps = 1e-7
        inter = (pred * y).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice = ((2 * inter + eps) / (union + eps)).mean()

        i_union = union - inter
        iou = ((inter + eps) / (i_union + eps)).mean()

        bs = x.size(0)
        total_loss      += float(loss) * bs
        total_dice      += float(dice) * bs
        total_iou       += float(iou)  * bs
        total_soft_dice += float(soft_dice) * bs
        n += bs
        used += 1

    print(f"[VAL] used={used} | skip_in={skip_in} | skip_logits={skip_logits} | skip_loss={skip_loss}")
    # trả về ba giá trị cũ để không phá code gọi
    return (total_loss / max(n, 1), total_dice / max(n, 1), total_iou / max(n, 1)), (total_soft_dice / max(n,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--in_channels', type=int, default=1)
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out_dir', type=str, default='./checkpoints')
    ap.add_argument('--amp', action='store_true', help='Use mixed precision (GPU only)')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=8, help='Early stopping patience (epochs)')

    # New: chọn loss & tham số
    ap.add_argument('--loss', type=str, default='weighted_dicebce', 
               choices=['dicebce', 'tversky', 'focaltversky', 'weighted_dicebce'])
    ap.add_argument('--bce_weight', type=float, default=0.3, help='Weight between BCE and Dice (0–1)')
    ap.add_argument('--pos_weight', type=float, default=15.0)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--beta', type=float, default=0.3)
    ap.add_argument('--gamma', type=float, default=0.75)
    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = ap.parse_args()
    # Thiết lập seed ngay đầu
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Data
    train_ds = PancreasSliceDataset(args.train_csv, augment=True)
    val_ds   = PancreasSliceDataset(args.val_csv, augment=False)

    # Sanity: adjust in_channels suggestion if mismatch
    sample_x, _, _ = train_ds[0]
    if sample_x.shape[0] != args.in_channels:
        print(f"[WARN] --in_channels ({args.in_channels}) != sample channels ({sample_x.shape[0]}). "
              f"Consider setting --in_channels {sample_x.shape[0]}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=(device.type=='cuda'),
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=(device.type=='cuda'),
        worker_init_fn=seed_worker,
        generator=g
    )

    # Model, opt, loss
    model = UNet(in_channels=args.in_channels, base_ch=args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if args.loss == 'dicebce':
        loss_fn = DiceBCELoss(bce_weight=args.bce_weight)
    elif args.loss == 'tversky':
        loss_fn = TverskyLoss(alpha=args.alpha, beta=args.beta)
    elif args.loss == 'weighted_dicebce':
        loss_fn = WeightedDiceBCELoss(bce_weight=args.bce_weight, pos_weight=args.pos_weight)
    else:  # focal_tversky
        loss_fn = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    scaler = torch.amp.GradScaler('cuda') if (args.amp and device.type=='cuda') else None

    best_val = float('inf')
    best_path = Path(args.out_dir) / 'unet_best.pt'
    last_path = Path(args.out_dir) / 'unet_last.pt'
    hist_csv = Path(args.out_dir) / 'history.csv'

    history = []
    epochs_no_improve = 0

    print(f"[INFO] Device: {device} | AMP: {bool(scaler)}")
    print(f"[INFO] Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, amp=bool(scaler))
        (val_loss, val_dice, val_iou), val_soft_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_soft_dice': val_soft_dice,
            'lr': optimizer.param_groups[0]['lr']
        })
        # Save history each epoch
        pd.DataFrame(history).to_csv(hist_csv, index=False)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | "
              f"val_soft_dice={val_soft_dice:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping on val_loss
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch}, best_path)
            print(f"[CKPT] Saved best -> {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[EARLY STOP] No improvement for {args.patience} epochs.")
                break

    # Save last
    torch.save({'model': model.state_dict(), 'epoch': epoch}, last_path)
    elapsed = time.time() - t0
    print(f"[DONE] Elapsed {elapsed/60:.1f} min | Best val_loss={best_val:.4f}")
    print(f"[FILES] Best: {best_path} | Last: {last_path} | History: {hist_csv}")

if __name__ == '__main__':
    main()
