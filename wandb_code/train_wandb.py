"""
# 기본 실행
python wandb_code/train_wandb.py

# wandb 실험 이름 지정 (권장)
python wandb_code/train_wandb.py --wandb_run_name "ex1"

실행할 때 resume 코드에만 옵티마이저와 스케줄러를 수정한 것은 아닌지 확인할 것!
의도한 것과 다른 하이퍼 파라미터를 받아갈 수도 있다.


# resume 하는 방법
- 노션에 wandb 사용법 페이지를 보면 좀 더 참고할 수 있습니다

python train_wandb.py \
    --resume \
    --checkpoint_path trained_models/epoch_150.pth \
    --wandb_run_id YOUR_WANDB_RUN_ID \
    --max_epoch 300

"""

import sys, os
import os
import os.path as osp

# 현재 파일의 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from logger_epoch import WandbLogger

import wandb

import numpy as np
import random

np.random.seed(16)
random.seed(16)


def save_top_k_checkpoints(model_dir, new_ckpt_path, new_loss, k=10):
    """
    상위 k개의 체크포인트만 유지하는 함수
    Args:
        model_dir: 체크포인트가 저장되는 디렉토리
        new_ckpt_path: 새로 저장된 체크포인트 경로
        new_loss: 새 체크포인트의 loss 값
        k: 유지할 체크포인트 개수
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return

    # 체크포인트 목록과 각각의 loss 저장
    ckpt_losses = []
    for ckpt in os.listdir(model_dir):
        if not ckpt.endswith('.pth'):
            continue
        ckpt_path = os.path.join(model_dir, ckpt)
        if ckpt_path == new_ckpt_path:  # 새로 저장된 체크포인트는 건너뛰기
            continue
        try:
            ckpt_data = torch.load(ckpt_path, map_location='cpu')
            loss = ckpt_data.get('loss', float('inf'))
            ckpt_losses.append((loss, ckpt_path))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
            continue

    # 새로운 체크포인트 추가
    ckpt_losses.append((new_loss, new_ckpt_path))
    
    # loss 기준으로 정렬
    ckpt_losses.sort(key=lambda x: x[0])  # loss 오름차순 정렬

    # 상위 k개 제외한 나머지 삭제
    for loss, ckpt_path in ckpt_losses[k:]:
        try:
            os.remove(ckpt_path)
            print(f"Removed checkpoint: {ckpt_path} (loss: {loss:.4f})")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint {ckpt_path}: {e}")
            

# 현재 스크립트의 디렉토리를 기준으로 절대 경로 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, '..', 'trained_models')
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
        
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', DEFAULT_DATA_DIR))
    parser.add_argument('--model_dir', type=str, 
                       default=os.environ.get('SM_MODEL_DIR', DEFAULT_MODEL_DIR))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Name for the wandb run (optional)')
    
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--wandb_run_id', type=str, help='Wandb run ID to resume')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, 
                wandb_run_name, resume=False, checkpoint_path=None, wandb_run_id=None):
    
    # Initialize wandb logger
    logger = WandbLogger(name=wandb_run_name)
    
    # Initialize wandb with config
    config = {
        "learning_rate": learning_rate,
        "max_epoch": max_epoch,
        "batch_size": batch_size,
        "image_size": image_size,
        "input_size": input_size,
        "optimizer": "AdamW",  # 실제 사용하는 옵티마이저로 수정
        "scheduler": "CosineAnnealingLR",
        "device": device
    }
    
    # Device 설정
    device = torch.device(device)
    
    # 모델 초기화
    model = EAST()
    model.to(device)
    
    # 옵티마이저 초기화
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 시작 epoch 초기화
    start_epoch = 0
    
    # 기존의 logger.initialize(config) 부분을 아래 코드로 대체
    if resume:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint path must be provided for resume training")
                
        # Load checkpoint
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Adjust scheduler for remaining epochs
        start_epoch = checkpoint['epoch']
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epoch - start_epoch,  # 남은 에폭 수만큼만 설정
            eta_min=1e-6,
            last_epoch=-1
        )
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Initialize wandb with resume
        config["resumed_from_epoch"] = start_epoch
        logger.run = wandb.init(
            project=logger.project_name,
            name=logger.name,
            id=wandb_run_id,  # 기존 run ID 사용
            resume="must",  # must로 설정하여 반드시 기존 run을 이어가도록 함
            config=config
        )
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch,  # 전체 에포크 수
            eta_min=1e-6      # 최소 학습률, 원하는 값으로 조정 가능
        )
        logger.initialize(config)
    
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    model.train()
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, max_epoch):
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_angle_loss = 0
        epoch_iou_loss = 0
        epoch_start = time.time()
        
        with tqdm(total=num_batches) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                # 현재 에폭과 배치 진행상황을 보여주는 description
                pbar.set_description(
                    f'[Epoch {epoch+1}/{max_epoch}][{batch_idx+1}/{num_batches}]'
                )

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 배치 크기를 고려한 loss 누적
                batch_size = img.size(0)  # 실제 배치 크기 (마지막 배치는 더 작을 수 있음)
                loss_val = loss.item() * batch_size
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss'] * batch_size
                epoch_angle_loss += extra_info['angle_loss'] * batch_size
                epoch_iou_loss += extra_info['iou_loss'] * batch_size

                pbar.update(1)
                val_dict = {
                    'Loss': f"{loss_val/batch_size:.4f}",
                    'Cls': f"{extra_info['cls_loss']:.4f}", 
                    'Angle': f"{extra_info['angle_loss']:.4f}",
                    'IoU': f"{extra_info['iou_loss']:.4f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.6f}"
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
        
        # 에폭 완료 시간 계산
        epoch_time = timedelta(seconds=time.time() - epoch_start)
        
        # 전체 데이터셋 크기로 나누어 평균 계산
        dataset_size = len(dataset)
        mean_epoch_loss = epoch_loss / dataset_size
        mean_cls_loss = epoch_cls_loss / dataset_size
        mean_angle_loss = epoch_angle_loss / dataset_size
        mean_iou_loss = epoch_iou_loss / dataset_size

        # 에폭 단위로 모든 메트릭 로깅
        logger.log_epoch_metrics(
            {
                "epoch": epoch,
                "loss": epoch_loss / num_batches,
                "cls_loss": epoch_cls_loss / num_batches,
                "angle_loss": epoch_angle_loss / num_batches,
                "iou_loss": epoch_iou_loss / num_batches,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time": timedelta(seconds=time.time() - epoch_start).total_seconds()
            }
        )

        print(f'Epoch {epoch+1}/{max_epoch} | Mean loss: {mean_epoch_loss:.4f} | Time: {epoch_time}')

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            # 체크포인트 데이터 준비
            ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': mean_epoch_loss,
            }

            #먼저 현재 체크포인트 저장
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(ckpt_data, ckpt_fpath)
            
            # Log model checkpoint
            logger.log_model(ckpt_fpath, f'model-epoch-{epoch+1}')
            
            # 상위 k개만 유지
            save_top_k_checkpoints(model_dir, ckpt_fpath, mean_epoch_loss, k=7)
            
            
            # 가장 좋은 모델 별도 저장
            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                best_ckpt_path = osp.join(model_dir, 'best.pth')
                torch.save(ckpt_data, best_ckpt_path)
                print(f"Saved best model with loss: {best_loss:.4f}")

    logger.finish()
    

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
