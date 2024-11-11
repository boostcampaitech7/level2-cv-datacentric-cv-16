#python train_wandb.py  # 기본 실행
#python train_wandb.py --wandb_run_name "experiment-1"  # 실험 이름 지정하고 싶을 때
"""
# resume 하는 방법
# 노션에 wandb 사용법 페이지를 보면 좀 더 참고할 수 있습니다

python train_wandb_epoch_v2_re.py \
    --resume \
    --checkpoint_path trained_models/epoch_150.pth \
    --wandb_run_id YOUR_WANDB_RUN_ID \
    --max_epoch 300

"""

import sys, os
import os
import os.path as osp
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

def save_top_k_checkpoints(model_dir, new_ckpt_path, new_loss, k=5, epoch=None, max_epoch=None):
    """
    상위 k개의 체크포인트와 마지막 5개의 체크포인트를 유지하는 함수
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        return

    # 체크포인트 목록과 각각의 loss 저장
    ckpt_losses = []
    ckpt_epochs = []
    for ckpt in os.listdir(model_dir):
        if not ckpt.endswith('.pth') or 'best' in ckpt:  # best 체크포인트는 제외
            continue
        ckpt_path = os.path.join(model_dir, ckpt)
        if ckpt_path == new_ckpt_path:
            continue
        try:
            ckpt_data = torch.load(ckpt_path, map_location='cpu')
            loss = ckpt_data.get('loss', float('inf'))
            epoch_num = ckpt_data.get('epoch', -1)
            ckpt_losses.append((loss, ckpt_path))
            ckpt_epochs.append((epoch_num, ckpt_path))
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
            continue

    # 새로운 체크포인트 추가
    ckpt_losses.append((new_loss, new_ckpt_path))
    ckpt_epochs.append((epoch, new_ckpt_path))

    # 유지할 체크포인트 경로들
    keep_paths = set()

    # loss 기준 상위 k개
    ckpt_losses.sort(key=lambda x: x[0])  # loss 오름차순 정렬
    keep_paths.update(path for _, path in ckpt_losses[:k])

    # 마지막 5개의 체크포인트
    if epoch is not None and max_epoch is not None:
        if max_epoch - epoch <= 5:  # 마지막 5개 에폭
            ckpt_epochs.sort(key=lambda x: x[0])  # 에폭 기준 정렬
            keep_paths.update(path for _, path in ckpt_epochs[-5:])

    # keep_paths에 포함되지 않은 체크포인트 삭제
    for loss, ckpt_path in ckpt_losses:
        if ckpt_path not in keep_paths:
            try:
                os.remove(ckpt_path)
                print(f"Removed checkpoint: {ckpt_path} (loss: {loss:.4f})")
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint {ckpt_path}: {e}")
            
        
def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

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
        "optimizer": "AdamW",
        "optimizer_params": {
            "weight_decay": 0.001,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        },
        "scheduler": "OneCycleLR",
        "scheduler_params": {
            "pct_start": 0.3,
            "div_factor": 25,
            "final_div_factor": 1e4,
            "anneal_strategy": 'cos'
        },
        "device": device
    }
    
    # Device 설정
    device = torch.device(device)
    
    # 모델 초기화
    model = EAST()
    model.to(device)
    
    # 데이터셋 설정 (나중에 steps_per_epoch를 위해 미리 설정)
    dataset = SceneTextDataset(
        data_dir,
        split='erased_empty_transcription',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    num_batches = math.ceil(len(dataset) / batch_size)
    
    # 옵티마이저 초기화
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.001,  # 0.01에서 0.001로 변경
        betas=(0.9, 0.999),  # AdamW의 기본값
        eps=1e-8
    )
    
    # 시작 epoch 초기화
    start_epoch = 0
    
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
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=max_epoch - start_epoch,  # 남은 에폭 수
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Initialize wandb with resume
        config["resumed_from_epoch"] = start_epoch
        logger.run = wandb.init(
            project=logger.project_name,
            name=logger.name,
            id=wandb_run_id,
            resume="must",
            config=config
        )
    else:
        # 새로운 학습을 시작할 때의 스케줄러
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=max_epoch,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        logger.initialize(config)

    model.train()
    
    # best_loss 초기화할 때 best_epoch도 같이 초기화
    best_loss = float('inf')
    best_epoch = -1

    
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
                
                loss.backward()
                optimizer.step()
                scheduler.step()  # OneCycleLR은 batch마다 step
                optimizer.zero_grad()

                # 배치 크기를 고려한 loss 누적
                batch_size = img.size(0)  # 실제 배치 크기 (마지막 배치는 더 작을 수 있음)
                loss_val = loss.item() * batch_size
                epoch_loss += loss_val
                epoch_cls_loss += extra_info['cls_loss'] * batch_size if extra_info.get('cls_loss') is not None else 0
                epoch_angle_loss += extra_info['angle_loss'] * batch_size if extra_info.get('angle_loss') is not None else 0
                epoch_iou_loss += extra_info['iou_loss'] * batch_size if extra_info.get('iou_loss') is not None else 0

                pbar.update(1)
                val_dict = {
                    'Loss': f"{loss_val/batch_size:.4f}",
                    'Cls': f"{extra_info['cls_loss']:.4f}", 
                    'Angle': f"{extra_info['angle_loss']:.4f}",
                    'IoU': f"{extra_info['iou_loss']:.4f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.6f}"
                }
                pbar.set_postfix(val_dict)
        
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
                "epoch_time": epoch_time.total_seconds()
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

            # 먼저 현재 체크포인트 저장
            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(ckpt_data, ckpt_fpath)
            
            # Log model checkpoint
            logger.log_model(ckpt_fpath, f'model-epoch-{epoch+1}')
            
            # 상위 k개와 마지막 5개 유지
            save_top_k_checkpoints(
                model_dir, 
                ckpt_fpath, 
                mean_epoch_loss, 
                k=5,
                epoch=epoch+1,
                max_epoch=max_epoch
            )
            
            # 가장 좋은 모델 별도 저장 (에폭 정보 포함)
            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                best_epoch = epoch + 1
                
                # 기존 best 모델 파일 찾아서 삭제
                for f in os.listdir(model_dir):
                    if f.startswith('best_ep') and f.endswith('.pth'):
                        try:
                            os.remove(os.path.join(model_dir, f))
                        except Exception as e:
                            print(f"Warning: Failed to remove old best model: {e}")
                
                # 새로운 best 모델 저장
                best_ckpt_path = osp.join(model_dir, f'best_ep{best_epoch}.pth')
                torch.save(ckpt_data, best_ckpt_path)
                print(f"Saved best model from epoch {best_epoch} with loss: {best_loss:.4f}")

    logger.finish()
    

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
