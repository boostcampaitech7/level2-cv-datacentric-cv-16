import sys, os
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
from logger_sweep import WandbLogger

import wandb

def save_top_k_checkpoints(model_dir, new_ckpt_path, new_loss, k=10):
    """
    상위 k개의 체크포인트만 유지하는 함수
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
        try:
            ckpt_data = torch.load(ckpt_path)
            loss = ckpt_data.get('loss', float('inf'))
            ckpt_losses.append((loss, ckpt_path))
        except:
            continue

    # 새로운 체크포인트 추가
    ckpt_losses.append((new_loss, new_ckpt_path))
    
    # loss 기준으로 정렬
    ckpt_losses.sort(key=lambda x: x[0])  # loss 오름차순 정렬

    # 상위 k개 제외한 나머지 삭제
    for loss, ckpt_path in ckpt_losses[k:]:
        try:
            os.remove(ckpt_path)
        except:
            pass

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    # Model args
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=5)

    # Optimizer and scheduler args
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=False)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--pct_start', type=float, default=0.3)

    # Loss weights
    parser.add_argument('--cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--angle_loss_weight', type=float, default=1.0)
    parser.add_argument('--iou_loss_weight', type=float, default=1.0)

    # Wandb args
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def get_optimizer(opt_name, model_params, lr, weight_decay, **kwargs):
    if opt_name == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'radam':
        return torch.optim.RAdam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        # SGD의 세부 파라미터는 고정값 사용
        return torch.optim.SGD(
            model_params, 
            lr=lr, 
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay
        )
    elif opt_name == 'adagrad':
        return torch.optim.Adagrad(
            model_params, 
            lr=lr, 
            lr_decay=0.01,  # learning rate decay 추가
            weight_decay=weight_decay,
            initial_accumulator_value=0.1
        )
    elif opt_name == 'adadelta':
        return torch.optim.Adadelta(
            model_params,
            lr=lr,
            rho=0.9,
            eps=1e-6,
            weight_decay=weight_decay
        )
    raise ValueError(f'Unsupported optimizer: {opt_name}')

# 50에폭 기준으로 인자들을 줄인 상태임 - 에폭 바꾸면 여기도 바꿔줘야 함
def get_scheduler(scheduler_name, optimizer, max_epoch, **kwargs):
    if scheduler_name == 'cosine':
        eta_min = kwargs.get('eta_min', 1e-6)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epoch, 
            eta_min=eta_min
        )
    # MultiStepLR (명확한 구간에서 학습률 감소)
    elif scheduler_name == 'multistep':
        # 50 에폭에 맞춰 milestone 조정
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20, 35],  # 40%, 70% 지점
            gamma=0.1
        )
    # ReduceLROnPlateau (성능 정체시 자동 조정)
    elif scheduler_name == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,  # 더 짧은 patience
            verbose=True
        )
    # OneCycleLR (super-convergence 활용 가능)
    elif scheduler_name == 'one_cycle':
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=max_epoch,
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
            pct_start=0.3,
            div_factor=25,      # 초기 학습률을 max_lr/25로 시작
            final_div_factor=4  # 최종 학습률을 초기 학습률의 1/4로
        )
    # CosineAnnealingWarmRestarts (주기적인 리셋으로 지역 최적해 탈출)
    elif scheduler_name == 'cosine_warm':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,     # 첫 번째 리셋까지의 에폭 수
            T_mult=2,   # 다음 주기는 이전 주기의 2배
            eta_min=1e-6
        )
    raise ValueError(f'Unsupported scheduler: {scheduler_name}')

def do_training(**kwargs):
    # wandb init을 가장 먼저 실행
    wandb.init(
        project="Project-OCR",
        config=kwargs,  # config를 직접 전달
        reinit=True    # 재초기화 허용
    )
    
    # # wandb config에서 값 가져오기
    # if wandb.config:
    #     for key, value in wandb.config.items():
    #         kwargs[key] = value
    
    if wandb.config:
        kwargs.update(wandb.config)  # dict.update 사용

    # Logger는 wandb.init() 이후에 초기화
    logger = WandbLogger(name=kwargs.get('wandb_run_name'))
    logger.initialize(config=kwargs)

    # model_dir 초기화 (선택사항) -> 빼도 될듯
    if os.path.exists(kwargs['model_dir']):
        for f in os.listdir(kwargs['model_dir']):
            if f.endswith('.pth'):
                os.remove(os.path.join(kwargs['model_dir'], f))

    # 데이터셋 설정
    dataset = SceneTextDataset(
        kwargs['data_dir'],
        split='train',
        image_size=kwargs['image_size'],
        crop_size=kwargs['input_size']
    )
    dataset = EASTDataset(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=kwargs['batch_size'],
        shuffle=True,
        num_workers=kwargs['num_workers']
    )
    num_batches = math.ceil(len(dataset) / kwargs['batch_size'])

    # 모델 설정
    device = torch.device(kwargs['device'])
    model = EAST()
    model.to(device)

    # 옵티마이저 설정
    optimizer = get_optimizer(
        kwargs['optimizer_type'],
        model.parameters(),
        kwargs['learning_rate'],
        kwargs['weight_decay'],
        momentum=kwargs.get('momentum', 0.9),
        nesterov=kwargs.get('nesterov', False)
    )

    # 스케줄러 설정
    scheduler = get_scheduler(
        kwargs['scheduler_type'],
        optimizer,
        kwargs['max_epoch'],
        steps_per_epoch=len(train_loader),
        eta_min=kwargs.get('eta_min', 1e-6),
        pct_start=kwargs.get('pct_start', 0.3)
    )

    model.train()
    for epoch in range(kwargs['max_epoch']):
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_angle_loss = 0
        epoch_iou_loss = 0
        epoch_start = time.time()

        with tqdm(total=num_batches) as pbar:
            for batch_idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(train_loader):
                pbar.set_description(
                    f'[Epoch {epoch+1}/{kwargs["max_epoch"]}][{batch_idx+1}/{num_batches}]'
                )

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # batch 단위로 scheduler step을 해야하는 경우
                if kwargs['scheduler_type'] == 'one_cycle':
                    scheduler.step()

                batch_size = img.size(0)
                loss_val = loss.item() * batch_size
                epoch_loss += loss_val
                
                # None 체크를 추가하여 안전하게 처리
                epoch_cls_loss += extra_info.get('cls_loss', 0) * batch_size if extra_info.get('cls_loss') is not None else 0
                epoch_angle_loss += extra_info.get('angle_loss', 0) * batch_size if extra_info.get('angle_loss') is not None else 0
                epoch_iou_loss += extra_info.get('iou_loss', 0) * batch_size if extra_info.get('iou_loss') is not None else 0

                pbar.update(1)
                val_dict = {
                    'Loss': f"{loss_val/batch_size:.4f}",
                    'Cls': f"{extra_info.get('cls_loss', 0):.4f}" if extra_info.get('cls_loss') is not None else "N/A",
                    'Angle': f"{extra_info.get('angle_loss', 0):.4f}" if extra_info.get('angle_loss') is not None else "N/A",
                    'IoU': f"{extra_info.get('iou_loss', 0):.4f}" if extra_info.get('iou_loss') is not None else "N/A",
                    'LR': f"{scheduler.get_last_lr()[0]:.6f}"
                }
                pbar.set_postfix(val_dict)

        if kwargs['scheduler_type'] != 'one_cycle':
            scheduler.step()
        
        # 에포크 끝에서의 metrics 계산
        epoch_time = timedelta(seconds=time.time() - epoch_start)
        dataset_size = len(dataset)
        
        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / dataset_size,
            "cls_loss": epoch_cls_loss / dataset_size,
            "angle_loss": epoch_angle_loss / dataset_size,
            "iou_loss": epoch_iou_loss / dataset_size,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_time": epoch_time.total_seconds()
        }
        logger.log_epoch_metrics(metrics)
        
        # 에포크 단위로 scheduler step을 해야하는 경우
        if kwargs['scheduler_type'] == 'reduce_on_plateau':
            scheduler.step(metrics['loss'])
        elif kwargs['scheduler_type'] != 'one_cycle':  # one_cycle은 이미 batch마다 step 했음
            scheduler.step()

        if (epoch + 1) % kwargs['save_interval'] == 0:
            if not osp.exists(kwargs['model_dir']):
                os.makedirs(kwargs['model_dir'])

            # 체크포인트 경로 설정 (loss 값 포함)
            loss_value = metrics['loss']
            ckpt_name = f'epoch_{epoch+1}_loss_{loss_value:.4f}.pth'
            ckpt_fpath = osp.join(kwargs['model_dir'], ckpt_name)

            # 체크포인트 저장
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
            }, ckpt_fpath)
            
            # 상위 5개만 유지
            save_top_k_checkpoints(kwargs['model_dir'], ckpt_fpath, loss_value, k=5)

            # wandb에 모델 기록
            logger.log_model(ckpt_fpath, f'model-epoch-{epoch+1}')

    logger.finish()

def main():
    args = parse_args()
    do_training(**vars(args))

if __name__ == '__main__':
    main()