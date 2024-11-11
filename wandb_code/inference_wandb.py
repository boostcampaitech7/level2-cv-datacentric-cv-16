"""
# 기존 방식 (latest.pth 사용)
python inference_wandb.py --data_dir data --model_dir trained_models

# wandb 코드로 나온 거 특정 체크포인트 지정
python inference_wandb.py --data_dir data --checkpoint_path trained_models/epoch_100.pth
python wandb_code/inference_wandb.py --data_dir data --checkpoint_path trained_models/epoch_100.pth

"""

import sys
import os
import os.path as osp

# 현재 파일의 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL', 'data'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))
    parser.add_argument('--checkpoint_path', type=str, help='Path to specific checkpoint file')
    
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def load_checkpoint(model, checkpoint_path):
    """
    체크포인트를 로드하는 함수. 두 가지 형식 모두 처리 가능
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # wandb 버전의 체크포인트
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 기존 버전의 체크포인트
        model.load_state_dict(checkpoint)
    
    return model

def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='test'):
    model = load_checkpoint(model, ckpt_fpath)
    model.eval()

    image_fnames, by_sample_bboxes = [], []
    images = []
    
    for image_fpath in tqdm(sum([glob(osp.join(data_dir, f'{lang}_receipt/img/{split}/*')) for lang in LANGUAGE_LIST], [])):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result

def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get checkpoint path
    if args.checkpoint_path:
        ckpt_fpath = args.checkpoint_path
    else:
        ckpt_fpath = osp.join(args.model_dir, 'latest.pth')
    
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f'Starting inference using checkpoint: {ckpt_fpath}')

    ufo_result = dict(images=dict())
    split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                              args.batch_size, split='test')
    ufo_result['images'].update(split_result['images'])

    output_fname = 'output_ep140best.csv'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)