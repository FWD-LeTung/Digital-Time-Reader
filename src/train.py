from ultralytics import YOLO
import os
import sys
import argparse
import numpy as np
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core import logger

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True


def train_model(args, mode='detection'):
    if not os.path.exists(args.data):
        logger.error(f"Critical Error: Data file not found at {args.data}")
        return
    
    try:
        model = YOLO(args.model)
        degrees = 180.0 if mode == 'detection' else  0.0
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            lr0=0.001,
            lrf=0.01,
            project=f'run/{mode}',
            name = args.name,
            exist_ok=True,

            optimizer='AdamW',
            seed=42,
            degrees=degrees,
            scale=0.5,
            mosaic=1.0,
            fliplr=0.0,
            workers=2,
            patience=20
        )
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        raise e


if __name__=="__main__":
    seed_everything(42)
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train Digital Clock Models")
    subparsers = parser.add_subparsers(dest='selection', required=True, help='Choose training mode')

    def add_common_args(sub_p):
        sub_p.add_argument('--data', type=str,  required=True, help='Path to data file')
        sub_p.add_argument('--epochs',  type=int, default=100, help='Number of training epochs')
        sub_p.add_argument('--batch', type=int, default=16, help="Batch size")
        sub_p.add_argument('--device', default='0', help='gpus')

    # Detection
    parser_det = subparsers.add_parser('detection', help='Train Pose Model(Find Clock)')
    
    parser_det.add_argument('--model', type=str, default='yolov8n-pose.pt', help='Base model (pose)')
    parser_det.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser_det.add_argument('--name', default='clock_pose', help='Name of the results folder')
    add_common_args(parser_det)

    # Recognition
    parser_rec = subparsers.add_parser('recognition', help='Train Recognition Model(Read Digits)')
    
    parser_rec.add_argument('--model', type=str, default='yolov8n.pt', help='Base model')
    parser_rec.add_argument('--imgsz', type=int, default=320, help='Input image size')
    parser_rec.add_argument('--name', default='digit_rec', help='Name of the results folder')
    add_common_args(parser_rec)

    args = parser.parse_args()

    train_model(args, mode=args.selection)