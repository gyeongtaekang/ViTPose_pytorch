import argparse
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np

from time import time
from PIL import Image
from torchvision.transforms import transforms

# 모델 및 유틸리티 함수 불러오기
from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.dist_util import get_dist_info, init_dist
from utils.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference']  # 외부에서 import 할 수 있도록 inference 함수만 공개

@torch.no_grad()  # PyTorch의 gradient 계산 비활성화 (inference 시 불필요한 연산 방지)
def inference(img_path: Path, img_size: tuple[int, int],
              model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True) -> np.ndarray:
    
    # 모델 불러오기
    vit_pose = ViTPose(model_cfg)  # ViTPose 모델 초기화
    
    # 모델 체크포인트 로드
    ckpt = torch.load(ckpt_path)  # 저장된 모델 가중치 로드
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])  # state_dict가 포함된 경우 해당 가중치 사용
    else:
        vit_pose.load_state_dict(ckpt)  # 그렇지 않으면 전체 가중치 로드
    vit_pose.to(device)  # 모델을 지정된 장치(GPU/CPU)로 이동
    print(f">>> Model loaded: {ckpt_path}")
    
    # 입력 이미지 불러오기 및 전처리
    img = Image.open(img_path)  # 이미지 열기
    org_w, org_h = img.size  # 원본 이미지 크기 가져오기
    print(f">>> Original image size: {org_h} X {org_w} (height X width)")
    print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
    print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")
    
    # 이미지 리사이징 및 텐서 변환
    img_tensor = transforms.Compose([
        transforms.Resize((img_size[1], img_size[0])),  # 모델 입력 크기로 조정
        transforms.ToTensor()
    ])(img).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치로 이동
    
    # 모델에 입력하여 heatmap 예측 수행
    tic = time()
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy()  # 모델 실행 후 결과 가져오기 (N, 17, h/4, w/4)
    elapsed_time = time() - tic  # 실행 시간 계산
    print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
    
    # heatmap에서 키포인트 좌표 추출
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps, 
        center=np.array([[org_w//2, org_h//2]]),  # 이미지 중심 좌표 설정
        scale=np.array([[org_w, org_h]]),  # 스케일 설정
        unbiased=True, use_udp=True  # 정밀한 좌표 계산
    )
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)  # (x, y, 신뢰도) 형태로 변환
    
    # 시각화 및 결과 저장
    if save_result:
        for pid, point in enumerate(points):
            img = np.array(img)[:, :, ::-1]  # RGB → BGR 변환 (OpenCV 호환성)
            img = draw_points_and_skeleton(
                img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                points_palette_samples=10, confidence_threshold=0.4
            )
            save_name = img_path.replace(".jpg", "_result.jpg")  # 결과 이미지 저장 경로 설정
            cv2.imwrite(save_name, img)  # 결과 이미지 저장
    
    return points  # 키포인트 좌표 반환
    

if __name__ == "__main__":
    # 모델 및 데이터 설정 파일 불러오기
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    from configs.ViTPose_base_coco_256x192 import data_cfg
    
    # 명령행 인자 파싱 (이미지 경로 입력 받기)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default='examples/sample.jpg', help='image path(s)')
    args = parser.parse_args()
    
    # 현재 디렉토리 기준 체크포인트 경로 설정
    CUR_DIR = osp.dirname(__file__)
    # CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    CKPT_PATH = r"C:\\Users\\AERO\\Downloads\\ViTPose_pytorch-main\\ViTPose_pytorch-main\\runs\\vitpose-b-multi-coco.pth"
    
    # 데이터 설정에서 이미지 크기 불러오기
    img_size = data_cfg['image_size']
    
    # 이미지 경로가 단일 문자열이 아닌 리스트 형태로 변환
    if type(args.image_path) != list:
         args.image_path = [args.image_path]
    
    # 각 이미지에 대해 추론 수행
    for img_path in args.image_path:
        print(img_path)
        keypoints = inference(
            img_path=img_path, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
            save_result=True
        )
