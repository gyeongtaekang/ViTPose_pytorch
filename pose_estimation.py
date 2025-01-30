import argparse
import os
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np

from time import time
from PIL import Image
from torchvision.transforms import transforms

# 모델 및 유틸리티 불러오기
from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.top_down_eval import keypoints_from_heatmaps

###############################################################
# 기존 이미지 추론(inference) 함수
###############################################################
@torch.no_grad()
def inference(
    img_path: Path,
    img_size: tuple[int, int],
    model_cfg: dict,
    ckpt_path: Path,
    device: torch.device,
    save_result: bool = True
) -> np.ndarray:
    """
    단일 이미지에 대한 사람 포즈 추론 함수.
    1. 이미지 로드 및 리사이즈
    2. 모델을 통해 히트맵 예측
    3. 히트맵에서 키포인트 좌표 추출
    4. 시각화 및 결과 이미지 저장
    """
    # 1) 모델 초기화 및 가중치 로드
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    vit_pose.eval()

    # 2) 이미지 전처리
    img = Image.open(img_path)
    org_w, org_h = img.size
    img_tensor = transforms.Compose([
        transforms.Resize((img_size[1], img_size[0])),
        transforms.ToTensor()
    ])(img).unsqueeze(0).to(device)

    # 3) 모델 추론(히트맵 생성)
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy()

    # 4) 히트맵에서 (x, y) 좌표 + confidence 추출
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w, org_h]]),
        unbiased=True,
        use_udp=True
    )
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)  # (x, y, score)

    # 5) 시각화 및 결과 저장
    if save_result:
        vis_img = np.array(img)[:, :, ::-1]  # RGB -> BGR
        for pid, point in enumerate(points):
            vis_img = draw_points_and_skeleton(
                vis_img.copy(),
                point,
                joints_dict()['coco']['skeleton'],
                person_index=pid,
                points_color_palette='gist_rainbow',
                skeleton_color_palette='jet',
                points_palette_samples=10,
                confidence_threshold=0.4
            )
        save_name = str(img_path).replace('.jpg', '_result.jpg')
        cv2.imwrite(save_name, vis_img)

    return points

###############################################################
# 새롭게 추가된: 비디오 추론(video_inference) 함수
###############################################################
@torch.no_grad()
def video_inference(
    video_path: str,
    img_size: tuple[int, int],
    model_cfg: dict,
    ckpt_path: str,
    device: torch.device
):
    """
    비디오에서 프레임을 추출하여 각 프레임에 대해
    사람 포즈를 추정하고, 결과를 다시 비디오로 저장.
    [변경 사항]
    - 결과 파일이름 = 원본 이름 + '-result'
    - 결과 파일 위치 = video/ 폴더 (없으면 생성)
    """
    # 1) 모델 초기화 및 가중치 로드
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt:
        vit_pose.load_state_dict(ckpt['state_dict'])
    else:
        vit_pose.load_state_dict(ckpt)
    vit_pose.to(device)
    vit_pose.eval()

    # 2) 비디오 캡쳐 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 저장 폴더 생성 (video/) -> 없으면 생성
    os.makedirs('video', exist_ok=True)

    # 원본 비디오 파일 이름에서 확장자 분리
    base_name = osp.basename(video_path)               # 예: sample.mp4
    stem, ext = osp.splitext(base_name)               # stem=sample, ext=.mp4
    result_name = f"{stem}-result{ext}"               # 예: sample-result.mp4
    output_video_path = osp.join('video', result_name) # video/sample-result.mp4

    # 3) 결과를 저장할 VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 코덱 설정
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video... This may take a while.")

    # 4) 비디오 읽기 및 프레임별 추론
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # (a) OpenCV 이미지(BGR) -> PIL 이미지(RGB)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # (b) 전처리 (리사이즈, 텐서 변환)
        img_tensor = transforms.Compose([
            transforms.Resize((img_size[1], img_size[0])),
            transforms.ToTensor()
        ])(pil_img).unsqueeze(0).to(device)

        # (c) 모델 추론
        heatmaps = vit_pose(img_tensor).detach().cpu().numpy()
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[frame_width // 2, frame_height // 2]]),
            scale=np.array([[frame_width, frame_height]]),
            unbiased=True,
            use_udp=True
        )
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)

        # (d) 시각화
        vis_frame = frame.copy()  # 원본 프레임 복사
        for pid, point in enumerate(points):
            vis_frame = draw_points_and_skeleton(
                vis_frame,
                point,
                joints_dict()['coco']['skeleton'],
                person_index=pid,
                points_color_palette='gist_rainbow',
                skeleton_color_palette='jet',
                points_palette_samples=10,
                confidence_threshold=0.4
            )

        # (e) 결과 프레임을 비디오로 저장
        out.write(vis_frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Finished processing video. Total frames: {frame_count}")
    print(f"Output saved to: {output_video_path}")

###############################################################
# main 함수에서 이미지/비디오 분기 처리
###############################################################
if __name__ == "__main__":
    # config 파일 불러오기 (모델/데이터 설정)
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    from configs.ViTPose_base_coco_256x192 import data_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default=None, help='image path(s)')
    parser.add_argument('--video-path', type=str, default=None, help='video path')
    args = parser.parse_args()

    # 모델 체크포인트 경로 예시 (다운로드 위치나 로컬 경로에 맞춰 수정 필요)
    CUR_DIR = osp.dirname(__file__)
    CKPT_PATH = osp.join(CUR_DIR, 'runs', 'vitpose-b-multi-coco.pth')

    # 데이터 설정에서 이미지 크기 불러오기
    img_size = data_cfg['image_size']

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 추론
    if args.image_path:
        if not isinstance(args.image_path, list):
            args.image_path = [args.image_path]
        for img_path in args.image_path:
            print(f"Processing image: {img_path}")
            inference(
                img_path=img_path,
                img_size=img_size,
                model_cfg=model_cfg,
                ckpt_path=CKPT_PATH,
                device=device,
                save_result=True
            )

    # 비디오 추론
    if args.video_path:
        print(f"Processing video: {args.video_path}")
        video_inference(
            video_path=args.video_path,
            img_size=img_size,
            model_cfg=model_cfg,
            ckpt_path=CKPT_PATH,
            device=device
        )
