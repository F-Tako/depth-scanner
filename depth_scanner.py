#!/usr/bin/env python3
"""
Depth Scanner — Marigold 기반 depth map 추출 도구
이미지 및 영상 모두 지원 / After Effects용 그레이스케일 depth map 출력

사용법:
  python depth_scanner.py input.mp4                     # 기본 설정
  python depth_scanner.py input.mp4 -o output.mp4       # 출력 경로 지정
  python depth_scanner.py input.png                     # 이미지도 가능
  python depth_scanner.py input.mp4 --steps 4 --ensemble 5  # 품질 조절
  python depth_scanner.py input.mp4 --side-by-side      # 원본+depth 나란히 출력

요구 사항:
  pip install diffusers torch torchvision accelerate opencv-python pillow tqdm
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def load_pipe(device: str, dtype, use_lcm: bool = False):
    """Marigold depth pipeline 로드"""
    import diffusers

    checkpoint = (
        "prs-eth/marigold-depth-lcm-v1-0" if use_lcm
        else "prs-eth/marigold-depth-v1-1"
    )
    print(f"[*] 모델 로딩 중: {checkpoint}")

    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        checkpoint,
        variant="fp16" if dtype == torch.float16 else None,
        torch_dtype=dtype,
    ).to(device)

    # 성능 최적화
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    print("[*] 모델 로딩 완료")
    return pipe


def depth_to_gray(depth_np: np.ndarray) -> np.ndarray:
    """
    depth prediction (0~1 float) -> 16bit 그레이스케일 numpy array
    가까울수록 밝게 (AE에서 displacement/luma key 등에 바로 사용 가능)
    """
    depth_norm = np.clip(depth_np, 0.0, 1.0)
    depth_inv = 1.0 - depth_norm  # 가까운 곳 = 밝게
    gray_16 = (depth_inv * 65535).astype(np.uint16)
    return gray_16


def depth_to_gray_8bit(depth_np: np.ndarray) -> np.ndarray:
    """depth -> 8bit grayscale (영상 인코딩용)"""
    depth_norm = np.clip(depth_np, 0.0, 1.0)
    depth_inv = 1.0 - depth_norm
    gray_8 = (depth_inv * 255).astype(np.uint8)
    return gray_8


def depth_to_exr(depth_np: np.ndarray) -> np.ndarray:
    """depth -> 32bit float EXR용 3채널 (AE 호환)"""
    depth_norm = np.clip(depth_np, 0.0, 1.0)
    depth_inv = (1.0 - depth_norm).astype(np.float32)
    return np.stack([depth_inv, depth_inv, depth_inv], axis=-1)


def save_depth(depth_np: np.ndarray, out_path: str, fmt: str = "exr"):
    """depth_np: 0~1 float. fmt: exr, png16, png8"""
    depth_inv = 1.0 - np.clip(depth_np, 0.0, 1.0)
    if fmt == "exr":
        out_path = str(Path(out_path).with_suffix(".exr"))
        exr = depth_inv.astype(np.float32)
        cv2.imwrite(out_path, np.stack([exr, exr, exr], axis=-1))
    elif fmt == "png8":
        out_path = str(Path(out_path).with_suffix(".png"))
        cv2.imwrite(out_path, (depth_inv * 255).astype(np.uint8))
    else:  # png16
        out_path = str(Path(out_path).with_suffix(".png"))
        cv2.imwrite(out_path, (depth_inv * 65535).astype(np.uint16))
    return out_path


# ─────────────────────────────────────────────
# 이미지 처리
# ─────────────────────────────────────────────

def process_image(
    pipe,
    input_path: str,
    output_path: str,
    num_steps: int,
    ensemble_size: int,
    side_by_side: bool = False,
    fmt: str = "exr",
):
    print(f"[*] 이미지 처리: {input_path} (출력: {fmt})")
    image = Image.open(input_path).convert("RGB")

    result = pipe(
        image,
        num_inference_steps=num_steps,
        ensemble_size=ensemble_size,
    )
    depth_np = result.prediction.squeeze()
    if torch.is_tensor(depth_np):
        depth_np = depth_np.cpu().numpy()

    if side_by_side:
        gray_16 = depth_to_gray(depth_np)
        orig = np.array(image.resize((gray_16.shape[1], gray_16.shape[0])))
        orig_gray_16 = (orig.mean(axis=2) / 255.0 * 65535).astype(np.uint16)
        combined = np.hstack([orig_gray_16, gray_16])
        cv2.imwrite(output_path, combined)
    else:
        out = save_depth(depth_np, output_path, fmt)
        output_path = out

    print(f"[+] 저장 완료: {output_path}")


# ─────────────────────────────────────────────
# 영상 처리
# ─────────────────────────────────────────────

def process_video(
    pipe,
    input_path: str,
    output_path: str,
    num_steps: int,
    ensemble_size: int,
    side_by_side: bool = False,
    temporal_blend: float = 0.3,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[*] 영상 정보: {w}x{h}, {fps:.2f}fps, {total_frames}프레임")
    print(f"[*] 설정: steps={num_steps}, ensemble={ensemble_size}, blend={temporal_blend}")

    out_w = w * 2 if side_by_side else w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h), isColor=False)

    if not out.isOpened():
        output_path = str(Path(output_path).with_suffix(".avi"))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h), isColor=False)

    prev_depth = None

    for i in tqdm(range(total_frames), desc="Depth 추출 중"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        result = pipe(
            pil_img,
            num_inference_steps=num_steps,
            ensemble_size=ensemble_size,
        )
        depth_np = result.prediction.squeeze()
        if torch.is_tensor(depth_np):
            depth_np = depth_np.cpu().numpy()

        # Temporal blending: 프레임 간 깜빡임 감소
        if prev_depth is not None and temporal_blend > 0:
            depth_np = (1 - temporal_blend) * depth_np + temporal_blend * prev_depth
        prev_depth = depth_np.copy()

        gray_8 = depth_to_gray_8bit(depth_np)

        if gray_8.shape[:2] != (h, w):
            gray_8 = cv2.resize(gray_8, (w, h), interpolation=cv2.INTER_LINEAR)

        if side_by_side:
            orig_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            combined = np.hstack([orig_gray, gray_8])
            out.write(combined)
        else:
            out.write(gray_8)

    cap.release()
    out.release()
    print(f"[+] 저장 완료: {output_path}")


# ─────────────────────────────────────────────
# 시퀀스 출력 — AE용 16bit PNG 시퀀스
# ─────────────────────────────────────────────

def process_video_to_sequence(
    pipe,
    input_path: str,
    output_dir: str,
    num_steps: int,
    ensemble_size: int,
    temporal_blend: float = 0.3,
    fmt: str = "exr",
):
    """영상 -> 이미지 시퀀스 (EXR/PNG16/PNG8)"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] 영상 -> {fmt.upper()} 시퀀스: {output_dir}/")
    print(f"[*] {total_frames}프레임, {fps:.2f}fps")

    prev_depth = None
    frame_count = 0

    for i in tqdm(range(total_frames), desc="Depth 시퀀스 추출 중"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        result = pipe(
            pil_img,
            num_inference_steps=num_steps,
            ensemble_size=ensemble_size,
        )
        depth_np = result.prediction.squeeze()
        if torch.is_tensor(depth_np):
            depth_np = depth_np.cpu().numpy()

        if prev_depth is not None and temporal_blend > 0:
            depth_np = (1 - temporal_blend) * depth_np + temporal_blend * prev_depth
        prev_depth = depth_np.copy()

        out_path = os.path.join(output_dir, f"depth_{i:06d}")
        save_depth(depth_np, out_path, fmt)
        frame_count = i + 1

    cap.release()
    ext = "exr" if fmt == "exr" else "png"
    print(f"[+] 시퀀스 저장 완료: {output_dir}/ ({frame_count}장 .{ext})")
    print(f"[i] AE: File > Import > File 에서 첫 번째 이미지 선택 > Sequence 체크")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Depth Scanner - Marigold 기반 depth map 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python depth_scanner.py photo.jpg                          이미지 -> 16bit depth PNG
  python depth_scanner.py video.mp4                          영상 -> depth 영상
  python depth_scanner.py video.mp4 --sequence               영상 -> 16bit PNG 시퀀스 (AE 최적)
  python depth_scanner.py video.mp4 --steps 4 --ensemble 5   고품질 (느림)
  python depth_scanner.py video.mp4 --lcm --steps 1          빠른 모드 (LCM)
  python depth_scanner.py video.mp4 --side-by-side            원본+depth 나란히
  python depth_scanner.py video.mp4 --blend 0.5              temporal blending 강도
        """,
    )
    parser.add_argument("input", help="입력 이미지 또는 영상 경로")
    parser.add_argument("-o", "--output", help="출력 경로 (기본: input_depth.ext)")
    parser.add_argument("--steps", type=int, default=1,
                        help="denoising steps (기본: 1, v1.1은 1로도 충분)")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="ensemble 횟수 (기본: 1, 높을수록 안정적)")
    parser.add_argument("--lcm", action="store_true",
                        help="LCM 모델 사용 (빠르지만 덜 정밀)")
    parser.add_argument("--side-by-side", action="store_true",
                        help="원본과 depth를 나란히 출력")
    parser.add_argument("--blend", type=float, default=0.3,
                        help="영상 temporal blending 강도 (0~1, 기본: 0.3)")
    parser.add_argument("--sequence", action="store_true",
                        help="영상을 이미지 시퀀스로 출력 (AE용)")
    parser.add_argument("--format", type=str, default="exr",
                        choices=["exr", "png16", "png8"],
                        help="출력 포맷 (기본: exr, 32bit float)")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU 강제 사용")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[!] 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 장치 설정
    if args.cpu:
        device = "cpu"
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"[*] 장치: {device}")

    pipe = load_pipe(device, dtype, use_lcm=args.lcm)

    inp = Path(args.input)
    fmt = args.format

    if is_image(args.input):
        ext = ".exr" if fmt == "exr" else ".png"
        out = args.output or str(inp.with_name(f"{inp.stem}_depth{ext}"))
        process_image(pipe, args.input, out, args.steps, args.ensemble, args.side_by_side, fmt)

    elif is_video(args.input):
        if args.sequence:
            out_dir = args.output or str(inp.with_name(f"{inp.stem}_depth_seq"))
            process_video_to_sequence(
                pipe, args.input, out_dir, args.steps, args.ensemble, args.blend, fmt
            )
        else:
            out = args.output or str(inp.with_name(f"{inp.stem}_depth.mp4"))
            process_video(
                pipe, args.input, out, args.steps, args.ensemble,
                args.side_by_side, args.blend
            )
    else:
        print(f"[!] 지원하지 않는 형식: {inp.suffix}")
        sys.exit(1)


if __name__ == "__main__":
    main()
