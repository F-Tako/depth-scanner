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
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    print("[*] 모델 로딩 완료")
    return pipe


def depth_to_gray(depth_np):
    depth_norm = np.clip(depth_np, 0.0, 1.0)
    depth_inv = 1.0 - depth_norm
    return (depth_inv * 65535).astype(np.uint16)


def depth_to_gray_8bit(depth_np):
    depth_norm = np.clip(depth_np, 0.0, 1.0)
    depth_inv = 1.0 - depth_norm
    return (depth_inv * 255).astype(np.uint8)


def save_depth(depth_np, out_path, fmt="exr"):
    depth_inv = 1.0 - np.clip(depth_np, 0.0, 1.0)
    if fmt == "exr":
        out_path = str(Path(out_path).with_suffix(".exr"))
        exr = depth_inv.astype(np.float32)
        cv2.imwrite(out_path, np.stack([exr, exr, exr], axis=-1))
    elif fmt == "png8":
        out_path = str(Path(out_path).with_suffix(".png"))
        cv2.imwrite(out_path, (depth_inv * 255).astype(np.uint8))
    else:
        out_path = str(Path(out_path).with_suffix(".png"))
        cv2.imwrite(out_path, (depth_inv * 65535).astype(np.uint16))
    return out_path


def process_image(pipe, input_path, output_path, num_steps, ensemble_size, side_by_side=False, fmt="exr"):
    print(f"[*] 이미지 처리: {input_path} (출력: {fmt})")
    image = Image.open(input_path).convert("RGB")
    result = pipe(image, num_inference_steps=num_steps, ensemble_size=ensemble_size)
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
        output_path = save_depth(depth_np, output_path, fmt)
    print(f"[+] 저장 완료: {output_path}")


def process_video(pipe, input_path, output_path, num_steps, ensemble_size, side_by_side=False, temporal_blend=0.3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {input_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = w * 2 if side_by_side else w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h), isColor=False)
    prev_depth = None
    for i in tqdm(range(total_frames), desc="Depth 추출 중"):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = pipe(pil_img, num_inference_steps=num_steps, ensemble_size=ensemble_size)
        depth_np = result.prediction.squeeze()
        if torch.is_tensor(depth_np):
            depth_np = depth_np.cpu().numpy()
        if prev_depth is not None and temporal_blend > 0:
            depth_np = (1 - temporal_blend) * depth_np + temporal_blend * prev_depth
        prev_depth = depth_np.copy()
        gray_8 = depth_to_gray_8bit(depth_np)
        if gray_8.shape[:2] != (h, w):
            gray_8 = cv2.resize(gray_8, (w, h), interpolation=cv2.INTER_LINEAR)
        if side_by_side:
            orig_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(np.hstack([orig_gray, gray_8]))
        else:
            out.write(gray_8)
    cap.release()
    out.release()
    print(f"[+] 저장 완료: {output_path}")


def process_video_to_sequence(pipe, input_path, output_dir, num_steps, ensemble_size, temporal_blend=0.3, fmt="exr"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {input_path}")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[*] 영상 -> {fmt.upper()} 시퀀스: {output_dir}/")
    prev_depth = None
    frame_count = 0
    for i in tqdm(range(total_frames), desc="Depth 시퀀스 추출 중"):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = pipe(pil_img, num_inference_steps=num_steps, ensemble_size=ensemble_size)
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
    print(f"[+] 시퀀스 저장 완료: {output_dir}/ ({frame_count}짤 .{ext})")
    print(f"[i] AE: File > Import > File 에서 첫 번째 이미지 선택 > Sequence 체크")


def main():
    parser = argparse.ArgumentParser(
        description="Depth Scanner - Marigold depth map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="잔력 이미지 또는 영상 경로")
    parser.add_argument("-o", "--output", help="출력 경로")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--lcm", action="store_true")
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--blend", type=float, default=0.3)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--format", type=str, default="exr", choices=["exr", "png16", "png8"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[!] 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    if args.cpu:
        device, dtype = "cpu", torch.float32
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

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
            process_video_to_sequence(pipe, args.input, out_dir, args.steps, args.ensemble, args.blend, fmt)
        else:
            out = args.output or str(inp.with_name(f"{inp.stem}_depth.mp4"))
            process_video(pipe, args.input, out, args.steps, args.ensemble, args.side_by_side, args.blend)
    else:
        print(f"[!] 지원하지 않는 형식: {inp.suffix}")
        sys.exit(1)


if __name__ == "__main__":
    main()
