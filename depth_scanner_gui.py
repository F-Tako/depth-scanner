#!/usr/bin/env python3
"""
Depth Scanner GUI — Marigold 기반 Depth Map 추출 도구
드래그앤드롭 / 이미지+영상 지원 / 8K+ 해상도 지원
After Effects용 32bit EXR / 16bit PNG 그레이스케일 출력

요구사항:
  pip install diffusers torch torchvision accelerate opencv-python Pillow tqdm tkinterdnd2
"""

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── 상수 ──
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}
PREVIEW_MAX = 600  # 미리보기 최대 크기


class DepthScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Scanner — Marigold")
        self.root.configure(bg="#0d0d14")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        self.pipe = None
        self.input_path = None
        self.output_path = None
        self.is_processing = False
        self.cancel_flag = False

        self.setup_styles()
        self.build_ui()
        self.try_enable_dnd()

    # ── 스타일 ──
    def setup_styles(self):
        self.colors = {
            "bg": "#0d0d14",
            "surface": "#14141e",
            "border": "#1e1e30",
            "text": "#e0e0e8",
            "dim": "#6a6a80",
            "accent": "#00d4ff",
            "accent2": "#7b61ff",
        }
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Surface.TFrame", background=self.colors["surface"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"],
                         font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=self.colors["accent"])
        style.configure("Dim.TLabel", foreground=self.colors["dim"], font=("Segoe UI", 9))
        style.configure("Status.TLabel", foreground=self.colors["dim"], font=("Consolas", 9))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 6))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TProgressbar", troughcolor=self.colors["border"],
                         background=self.colors["accent"], thickness=6)
        style.configure("TCombobox", font=("Segoe UI", 9))
        style.configure("TCheckbutton", background=self.colors["bg"], foreground=self.colors["text"],
                         font=("Segoe UI", 9))

    # ── UI 구성 ──
    def build_ui(self):
        c = self.colors

        # 헤더
        hdr = ttk.Frame(self.root)
        hdr.pack(fill="x", padx=20, pady=(15, 5))
        ttk.Label(hdr, text="DEPTH SCANNER", style="Title.TLabel").pack(side="left")
        ttk.Label(hdr, text="Marigold v1.1 — GPU/CPU 자동감지",
                  style="Dim.TLabel").pack(side="left", padx=(15, 0))

        # ── 상단: 입력 영역 ──
        inp_frame = ttk.Frame(self.root)
        inp_frame.pack(fill="x", padx=20, pady=10)

        # 드롭 영역
        self.drop_frame = tk.Frame(inp_frame, bg=c["surface"], highlightbackground=c["border"],
                                    highlightthickness=2, cursor="hand2")
        self.drop_frame.pack(fill="x", ipady=25)
        self.drop_label = tk.Label(self.drop_frame, text="이미지 또는 영상을 여기에 드래그하거나 클릭하여 선택",
                                    bg=c["surface"], fg=c["dim"],
                                    font=("Segoe UI", 11))
        self.drop_label.pack(expand=True)
        self.drop_sub = tk.Label(self.drop_frame,
                                  text="jpg / png / tiff / mp4 / mov / avi — 최대 8K+ 해상도 지원",
                                  bg=c["surface"], fg=c["border"],
                                  font=("Consolas", 8))
        self.drop_sub.pack()

        self.drop_frame.bind("<Button-1>", lambda e: self.browse_file())
        self.drop_label.bind("<Button-1>", lambda e: self.browse_file())

        # ── 설정 영역 ──
        opt_frame = ttk.Frame(self.root)
        opt_frame.pack(fill="x", padx=20, pady=5)

        # Steps
        ttk.Label(opt_frame, text="Steps:", style="Dim.TLabel").pack(side="left")
        self.steps_var = tk.StringVar(value="1")
        steps_cb = ttk.Combobox(opt_frame, textvariable=self.steps_var, values=["1", "2", "4", "10"],
                                 width=4, state="readonly")
        steps_cb.pack(side="left", padx=(5, 15))

        # Ensemble
        ttk.Label(opt_frame, text="Ensemble:", style="Dim.TLabel").pack(side="left")
        self.ensemble_var = tk.StringVar(value="1")
        ens_cb = ttk.Combobox(opt_frame, textvariable=self.ensemble_var, values=["1", "3", "5", "10"],
                               width=4, state="readonly")
        ens_cb.pack(side="left", padx=(5, 15))

        # Temporal blend (영상용)
        ttk.Label(opt_frame, text="Blend:", style="Dim.TLabel").pack(side="left")
        self.blend_var = tk.StringVar(value="0.3")
        blend_cb = ttk.Combobox(opt_frame, textvariable=self.blend_var,
                                 values=["0", "0.2", "0.3", "0.5", "0.7"],
                                 width=4, state="readonly")
        blend_cb.pack(side="left", padx=(5, 15))

        # 출력 형식
        ttk.Label(opt_frame, text="출력:", style="Dim.TLabel").pack(side="left", padx=(15, 0))
        self.format_var = tk.StringVar(value="exr")
        fmt_cb = ttk.Combobox(opt_frame, textvariable=self.format_var,
                               values=["exr", "png16", "png8", "mp4"],
                               width=6, state="readonly")
        fmt_cb.pack(side="left", padx=(5, 0))

        # 시퀀스 출력 (영상용)
        self.seq_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="시퀀스 출력 (AE용)", variable=self.seq_var).pack(side="left", padx=(10, 0))

        # ── 버튼 영역 ──
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=20, pady=8)

        self.run_btn = ttk.Button(btn_frame, text="▶  Scan", style="Accent.TButton",
                                   command=self.start_scan)
        self.run_btn.pack(side="left")

        self.cancel_btn = ttk.Button(btn_frame, text="■  취소", command=self.cancel_scan, state="disabled")
        self.cancel_btn.pack(side="left", padx=(10, 0))

        self.open_btn = ttk.Button(btn_frame, text="📁  출력 폴더 열기", command=self.open_output, state="disabled")
        self.open_btn.pack(side="left", padx=(10, 0))

        # 장치 표시
        device_text = self.detect_device_text()
        ttk.Label(btn_frame, text=device_text, style="Dim.TLabel").pack(side="right")

        # ── 프로그레스 ──
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill="x", padx=20, pady=5)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x")
        self.status_label = ttk.Label(prog_frame, text="대기 중", style="Status.TLabel")
        self.status_label.pack(anchor="w", pady=(3, 0))

        # ── 미리보기 영역 ──
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=(5, 15))

        # 원본
        left = tk.Frame(preview_frame, bg=c["surface"], highlightbackground=c["border"], highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(left, text="ORIGINAL", bg=c["surface"], fg=c["border"],
                 font=("Consolas", 8)).pack(anchor="nw", padx=8, pady=4)
        self.orig_label = tk.Label(left, bg=c["surface"])
        self.orig_label.pack(expand=True)

        # Depth
        right = tk.Frame(preview_frame, bg=c["surface"], highlightbackground=c["border"], highlightthickness=1)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        tk.Label(right, text="DEPTH MAP", bg=c["surface"], fg=c["border"],
                 font=("Consolas", 8)).pack(anchor="nw", padx=8, pady=4)
        self.depth_label = tk.Label(right, bg=c["surface"])
        self.depth_label.pack(expand=True)

    # ── 드래그앤드롭 ──
    def try_enable_dnd(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind("<<Drop>>", self.on_drop)
        except Exception:
            # tkinterdnd2 없거나 Tcl 에러 시 클릭만 사용
            pass

    def on_drop(self, event):
        path = event.data.strip().strip("{}")
        if os.path.isfile(path):
            self.set_input(path)

    # ── 파일 선택 ──
    def browse_file(self):
        ftypes = [
            ("지원 파일", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.mp4 *.mov *.avi *.mkv *.webm"),
            ("이미지", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("영상", "*.mp4 *.mov *.avi *.mkv *.webm"),
            ("모든 파일", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=ftypes)
        if path:
            self.set_input(path)

    def set_input(self, path):
        self.input_path = path
        name = Path(path).name
        size_mb = os.path.getsize(path) / (1024 * 1024)
        ext = Path(path).suffix.lower()

        if ext in IMAGE_EXTS:
            ftype = "이미지"
        elif ext in VIDEO_EXTS:
            ftype = "영상"
        else:
            messagebox.showerror("오류", f"지원하지 않는 형식: {ext}")
            return

        self.drop_label.config(text=f"{name} ({size_mb:.1f}MB) — {ftype}")
        self.status_label.config(text=f"입력: {path}")

        # 이미지면 미리보기
        if ext in IMAGE_EXTS:
            self.show_preview_image(path)

    def show_preview_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
            photo = ImageTk.PhotoImage(img)
            self.orig_label.config(image=photo)
            self.orig_label.image = photo
        except Exception:
            pass

    def show_preview_depth(self, depth_np):
        try:
            depth_inv = 1.0 - np.clip(depth_np, 0, 1)
            gray = (depth_inv * 255).astype(np.uint8)
            img = Image.fromarray(gray, mode="L")
            img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
            photo = ImageTk.PhotoImage(img)
            self.depth_label.config(image=photo)
            self.depth_label.image = photo
        except Exception:
            pass

    # ── 장치 감지 ──
    def detect_device_text(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            return f"GPU: {name} ({vram:.1f}GB)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU: Apple MPS"
        else:
            return "CPU (GPU 없음 — 느릴 수 있음)"

    def get_device_dtype(self):
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float32
        else:
            return "cpu", torch.float32

    # ── 모델 로드 ──
    def load_model(self):
        if self.pipe is not None:
            return
        import diffusers
        device, dtype = self.get_device_dtype()
        self.update_status("모델 로딩 중... (첫 실행 시 ~2GB 다운로드)")
        self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-v1-1",
            variant="fp16" if dtype == torch.float16 else None,
            torch_dtype=dtype,
        ).to(device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        self.update_status("모델 준비 완료")

    # ── 스캔 시작 ──
    def start_scan(self):
        if not self.input_path or not os.path.isfile(self.input_path):
            messagebox.showwarning("경고", "먼저 파일을 선택하세요.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.cancel_flag = False
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.open_btn.config(state="disabled")
        self.progress["value"] = 0

        thread = threading.Thread(target=self.scan_worker, daemon=True)
        thread.start()

    def cancel_scan(self):
        self.cancel_flag = True
        self.update_status("취소 중...")

    def scan_worker(self):
        try:
            self.load_model()
            ext = Path(self.input_path).suffix.lower()
            if ext in IMAGE_EXTS:
                self.process_image()
            elif ext in VIDEO_EXTS:
                self.process_video()
        except Exception as e:
            self.update_status(f"오류: {e}")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            self.root.after(0, lambda: self.cancel_btn.config(state="disabled"))

    # ── 이미지 처리 ──
    def process_image(self):
        self.update_status("이미지 처리 중...")
        self.update_progress(10)

        img = Image.open(self.input_path).convert("RGB")
        w, h = img.size

        steps = int(self.steps_var.get())
        ensemble = int(self.ensemble_var.get())

        self.update_status(f"Depth 추출 중... ({w}x{h}, steps={steps}, ensemble={ensemble})")
        self.update_progress(20)

        result = self.pipe(img, num_inference_steps=steps, ensemble_size=ensemble)
        depth_np = result.prediction.squeeze()
        if torch.is_tensor(depth_np):
            depth_np = depth_np.cpu().numpy()

        self.update_progress(80)

        inp = Path(self.input_path)
        depth_inv = 1.0 - np.clip(depth_np, 0, 1)
        fmt = self.format_var.get()
        out_path = self.save_depth(depth_inv, inp.parent, f"{inp.stem}_depth", fmt)

        self.output_path = str(inp.parent)
        self.update_progress(100)
        self.update_status(f"완료: {out_path}")
        self.root.after(0, lambda: self.open_btn.config(state="normal"))

        # 미리보기
        self.root.after(0, lambda: self.show_preview_depth(depth_np))

    # ── 영상 처리 ──
    def process_video(self):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.update_status("영상을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        steps = int(self.steps_var.get())
        ensemble = int(self.ensemble_var.get())
        blend = float(self.blend_var.get())
        use_seq = self.seq_var.get()

        inp = Path(self.input_path)
        fmt = self.format_var.get()
        self.update_status(f"영상: {w}x{h}, {fps:.1f}fps, {total}프레임, 출력: {fmt}")

        if use_seq:
            out_dir = str(inp.with_name(f"{inp.stem}_depth_seq"))
            os.makedirs(out_dir, exist_ok=True)
            self.output_path = out_dir
        else:
            out_path = str(inp.with_name(f"{inp.stem}_depth.mp4"))
            self.output_path = str(inp.parent)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

        prev_depth = None

        for i in range(total):
            if self.cancel_flag:
                self.update_status("취소됨")
                break

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            result = self.pipe(pil_img, num_inference_steps=steps, ensemble_size=ensemble)
            depth_np = result.prediction.squeeze()
            if torch.is_tensor(depth_np):
                depth_np = depth_np.cpu().numpy()

            # Temporal blend
            if prev_depth is not None and blend > 0:
                depth_np = (1 - blend) * depth_np + blend * prev_depth
            prev_depth = depth_np.copy()

            depth_inv = 1.0 - np.clip(depth_np, 0, 1)

            if use_seq:
                # 원본 크기로 리사이즈
                if depth_inv.shape[:2] != (h, w):
                    depth_inv_resized = cv2.resize(depth_inv.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    depth_inv_resized = depth_inv.astype(np.float32)
                self.save_depth(depth_inv_resized, out_dir, f"depth_{i:06d}", fmt)
            else:
                gray_8 = (depth_inv * 255).astype(np.uint8)
                if gray_8.shape[:2] != (h, w):
                    gray_8 = cv2.resize(gray_8, (w, h), interpolation=cv2.INTER_LINEAR)
                out.write(gray_8)

            pct = int((i + 1) / total * 100)
            self.update_progress(pct)
            self.update_status(f"프레임 {i+1}/{total} ({pct}%) — {w}x{h}")

            # 미리보기 업데이트 (10프레임마다)
            if i % 10 == 0:
                self.root.after(0, lambda d=depth_np: self.show_preview_depth(d))
                # 원본 미리보기
                orig_pil = pil_img.copy()
                orig_pil.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
                self.root.after(0, lambda im=orig_pil: self._set_orig_preview(im))

        cap.release()
        if not use_seq and not self.cancel_flag:
            out.release()

        if not self.cancel_flag:
            if use_seq:
                self.update_status(f"완료: {out_dir}/ ({i+1}장 PNG 시퀀스)")
            else:
                self.update_status(f"완료: {out_path}")
            self.root.after(0, lambda: self.open_btn.config(state="normal"))

    def _set_orig_preview(self, pil_img):
        photo = ImageTk.PhotoImage(pil_img)
        self.orig_label.config(image=photo)
        self.orig_label.image = photo

    # ── 출력 폴더 열기 ──
    def open_output(self):
        if self.output_path and os.path.exists(self.output_path):
            if sys.platform == "win32":
                os.startfile(self.output_path)
            elif sys.platform == "darwin":
                os.system(f'open "{self.output_path}"')
            else:
                os.system(f'xdg-open "{self.output_path}"')

    # ── 유틸 ──
    def save_depth(self, depth_inv, out_dir, name, fmt="exr"):
        """
        depth_inv: 0~1 float array (가까움=1, 멀리=0)
        fmt: 'exr' | 'png16' | 'png8'
        Returns: saved file path
        """
        out_dir = str(out_dir)
        if fmt == "exr":
            out_path = os.path.join(out_dir, f"{name}.exr")
            # 32bit float EXR — 단일 채널을 3채널로 확장 (AE 호환)
            depth_f32 = depth_inv.astype(np.float32)
            exr_img = np.stack([depth_f32, depth_f32, depth_f32], axis=-1)
            cv2.imwrite(out_path, exr_img)
        elif fmt == "png16":
            out_path = os.path.join(out_dir, f"{name}.png")
            gray_16 = (depth_inv * 65535).astype(np.uint16)
            cv2.imwrite(out_path, gray_16)
        elif fmt == "png8":
            out_path = os.path.join(out_dir, f"{name}.png")
            gray_8 = (depth_inv * 255).astype(np.uint8)
            cv2.imwrite(out_path, gray_8)
        else:
            # fallback to exr
            out_path = os.path.join(out_dir, f"{name}.exr")
            depth_f32 = depth_inv.astype(np.float32)
            exr_img = np.stack([depth_f32, depth_f32, depth_f32], axis=-1)
            cv2.imwrite(out_path, exr_img)
        return out_path

    def update_status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def update_progress(self, val):
        self.root.after(0, lambda: self.progress.configure(value=val))


def main():
    root = tk.Tk()
    app = DepthScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
