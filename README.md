# Depth Scanner

Marigold 기반 monocular depth map 추출 도구.  
이미지/영상을 넣으면 **그레이스케일 depth map**을 출력합니다.  
After Effects에서 바로 사용 가능한 **16bit PNG 시퀀스** 출력을 지원합니다.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 주요 기능

- **이미지 → 16bit depth PNG** 출력
- **영상 → depth map 영상** (mp4) 출력
- **영상 → 16bit PNG 시퀀스** 출력 (After Effects 최적)
- **Temporal blending** — 영상 프레임 간 깜빡임 감소
- **Side-by-side** 모드 — 원본과 depth 나란히 비교
- GPU / CPU / Apple Silicon(MPS) 자동 감지

## 설치

```bash
git clone https://github.com/F-Tako/depth-scanner.git
cd depth-scanner
pip install -r requirements.txt
```

> 첫 실행 시 Marigold 모델(~2GB)이 자동 다운로드됩니다.

## 사용법

### 기본

```bash
# 이미지 → depth map
python depth_scanner.py photo.jpg

# 영상 → depth 영상
python depth_scanner.py video.mp4

# 영상 → 16bit PNG 시퀀스 (AE용)
python depth_scanner.py video.mp4 --sequence
```

### 고급 옵션

```bash
# 고품질 (느림)
python depth_scanner.py video.mp4 --steps 4 --ensemble 5

# 빠른 모드 (LCM 모델)
python depth_scanner.py video.mp4 --lcm --steps 1

# 원본+depth 나란히
python depth_scanner.py video.mp4 --side-by-side

# temporal blending 강도 조절 (0~1)
python depth_scanner.py video.mp4 --blend 0.5

# 출력 경로 지정
python depth_scanner.py video.mp4 -o result_depth.mp4

# CPU 강제 사용
python depth_scanner.py video.mp4 --cpu
```

## 전체 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `input` | (필수) | 입력 이미지 또는 영상 경로 |
| `-o, --output` | 자동 | 출력 경로 |
| `--steps` | 1 | denoising steps (높을수록 정밀) |
| `--ensemble` | 1 | ensemble 횟수 (높을수록 안정적) |
| `--lcm` | off | LCM 모델 사용 (빠르지만 덜 정밀) |
| `--sequence` | off | 영상을 16bit PNG 시퀀스로 출력 |
| `--side-by-side` | off | 원본과 depth 나란히 출력 |
| `--blend` | 0.3 | temporal blending 강도 (0=없음, 1=최대) |
| `--cpu` | off | CPU 강제 사용 |

## After Effects에서 사용하기

### PNG 시퀀스 (권장)

1. `--sequence` 옵션으로 16bit PNG 시퀀스 생성
2. AE → File → Import → File
3. 첫 번째 이미지 선택 → **PNG Sequence** 체크
4. 타임라인에 배치 후 활용

### 활용 예시

- **Displacement Map** — depth map으로 3D 느낌 표현
- **Luma Key / Matte** — 깊이 기반 합성
- **Camera Lens Blur** — depth 기반 피사계 심도
- **Parallax Effect** — 2.5D 카메라 움직임

## Depth Map 방향

- **밝은 부분** = 카메라에 가까움
- **어두운 부분** = 카메라에서 멀리

## 모델 정보

- **Marigold v1.1** — Stable Diffusion 기반, CVPR 2024 Oral / Best Paper Award Candidate
- **Marigold LCM** — `--lcm` 옵션 사용 시, 더 빠르지만 정밀도 약간 낮음

## 요구 사항

- Python 3.8+
- GPU 권장 (NVIDIA CUDA / Apple MPS)
- CPU에서도 동작하나 속도 느림

## License

MIT License
