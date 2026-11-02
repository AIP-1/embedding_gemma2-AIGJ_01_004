# EmbeddingGemma 2 실습 결과 (보드 B, team5-b)

측정일: 2026-10-23 · Jetson Orin Nano 8GB, L4T R39 (JetPack 7 계열), MAXN_SUPER, 스왑 8GB
환경: torch 2.14.1+cu132, sentence-transformers 6.2.0, transformers 5.20.2, 모델 google/embeddinggemma-2 (bfloat16)

## 6단계 성능 표

| 구성 | 로드 크기 | 로드 후 RAM (MB) | 입력 | 지연 (ms) | GPU 사용률 (%) |
| --- | --- | --- | --- | --- | --- |
| 텍스트 전용 | 270M | 4272 (+1397) | 질의 2개 | 약 150~160 | 24 |
| +비전 | 440M | 5126 (+2164) | 이미지 5장 | 2739 | 100 |
| +비전 | 440M | 5126 (+2164) | 동영상 10초 | 1960 | 100 |
| +오디오 | 570M | 4919 (+2032) | WAV 4개 | 1229 | 84 |
| 전체 | 740M | 미측정 | 혼합 | 미측정 | 미측정 |

- RAM은 tegrastats 최대값이며, 괄호는 실행 전(약 2.9GB) 대비 증가량.
- 텍스트 지연은 3단계 결과(dim 768/256/128 각각 160/151/162 ms).

## 3단계 텍스트 검색

| 차원 | 지연 | "졸리고 머리가 맑지 않아요" → | "여름 냉방 몇 도?" → |
| --- | --- | --- | --- |
| 768 | 160 ms | CO2 환기 문장 (0.717) | 에어컨 온도 문장 (0.725) |
| 256 | 151 ms | CO2 환기 문장 (0.734) | 에어컨 온도 문장 (0.754) |
| 128 | 162 ms | CO2 환기 문장 (0.785) | 에어컨 온도 문장 (0.810) |

모든 차원에서 정답 1위. 차원별 점수는 기준이 달라 직접 비교하지 않는다.

## 4단계 이미지·동영상 검색

- 자료: C920 웹캠으로 10초 녹화(clip.mp4) + 2초마다 프레임 5장
- "모니터 화면" → frame_04 (0.706), frame_05 (0.700)
- 나머지 질의는 0.59~0.66으로 비슷함 (같은 공간의 장면이라 서로 닮음)
- 질의-동영상 유사도: 빈 강의실 0.692, 수업 모습 0.651, 화이트보드 0.635, 모니터 화면 0.698

## 5단계 음성 검색

| 질의 | 합성음만 있을 때 | 실제 목소리(mic.wav) 추가 후 |
| --- | --- | --- |
| 덥다고 냉방을 요청하는 목소리 | noise.wav (0.651) ✗ | mic.wav (0.713) ✓ |
| 날카로운 경보음 | beep.wav (0.702) ✓ | beep.wav (0.702) ✓ |
| 웅웅거리는 배경 소음 | noise.wav (0.682) ✓ | noise.wav (0.682) ✓ |

espeak-ng 합성음은 사람 목소리로 잘 인식되지 않았고, C920 마이크 실제 녹음은 정확히 찾음 (README 예상과 일치).

## README와 다르게 한 점 (이 보드에서 필요했던 것)

1. 전원 모드: MAXN_SUPER가 이 보드에선 2번 (README의 `-m 0`은 15W).
2. torchaudio 미설치 (README Q7 충돌 회피), sentence-transformers는 `[image,video]`만 설치.
3. NVIDIA FFmpeg 8 축소판에는 USB 카메라 입력, 테스트 음 생성(lavfi), libswscale이 없음.
   - 녹화: ffmpeg 대신 OpenCV(`record.py`)
   - 동영상 임베딩: `sudo apt install libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4 libavdevice60 libavfilter9` 후 `pip install -c ~/eg2/torch-pin.txt torchcodec`
   - 경보음·잡음: 파이썬(numpy)으로 생성(`gen_audio.py`)
