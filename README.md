# EmbeddingGemma 2 멀티모달 임베딩 실습 1~7단계 (AIGJ_01_004)

Jetson Orin Nano에서 EmbeddingGemma 2로 텍스트·이미지·동영상·음성을 같은 임베딩 공간에 넣고 검색한 실습 결과입니다. 8~10단계(웹캠 물체 추출, 동작 인식, 두 카메라 근접 알람)는 [practice_task_for_jetson_camera_applications-AIGJ_01_004](https://github.com/AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_004) 저장소에 있습니다. 실습 안내문은 [docs/guide_README.md](docs/guide_README.md)에 있습니다.

## 환경 (보드 B, team5-b)

- Jetson Orin Nano 8GB, L4T R39 (JetPack 7 계열), Python 3.12, NVMe 부팅
- 전원 모드 MAXN_SUPER (nvpmodel 모드 2), 스왑 8GB
- torch 2.14.1+cu132, sentence-transformers 6.2.0, transformers 5.20.2
- 모델 google/embeddinggemma-2 (bfloat16)
- USB 웹캠 Logitech C920 (영상, 마이크)

## 진행 단계

| 단계 | 내용 | 코드 | 상태 |
| --- | --- | --- | --- |
| 1 | 환경 준비 (전원 모드, 스왑, PyTorch) | 명령어만 사용 | 완료 |
| 2 | 패키지·모델 설치 확인 | `check.py`, `eg2.py` | 완료 |
| 3 | 텍스트 검색 (768/256/128차원) | `test_text.py` | 완료 |
| 4 | 이미지·동영상 검색 | `record.py`, `test_vision.py` | 완료 |
| 5 | 음성 검색 | `gen_audio.py`, `test_audio.py` | 완료 |
| 6 | 성능 측정 (지연, 메모리, GPU) | 3~5단계 코드 + tegrastats | 완료 |
| 7 | 도메인 미세 조정 (텍스트 모델) | `eval_ir.py`, `train_ft.py`, `compare_text.py` | 완료 |

## 폴더 구성

```
check.py         2단계 GPU·패키지 확인
eg2.py           모델 로드 공통 코드
test_text.py     3단계 텍스트 검색
record.py        4단계 웹캠 녹화 (OpenCV)
test_vision.py   4단계 이미지·동영상 검색
gen_audio.py     5단계 경보음·잡음 생성 (numpy)
test_audio.py    5단계 음성 검색
eval_ir.py       7단계 검색 평가 (accuracy@k, ndcg@10)
train_ft.py      7단계 미세 조정 학습
compare_text.py  7단계 학습 전후 3단계 문장 점수 비교
data/            7단계 학습 300쌍, 평가 60쌍 (jsonl)
results/         3~7단계 측정 결과, 7단계 학습 로그
reports/         최종 보고서 PDF (1~10단계)
docs/            실습 안내문
```

## 주요 결과

| 구성 | 로드 후 RAM | 입력 | 지연 | GPU 사용률 |
| --- | --- | --- | --- | --- |
| 텍스트 전용 (270M) | 4272 MB | 질의 2개 | 약 150~160 ms | 24% |
| +비전 (440M) | 5126 MB | 이미지 5장 | 2739 ms | 100% |
| +비전 (440M) | 5126 MB | 동영상 10초 | 1960 ms | 100% |
| +오디오 (570M) | 4919 MB | WAV 4개 | 1229 ms | 84% |

- **3단계:** 768/256/128차원 모두 두 질의의 정답 문장이 1위.
- **4단계:** "모니터 화면" 질의가 모니터가 보이는 프레임(0.706)을 1위로 찾음. 같은 강의실 장면이라 나머지 점수는 0.59~0.66으로 비슷함.
- **5단계:** espeak 합성음만 있을 때 "냉방 요청 목소리" 질의가 잡음을 1위로 찾았고, C920 마이크로 실제 목소리를 녹음해 넣자 3개 질의 모두 정답.

**7단계 미세 조정**

| 모델 | accuracy@1 | accuracy@3 | ndcg@10 | 학습 시간 |
| --- | --- | --- | --- | --- |
| 기준 (google/embeddinggemma-2) | 0.783 | 0.967 | 0.903 | - |
| 미세 조정 (eg2-classroom) | 0.800 | 0.983 | 0.915 | 246초 |

- 평가 질의 60개 중 1위 정답 47 → 48개, 3위 안 정답 58 → 59개. 학습 후 평가 점수가 떨어지지 않아 과적합 징후 없음.
- 학습 중 최대 RAM 6769 / 7485 MB, 배치 8로 OOM 없이 완료.
- 3단계 질의 재확인: "냉방 몇 도" 질의는 정답과 2위 차이가 0.070 → 0.177로 커졌고, "졸리고 머리가 맑지 않아요" 질의는 학습 전에도 CO2·조도 문장 차이가 0.007로 거의 같았는데 학습 후 조도 문장이 0.014 앞서며 순위가 바뀜.

## 안내문과 다르게 한 점

1. 전원 모드: MAXN_SUPER가 이 보드에선 모드 2 (안내문의 `-m 0`은 15W).
2. torchaudio 미설치 (안내문 Q7 충돌 회피).
3. NVIDIA FFmpeg 축소판에 USB 카메라 입력, 테스트 음 생성, libswscale이 없어서:
   - 녹화는 OpenCV(`record.py`)로 대체
   - 동영상 임베딩은 `sudo apt install libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4 libavdevice60 libavfilter9` 후 `pip install -c ~/eg2/torch-pin.txt torchcodec`
   - 경보음·잡음은 numpy로 생성(`gen_audio.py`)
4. 7단계 학습 데이터: 안내문이 말하는 제공 파일(`train.jsonl` 282쌍, `eval.jsonl` 60쌍)이 강의 저장소에 없어서 안내문 규격대로 새로 작성 (학습 문서 100개 × 질의 3개 = 300쌍, 평가 문서 20개 × 질의 3개 = 60쌍, 문서·질의 겹침 없음, 수치는 실습용 가상 운영 기준).
5. 7단계 당시 보드 인터넷이 끊겨 있어서 `datasets` 설치 파일을 따로 받아 `pip install --no-index --find-links`로 설치하고, 모델은 `HF_HUB_OFFLINE=1`로 캐시에서 불러옴.
6. 7단계에서 `eg2.py`의 `MODEL_ID`를 `models/eg2-classroom`으로 바꿔 3단계 테스트를 다시 돌렸음. 저장소의 `eg2.py`는 원래 모델(`google/embeddinggemma-2`)로 둠. 학습한 모델(517MB)은 GitHub 파일 크기 제한 때문에 넣지 않음.

녹음·녹화 파일(wav, mp4, jpg)과 학습한 모델은 저장소에 넣지 않았습니다.
