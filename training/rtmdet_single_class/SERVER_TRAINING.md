# Server Training

학습은 서버에서 실행하고 노트북은 코드 수정, 결과 확인, Gradio 접속에 사용합니다. 데이터셋과 가중치는 Git에 올리지 않고 서버가 직접 다운로드합니다.

## 요구사항

- Ubuntu 계열 Linux
- NVIDIA GPU와 정상 설치된 드라이버 (`nvidia-smi` 실행 가능)
- CUDA 11.8 PyTorch를 지원하는 GPU
- 여유 디스크 15GB 이상
- Git, curl, tmux

## 최초 설치

```bash
git clone --branch pill-baseline git@github.com:iccas6-click/ai.git
cd ai
bash training/rtmdet_single_class/scripts/bootstrap_server.sh
```

부트스트랩은 Python 3.11 가상환경과 CUDA PyTorch/MMDetection을 설치하고, 합성 데이터셋 다운로드 및 단일 클래스 변환까지 수행합니다.

## 학습 실행

SSH 연결이 끊겨도 계속 실행되도록 `tmux` 안에서 시작합니다.

```bash
tmux new -s click-pill
cd ~/ai
bash training/rtmdet_single_class/scripts/run_training.sh \
  --num-workers 4 \
  2>&1 | tee training/runs/server-training.log
```

`Ctrl-b`, `d`로 세션에서 빠져나오고 다음 명령으로 다시 접속합니다.

```bash
tmux attach -t click-pill
```

상태 확인:

```bash
nvidia-smi
tail -f ~/ai/training/runs/server-training.log
```

중단된 학습을 마지막 체크포인트에서 재개할 때는 다음 명령을 사용합니다.

```bash
bash training/rtmdet_single_class/scripts/run_training.sh --resume
```

## 결과 회수

서버의 최고 체크포인트만 노트북으로 가져옵니다.

```bash
rsync -avP USER@SERVER:~/ai/training/runs/rtmdet-single-class/best_*.pth \
  /home/gyuha_lee/pill/code/ai/training/runs/rtmdet-single-class/
```

서버에서 Gradio를 실행한다면 외부 포트를 열지 않고 SSH 터널을 사용합니다.

```bash
# 서버
cd ~/ai/inference
../.venv/bin/python -m pill_recognition.app

# 노트북
ssh -L 7860:127.0.0.1:7860 USER@SERVER
```

노트북 브라우저에서 `http://127.0.0.1:7860`으로 접속합니다.
