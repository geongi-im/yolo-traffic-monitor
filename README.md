# YOLO Traffic Monitor

실시간 CCTV 교통량 분석 시스템 - YOLOv11을 활용한 차량 탐지 및 모니터링

![Demo](demo.gif)

## 주요 기능

- **이중 운영 모드**: 배치 분석 / 실시간 스트리밍
- **YOLOv11 차량 탐지**: 승용차, 오토바이, 버스, 트럭
- **웹 대시보드**: HLS 원본 스트림 / MJPEG 실시간 분석
- **텔레그램 알림**: 오류 발생 시 자동 알림
- **다중 CCTV 지원**: 수영로(6301), 자성로(6300), 황령대로(6323)

## 시스템 아키텍처

```
Naver CCTV API → HLS Stream
        │
    ┌───┴────┐
    ▼        ▼
main.py  server.py
(배치)   (실시간)
    │        │
    └───┬────┘
        ▼
  ImageFetcher → 프레임 캡처
        │
        ▼
  ImageAnalyzer
   (YOLO 분석)
        │
    ┌───┴────┐
    ▼        ▼
Telegram   Web UI
```

### 주요 컴포넌트

| 컴포넌트 | 역할 |
|---------|------|
| **ImageFetcher** | CCTV API 인증, HLS URL 추출, 프레임 캡처 |
| **VehicleDetector** | YOLO 모델 추론 |
| **ImageAnalyzer** | 이미지 분석, 바운딩 박스, 통계 |
| **VideoStreamer** | 실시간 MJPEG 스트리밍 |

## 설치

### 필수 요구사항
- Python 3.8+
- FFmpeg
- YOLO 모델 (`yolo11n.pt`)

### 설치 과정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd yolo-traffic-monitor

# 2. 의존성 설치
pip install -r requirements.txt

# 3. FFmpeg 설치
# Windows: choco install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# 4. 환경 변수 설정
cp env.sample .env
# .env 파일에서 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 수정

# 5. YOLO 모델 (첫 실행 시 자동 다운로드)
mkdir model
```

## 환경 설정

| 변수 | 설명 | 기본값 |
|-----|------|-------|
| `INTERVAL_SECONDS` | 배치 분석 주기 (초) | 60 |
| `YOLO_MODEL` | YOLO 모델 경로 | model/yolo11n.pt |
| `CONFIDENCE_THRESHOLD` | 탐지 신뢰도 (0.0-1.0) | 0.3 |
| `DEVICE` | 추론 장치 (cpu/cuda) | cpu |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | - |
| `TELEGRAM_CHAT_ID` | 텔레그램 채팅 ID | - |

## 사용 방법

### 배치 모드

```bash
python main.py
```

- HLS 스트림에서 약 15초간 프레임 캡처 (1초 간격)
- YOLO 분석 후 평균 차량 수 계산
- 결과 이미지 `output/` 폴더에 저장
- `INTERVAL_SECONDS` 주기로 반복

### 실시간 모드

```bash
python server.py
```

웹 브라우저에서 `http://localhost:8000` 접속

- **LIVE**: HLS 원본 스트림
- **LIVE ANALYSIS**: YOLO 분석 적용 스트림
- 자동 재연결 기능

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /` | 웹 인터페이스 |
| `GET /api/hls-url?cctv_id={id}` | HLS 스트림 URL 반환 |
| `GET /api/video_feed?cctv_id={id}` | MJPEG 분석 스트림 |

## 프로젝트 구조

```
yolo-traffic-monitor/
├── main.py              # 배치 모드
├── server.py            # 웹 서버
├── image_fetcher.py     # 스트림 처리
├── image_analyzer.py    # YOLO 분석
├── utils/
│   ├── logger_util.py
│   └── telegram_util.py
├── static/index.html
├── model/yolo11n.pt
├── output/              # 분석 결과
├── logs/                # 로그
└── .env                 # 환경 변수
```

## 텔레그램 연동

### 봇 생성
1. [@BotFather](https://t.me/BotFather)와 대화
2. `/newbot` 명령어 입력
3. Bot Token 저장

### Chat ID 확인
- **방법 1**: 봇에게 메시지 전송 후 `https://api.telegram.org/bot<TOKEN>/getUpdates` 방문
- **방법 2**: [@userinfobot](https://t.me/userinfobot) 사용

### .env 설정
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

## 사용 기술

| 카테고리 | 기술 스택 |
|---------|----------|
| **AI/ML** | YOLOv11 (Ultralytics 8.3.0) |
| **컴퓨터 비전** | OpenCV 4.10.0 |
| **웹** | FastAPI 0.104.1, Uvicorn 0.24.0 |
| **스트리밍** | Streamlink 8.0.0, FFmpeg |
| **프론트엔드** | Tailwind CSS, hls.js |

## 문제 해결

| 문제 | 해결 방법 |
|-----|---------|
| 프레임 캡처 실패 | FFmpeg 설치 확인, 네트워크 연결 확인 |
| YOLO 모델 로드 실패 | `model/yolo11n.pt` 파일 존재 확인 |
| 텔레그램 전송 실패 | Bot Token/Chat ID 확인, 봇에게 먼저 메시지 전송 |
| 웹 접속 불가 | 포트 8000 사용 여부 확인 |
