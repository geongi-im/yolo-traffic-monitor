# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from image_fetcher import ImageFetcher
from utils.logger_util import LoggerUtil
from dotenv import load_dotenv
import os
import glob
import cv2
import threading
import time
from typing import Generator
from image_analyzer import VehicleDetector, ImageAnalyzer

# 환경변수 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(
    title="YOLO Traffic Monitor API",
    description="CCTV HLS 스트림 분석 및 모니터링 API",
    version="1.0.0"
)

# CORS 설정 (모든 오리진 허용 - 개발 환경)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로거 초기화
logger = LoggerUtil().get_logger()

# 스트리머 관리
streaming_active = False

def validate_env_variables():
    """필수 환경변수 체크 및 설정 반환"""
    required_vars = {
        'YOLO_MODEL': os.getenv('YOLO_MODEL'),
        'CONFIDENCE_THRESHOLD': os.getenv('CONFIDENCE_THRESHOLD'),
        'DEVICE': os.getenv('DEVICE'),
    } 
    
    missing = [key for key, value in required_vars.items() if not value]
    if missing:
        raise RuntimeError(f"Server startup failed: Missing required environment variables: {', '.join(missing)}")
    
    return {
        'YOLO_MODEL': required_vars['YOLO_MODEL'],
        'CONFIDENCE_THRESHOLD': float(required_vars['CONFIDENCE_THRESHOLD']),
        'DEVICE': required_vars['DEVICE'],
    }

class VideoStreamer:
    """OpenCV 기반 MJPEG 스트리머"""
    
    def __init__(self, cctv_id: int):
        self.cctv_id = cctv_id
        self.logger = LoggerUtil().get_logger()
        self.detector = None
        self.analyzer = None
        
        # YOLO 모델 초기화
        try:
            config = validate_env_variables()
            self.detector = VehicleDetector(
                config['YOLO_MODEL'],
                config['CONFIDENCE_THRESHOLD'],
                [2, 3, 5, 7],  # car, motorcycle, bus, truck
                config['DEVICE'],
                self.logger
            )
            # ImageAnalyzer 초기화 (output_dir은 스트리밍에 불필요하므로 None)
            self.analyzer = ImageAnalyzer(self.detector, self.logger, output_dir=None)
            
        except Exception as e:
            self.logger.error(f"Detector 초기화 실패: {e}")

    def get_stream_url(self):
        """HLS URL 획득"""
        fetcher = ImageFetcher(self.cctv_id, self.logger)
        cookies = fetcher._get_cookies()
        if not cookies:
            return None
        return fetcher._get_hls_url(cookies)

    def generate_frames(self) -> Generator[bytes, None, None]:
        """프레임 생성 및 분석 (Generator)"""
        hls_url = self.get_stream_url()
        if not hls_url:
            self.logger.error("HLS URL을 가져올 수 없습니다.")
            return

        cap = cv2.VideoCapture(hls_url)
        
        if not cap.isOpened():
            self.logger.error("비디오 스트림을 열 수 없습니다.")
            return

        # FPS 제어
        target_fps = 15
        frame_interval = 1.0 / target_fps
        last_frame_time = 0

        try:
            while True:
                current_time = time.time()
                
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("프레임 읽기 실패, 재연결 시도...")
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(hls_url)
                    continue

                # FPS 제한
                if current_time - last_frame_time < frame_interval:
                    continue
                last_frame_time = current_time

                # YOLO 분석 및 그리기
                if self.analyzer:
                    frame = self.analyzer.process_live_frame(frame)

                # JPEG 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # MJPEG 포맷으로 전송
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            self.logger.error(f"스트리밍 중 오류: {e}")
        finally:
            if cap:
                cap.release()
            self.logger.info("스트리밍 종료")


@app.get("/api/video_feed")
async def video_feed(cctv_id: int):
    """실시간 분석 영상 스트리밍 엔드포인트"""
    streamer = VideoStreamer(cctv_id)
    return StreamingResponse(
        streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/hls-url")
async def get_hls_url(cctv_id: int = Query(..., description="CCTV ID (예: 6301, 6300, 6323)")):
    """
    네이버 CCTV API에서 HLS 스트림 URL을 가져옵니다.

    Args:
        cctv_id: CCTV 채널 ID
            - 6301: 수영로
            - 6300: 자성로
            - 6323: 황령대로

    Returns:
        dict: HLS URL 정보
            - success (bool): 성공 여부
            - hls_url (str): HLS 스트림 URL (.m3u8)
            - cctv_id (int): CCTV ID

    Raises:
        HTTPException: HLS URL 획득 실패 시
    """
    try:
        logger.info(f"HLS URL 요청 - CCTV ID: {cctv_id}")

        # ImageFetcher로 HLS URL 획득
        fetcher = ImageFetcher(cctv_id, logger)
        
        # 쿠키 획득
        cookies = fetcher._get_cookies()
        if not cookies:
            logger.error("네이버 인증 쿠키 획득 실패")
            raise HTTPException(status_code=503, detail="네이버 인증 서버 오류")

        # HLS URL 추출
        hls_url = fetcher._get_hls_url(cookies)
        if not hls_url:
            raise HTTPException(status_code=404, detail="CCTV 스트림을 찾을 수 없습니다")

        return {
            "success": True,
            "hls_url": hls_url,
            "cctv_id": cctv_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")
if __name__ == "__main__":
    import uvicorn
    # 서버 시작 전 환경변수 체크
    try:
        validate_env_variables()
    except RuntimeError as e:
        logger.error(str(e))
        exit(1)
        
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
