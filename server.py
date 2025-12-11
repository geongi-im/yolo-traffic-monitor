# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from image_fetcher import ImageFetcher
from utils.logger_util import LoggerUtil
from dotenv import load_dotenv
import os

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


@app.get("/api/hls-url")
async def get_hls_url(cctv_id: int = Query(..., description="CCTV ID (예: 6301, 6300, 6302)")):
    """
    네이버 CCTV API에서 HLS 스트림 URL을 가져옵니다.

    Args:
        cctv_id: CCTV 채널 ID
            - 6301: 강남대로
            - 6300: 테헤란로
            - 6302: 올림픽대로

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

        # 1. 쿠키 획득
        cookies = fetcher._get_cookies()
        if not cookies:
            logger.error("네이버 인증 쿠키 획득 실패")
            raise HTTPException(
                status_code=503,
                detail="네이버 인증 서버에 접속할 수 없습니다"
            )

        # 2. HLS URL 추출
        hls_url = fetcher._get_hls_url(cookies)
        if not hls_url:
            logger.error(f"CCTV ID {cctv_id}의 HLS URL을 찾을 수 없음")
            raise HTTPException(
                status_code=404,
                detail=f"CCTV ID {cctv_id}의 스트림을 찾을 수 없습니다"
            )

        logger.info(f"HLS URL 획득 성공: {hls_url[:50]}...")

        return {
            "success": True,
            "hls_url": hls_url,
            "cctv_id": cctv_id
        }

    except HTTPException:
        # 이미 처리된 HTTP 예외는 그대로 re-raise
        raise
    except Exception as e:
        logger.error(f"HLS URL 획득 중 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류: {str(e)}"
        )


@app.get("/")
async def root():
    """메인 페이지"""
    return FileResponse("static/index.html")


# 정적 파일 마운트 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("FastAPI 서버 시작")
    logger.info("서버 주소: http://localhost:8000")
    logger.info("API 문서: http://localhost:8000/docs")
    logger.info("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
