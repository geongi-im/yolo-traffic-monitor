import os
import threading
import shutil
from datetime import datetime
from dotenv import load_dotenv
from image_fetcher import ImageFetcher
from image_analyzer import VehicleDetector, ImageAnalyzer
from utils.logger_util import LoggerUtil
from utils.telegram_util import TelegramUtil

# CCTV 설정 (상수)
CCTV_ID = 6301  # 강남대로

# 환경변수 로드
load_dotenv()

def validate_env_variables():
    """필수 환경변수 체크"""
    required_vars = {
        'YOLO_MODEL': os.getenv('YOLO_MODEL'),
        'CONFIDENCE_THRESHOLD': os.getenv('CONFIDENCE_THRESHOLD'),
        'DEVICE': os.getenv('DEVICE'),
        'INTERVAL_SECONDS': os.getenv('INTERVAL_SECONDS'),
        'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
    }

    missing = [key for key, value in required_vars.items() if not value]

    if missing:
        raise ValueError(f"필수 환경변수가 설정되지 않음: {', '.join(missing)}")

    # 타입 체크를 통과한 값들만 변환
    return {
        'CCTV_ID': CCTV_ID,  # 상수 사용
        'YOLO_MODEL': required_vars['YOLO_MODEL'] or '',
        'CONFIDENCE_THRESHOLD': float(required_vars['CONFIDENCE_THRESHOLD']) if required_vars['CONFIDENCE_THRESHOLD'] else 0.0,
        'DEVICE': required_vars['DEVICE'] or 'cpu',
        'INTERVAL_SECONDS': int(required_vars['INTERVAL_SECONDS']) if required_vars['INTERVAL_SECONDS'] else 60,
    }

def create_directories():
    """필요한 디렉토리 생성"""
    os.makedirs('temp', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

class CCTVAnalyzer:
    """CCTV 이미지 분석 메인 클래스"""

    def __init__(self, config):
        self.config = config
        self.logger = LoggerUtil().get_logger()
        self.telegram = TelegramUtil()
        self.timer = None
        self.iteration = 0

        # 컴포넌트 초기화
        self.fetcher = ImageFetcher(config['CCTV_ID'], self.logger)
        detector = VehicleDetector(
            config['YOLO_MODEL'],
            config['CONFIDENCE_THRESHOLD'],
            [2, 3, 5, 7],  # car, motorcycle, bus, truck
            config['DEVICE'],
            self.logger
        )
        self.analyzer = ImageAnalyzer(detector, self.logger, 'output')

    def run_analysis(self):
        """한 번의 분석 사이클 실행"""
        self.iteration += 1
        self.logger.info(f"\n--- Iteration {self.iteration} ---")

        temp_dir = None

        try:
            # 1. 영상 전체 프레임 캡처 (1초 간격)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = os.path.join('temp', f"batch_{timestamp}")

            self.logger.info("HLS 스트림에서 프레임 캡처 시작...")
            frame_paths = self.fetcher.fetch_and_download(temp_dir)

            if not frame_paths:
                self.logger.error("프레임 캡처 실패")
                return

            self.logger.info(f"{len(frame_paths)}개 프레임 캡처 완료")

            # 2. 멀티프레임 YOLO 분석 (output 폴더에 저장)
            result = self.analyzer.analyze_multiple_frames(frame_paths)

            self.logger.info(
                f"분석 완료 - 평균 차량: {result['avg_vehicle_count']:.1f}대 "
                f"(프레임별: {result['frame_counts']})"
            )

        except Exception as e:
            error_msg = f"분석 중 오류 발생: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # 오류 발생 시 텔레그램 알림
            try:
                self.telegram.send_message(f"❌ {error_msg}")
            except Exception as telegram_error:
                self.logger.error(f"텔레그램 전송 실패: {telegram_error}")

        finally:
            # 3. temp 배치 폴더 전체 삭제
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"임시 폴더 삭제: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"임시 폴더 삭제 실패: {e}")

            # 4. 다음 실행 예약
            self.schedule_next()

    def schedule_next(self):
        """다음 분석 예약"""
        self.timer = threading.Timer(
            self.config['INTERVAL_SECONDS'],
            self.run_analysis
        )
        self.timer.daemon = True
        self.timer.start()
        self.logger.info(f"{self.config['INTERVAL_SECONDS']}초 후 다음 분석 예정")

    def start(self):
        """분석 시작"""
        self.logger.info("=" * 60)
        self.logger.info("YOLO CCTV Image Analysis Started")
        self.logger.info("=" * 60)

        # 즉시 첫 분석 실행
        self.run_analysis()

        # 메인 스레드 유지
        try:
            while True:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            self.logger.info("\n사용자에 의해 중단됨")
            if self.timer:
                self.timer.cancel()

    def stop(self):
        """분석 중지"""
        if self.timer:
            self.timer.cancel()
        self.logger.info("=" * 60)
        self.logger.info("Application Stopped")
        self.logger.info("=" * 60)

def main():
    logger = LoggerUtil().get_logger()

    try:
        # 1. 환경변수 검증
        config = validate_env_variables()
        logger.info("환경변수 검증 완료")

        # 2. 디렉토리 생성
        create_directories()
        logger.info("디렉토리 생성 완료")

        # 3. 분석 시작
        analyzer = CCTVAnalyzer(config)
        analyzer.start()

    except ValueError as e:
        logger.error(f"환경변수 오류: {e}")
        logger.error("프로그램을 종료합니다.")
        return
    except Exception as e:
        logger.error(f"초기화 오류: {e}", exc_info=True)

        # 초기화 오류 시 텔레그램 알림
        try:
            telegram = TelegramUtil()
            telegram.send_message(f"❌ 초기화 실패: {str(e)}")
        except:
            pass

if __name__ == "__main__":
    main()