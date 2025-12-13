import requests
import os
import cv2



class ImageFetcher:
    """Naver CCTV API에서 HLS 스트림 URL을 가져와 첫 프레임을 캡처하는 클래스"""

    def __init__(self, cctv_id, logger):
        """
        Args:
            cctv_id: CCTV 채널 ID (예: 6301)
            logger: LoggerUtil 인스턴스
        """
        self.cctv_id = cctv_id
        self.logger = logger
        self.naver_auth_url = "https://nam.veta.naver.com/nac/1"
        self.cctv_api_url = f"https://map.naver.com/p/api/cctv?cctvId={cctv_id}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://map.naver.com/"
        }

    def _get_cookies(self):
        """Naver 인증 쿠키 획득"""
        try:
            response = requests.get(self.naver_auth_url)
            self.logger.debug(f"쿠키 획득 완료: {len(response.cookies)} 개")
            return response.cookies
        except Exception as e:
            self.logger.error(f"쿠키 획득 실패: {e}")
            return None

    def _get_hls_url(self, cookies):
        """CCTV API에서 HLS URL 추출"""
        try:
            response = requests.get(
                self.cctv_api_url,
                cookies=cookies,
                headers=self.headers
            )

            if response.status_code != 200:
                self.logger.error(f"API 호출 실패: HTTP {response.status_code}")
                return None

            data = response.json()
            cctv_list = data.get('message', {}).get('result', {}).get('cctvList', [])

            for cctv in cctv_list:
                if cctv.get('channel') == self.cctv_id:
                    hls_url = cctv.get('hlsUrl')
                    if hls_url:
                        self.logger.debug(f"HLS URL 발견: {hls_url[:50]}...")
                        return hls_url

            self.logger.error(f"Channel {self.cctv_id}를 찾을 수 없음")
            return None

        except Exception as e:
            self.logger.error(f"HLS URL 추출 실패: {e}")
            return None

    def _capture_all_frames_by_duration(self, hls_url, temp_dir):
        """
        HLS 스트림 영상 길이를 계산하여 1초 간격으로 모든 프레임 캡처

        Args:
            hls_url: HLS 스트림 URL
            temp_dir: 저장할 임시 디렉토리

        Returns:
            List[str]: 저장된 프레임 경로 리스트
        """
        cap = None
        saved_paths = []

        try:
            # Streamlink 의존성 제거: OpenCV가 HLS를 직접 처리하도록 변경
            self.logger.info(f"HLS 스트림 연결 시도: {hls_url}")
            
            # VideoCapture로 스트림 열기 (FFmpeg 백엔드 사용)
            cap = cv2.VideoCapture(hls_url)

            if not cap.isOpened():
                self.logger.error("VideoCapture 열기 실패 (스트림을 찾을 수 없거나 코덱 지원 안됨)")
                return []

            # temp 디렉토리 생성
            os.makedirs(temp_dir, exist_ok=True)

            # FPS 및 영상 길이 계산
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 15  # HLS 스트림의 실제 FPS (테스트 결과)
                self.logger.warning(f"FPS를 확인할 수 없어 기본값 {fps} 사용")
            else:
                self.logger.debug(f"스트림 FPS: {fps}")

            # 영상을 전부 읽으면서 실제 길이 계산
            # HLS 라이브 스트림은 CAP_PROP_FRAME_COUNT가 정확하지 않으므로
            # 실제로 읽을 수 있는 프레임 수를 세어서 계산
            self.logger.info("영상 길이 계산 중...")

            frame_count = 0
            skip_frames = int(fps * 1.0)  # 1초 간격

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # 1초마다 프레임 저장
                if frame_count % skip_frames == 0:
                    frame_index = frame_count // skip_frames
                    save_path = os.path.join(temp_dir, f"frame_{frame_index}.jpg")
                    success = cv2.imwrite(save_path, frame)

                    if success:
                        saved_paths.append(save_path)
                        self.logger.debug(f"프레임 {frame_index} 저장 ({frame_count // skip_frames}초)")

                frame_count += 1

                # 안전장치: 최대 15초까지만 (무한루프 방지)
                if frame_count > fps * 15:
                    self.logger.warning("15초 제한 도달, 캡처 중단")
                    break

            duration = frame_count / fps
            self.logger.info(
                f"영상 길이: {duration:.1f}초 "
                f"({len(saved_paths)}개 프레임 캡처 완료)"
            )

            return saved_paths

        except Exception as e:
            self.logger.error(f"프레임 캡처 중 오류: {e}", exc_info=True)
            return []

        finally:
            if cap is not None:
                cap.release()
                self.logger.debug("VideoCapture 리소스 해제됨")

    def fetch_and_download(self, temp_dir):
        """
        HLS 스트림에서 영상 길이를 자동 계산하여 1초 간격으로 프레임 샘플링

        Args:
            temp_dir: 프레임을 저장할 임시 디렉토리

        Returns:
            List[str]: 저장된 프레임 경로 리스트 (실패 시 빈 리스트)
        """
        try:
            # 1. 쿠키 획득
            cookies = self._get_cookies()
            if not cookies:
                return []

            # 2. HLS URL 가져오기
            hls_url = self._get_hls_url(cookies)
            if not hls_url:
                return []

            # 3. 영상 길이 계산 후 프레임 캡처
            return self._capture_all_frames_by_duration(hls_url, temp_dir)

        except Exception as e:
            self.logger.error(f"프레임 캡처 프로세스 실패: {e}", exc_info=True)
            return []
