import cv2
import os
from datetime import datetime
from ultralytics import YOLO


class VehicleDetector:
    """YOLO 기반 차량 탐지기"""

    def __init__(self, model_path, confidence_threshold, vehicle_classes, device, logger):
        self.logger = logger
        self.conf_threshold = confidence_threshold
        self.vehicle_classes = vehicle_classes
        self.device = device

        # FP16 half-precision support (GPU only)
        self.use_half = (device == 'cuda')

        # Load YOLO model (auto-downloads if not exists)
        self.logger.info(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)

            # GPU half-precision optimization
            if self.use_half:
                self.model.model.half()
                self.logger.info("Using FP16 half-precision inference")

            self.logger.info("YOLO model loaded successfully")

            # Log device info
            if device == 'cuda':
                self.logger.info("Using GPU for inference")
            else:
                self.logger.info("Using CPU for inference")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image_path):
        """
        이미지 파일에서 차량 탐지

        Args:
            image_path: 분석할 이미지 파일 경로

        Returns:
            tuple: (annotated_image, detections)
        """
        try:
            # 이미지 읽기
            frame = cv2.imread(image_path)
            if frame is None:
                self.logger.error(f"이미지 읽기 실패: {image_path}")
                return None, []

            # P0 OPTIMIZATION: Input resizing to 640x640
            # Save original frame size
            h, w = frame.shape[:2]
            inference_size = 640

            # Resize maintaining aspect ratio
            if h > w:
                new_h = inference_size
                new_w = int(w * (inference_size / h))
            else:
                new_w = inference_size
                new_h = int(h * (inference_size / w))

            resized_frame = cv2.resize(frame, (new_w, new_h))

            # Run inference on resized frame
            results = self.model(
                resized_frame,
                conf=self.conf_threshold,
                classes=self.vehicle_classes,
                device=self.device,
                verbose=False,
                imgsz=inference_size,
                half=self.use_half
            )

            # Scale coordinates back to original size
            detections = []
            scale_x = w / new_w
            scale_y = h / new_h

            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy[0].tolist()
                    scaled_bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]

                    detections.append({
                        'class_id': int(box.cls),
                        'class_name': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': scaled_bbox
                    })

            # Get annotated frame and resize back to original size
            annotated_resized = results[0].plot()
            annotated_frame = cv2.resize(annotated_resized, (w, h))

            return annotated_frame, detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return None, []


class ImageAnalyzer:
    """이미지 분석 및 결과 처리"""

    def __init__(self, detector, logger, output_dir):
        self.detector = detector
        self.logger = logger
        self.output_dir = output_dir

    def _draw_compact_stats(self, frame, detections, avg_vehicle_count=None):
        """
        우측하단에 차량 통계를 작고 깔끔하게 표시
        폰트 크기는 이미지 높이의 약 10% 정도

        Args:
            frame: 프레임 이미지
            detections: 탐지 결과 리스트
            avg_vehicle_count: 평균 차량 수 (옵션, 멀티프레임 분석 시 사용)
        """
        if not detections and avg_vehicle_count is None:
            # 차량이 없어도 "Vehicles: 0" 표시
            h, w = frame.shape[:2]
            font_scale = max(0.3, h / 2000)  # 이미지 크기에 따라 폰트 크기 조정
            thickness = max(1, int(h / 1000))

            text = "Vehicles: 0"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 우측하단 위치
            padding = int(h * 0.02)  # 2% 패딩
            x = w - text_width - padding
            y = h - padding

            # 반투명 배경
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x - padding // 2, y - text_height - padding),
                (w, h),
                (0, 0, 0),
                -1
            )
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # 텍스트 그리기
            cv2.putText(
                frame, text, (x, y - padding // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
            return frame

        # 차량 종류별 카운트
        vehicle_counts = {}
        for det in detections:
            class_name = det['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

        # 통계 텍스트 생성
        stats_text = []

        # 평균 차량 수가 있으면 최상단에 표시
        if avg_vehicle_count is not None:
            stats_text.append(f"Avg Vehicles: {avg_vehicle_count:.1f}")
            stats_text.append(f"(Frame: {len(detections)})")  # 현재 프레임 차량 수
        else:
            stats_text.append(f"Vehicles: {len(detections)}")  # "car: 9" 대신 "Vehicles: 9"

        for class_name, count in sorted(vehicle_counts.items()):
            stats_text.append(f"  {class_name}: {count}")  # 들여쓰기로 구분

        # 이미지 크기에 따른 폰트 크기 계산 (약 10%)
        h, w = frame.shape[:2]
        font_scale = max(0.3, h / 2000)  # 최소 0.3, 이미지가 클수록 커짐
        thickness = max(1, int(h / 1000))

        # 텍스트 크기 계산
        max_text_width = 0
        total_text_height = 0
        line_spacing = int(font_scale * 30)

        for text in stats_text:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            max_text_width = max(max_text_width, text_width)
            total_text_height += text_height + line_spacing

        # 우측하단 위치 계산
        padding = int(h * 0.02)  # 이미지 높이의 2%를 패딩으로
        box_width = max_text_width + padding * 2
        box_height = total_text_height + padding

        x = w - box_width - padding
        y = h - box_height - padding

        # 반투명 배경 그리기
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # 텍스트 그리기
        y_offset = y + padding
        for text in stats_text:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.putText(
                frame, text, (x + padding, y_offset + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )
            y_offset += text_height + line_spacing

        return frame

    def analyze(self, image_path):
        """
        단일 이미지 분석 및 결과 저장

        Args:
            image_path: temp 폴더의 이미지 경로

        Returns:
            dict: {
                'vehicle_count': int,
                'detections': list,
                'saved_image_path': str
            }
        """
        try:
            # 1. YOLO 탐지
            annotated_frame, detections = self.detector.detect(image_path)

            if annotated_frame is None:
                self.logger.error("탐지 실패")
                return {'vehicle_count': 0, 'detections': [], 'saved_image_path': None}

            # 2. 차량 통계만 우측하단에 표시 (FPS, Time 제거)
            display_frame = self._draw_compact_stats(annotated_frame, detections)

            # 3. output 폴더에 저장
            os.makedirs(self.output_dir, exist_ok=True)
            save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"analyzed_{save_timestamp}.jpg")

            cv2.imwrite(save_path, display_frame)
            self.logger.info(f"분석 결과 저장: {save_path}")

            return {
                'vehicle_count': len(detections),
                'detections': detections,
                'saved_image_path': save_path
            }

        except Exception as e:
            self.logger.error(f"분석 중 오류: {e}", exc_info=True)
            return {'vehicle_count': 0, 'detections': [], 'saved_image_path': None}

    def analyze_multiple_frames(self, image_paths):
        """
        여러 프레임 분석 후 평균 계산 및 결과 저장

        Args:
            image_paths: 분석할 이미지 경로 리스트

        Returns:
            dict: {
                'avg_vehicle_count': float,
                'frame_counts': list,
                'saved_image_path': str
            }
        """
        try:
            if not image_paths:
                self.logger.error("분석할 이미지가 없음")
                return {'avg_vehicle_count': 0, 'frame_counts': [], 'saved_image_path': None}

            all_detections = []
            vehicle_counts = []

            # 각 프레임 분석
            self.logger.info(f"{len(image_paths)}개 프레임 YOLO 분석 시작...")
            for idx, path in enumerate(image_paths):
                _, detections = self.detector.detect(path)
                all_detections.append(detections)
                vehicle_counts.append(len(detections))
                self.logger.debug(f"프레임 {idx+1}/{len(image_paths)}: {len(detections)}대")

            # 평균 계산
            avg_count = sum(vehicle_counts) / len(vehicle_counts) if vehicle_counts else 0
            self.logger.info(f"프레임별 차량 수: {vehicle_counts}, 평균: {avg_count:.1f}대")

            # 중간 프레임 선택하여 저장용 이미지로 사용
            middle_idx = len(image_paths) // 2
            middle_frame, middle_detections = self.detector.detect(image_paths[middle_idx])

            if middle_frame is None:
                self.logger.error("중간 프레임 탐지 실패")
                return {
                    'avg_vehicle_count': avg_count,
                    'frame_counts': vehicle_counts,
                    'saved_image_path': None
                }

            # 평균값으로 UI 표시
            display_frame = self._draw_compact_stats(
                middle_frame,
                middle_detections,
                avg_vehicle_count=avg_count
            )

            # output 폴더에 저장
            os.makedirs(self.output_dir, exist_ok=True)
            save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"analyzed_{save_timestamp}.jpg")

            cv2.imwrite(save_path, display_frame)
            self.logger.info(f"분석 결과 저장: {save_path}")

            return {
                'avg_vehicle_count': avg_count,
                'frame_counts': vehicle_counts,
                'saved_image_path': save_path
            }

        except Exception as e:
            self.logger.error(f"멀티프레임 분석 중 오류: {e}", exc_info=True)
            return {'avg_vehicle_count': 0, 'frame_counts': [], 'saved_image_path': None}
