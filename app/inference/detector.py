from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO

from app.config.settings import IMAGE_SIZE, LOW_CONFIDENCE_FLOOR


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    original_image: Image.Image
    annotated_image: Image.Image
    detections: list[Detection]
    low_confidence_detections: list[Detection]

    @property
    def note_count(self) -> int:
        return len(self.detections)

    @property
    def best_confidence(self) -> float:
        all_detections = self.detections or self.low_confidence_detections
        if not all_detections:
            return 0.0
        return max(detection.confidence for detection in all_detections)


class BanknoteDetector:
    """Small wrapper around the trained YOLO model."""

    def __init__(self, model_path: Path, class_labels: dict[int, str]) -> None:
        self.model_path = model_path
        self.class_labels = class_labels
        self.model = YOLO(str(model_path))

    def predict_image(self, image: Image.Image, confidence_threshold: float) -> DetectionResult:
        """Run inference on a PIL image and return display-ready results."""
        original = image.convert("RGB")
        source = np.asarray(original)
        results = self.model.predict(
            source=source,
            conf=LOW_CONFIDENCE_FLOOR,
            imgsz=IMAGE_SIZE,
            save=False,
            verbose=False,
        )
        result = results[0]

        detections: list[Detection] = []
        low_confidence: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            detection = Detection(
                label=self.class_labels.get(class_id, f"Class {class_id}"),
                confidence=confidence,
                box=xyxy,
            )
            if confidence >= confidence_threshold:
                detections.append(detection)
            else:
                low_confidence.append(detection)

        annotated = self._draw_annotations(original, detections)
        return DetectionResult(
            original_image=original,
            annotated_image=annotated,
            detections=detections,
            low_confidence_detections=low_confidence,
        )

    def _draw_annotations(self, image: Image.Image, detections: list[Detection]) -> Image.Image:
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.box)
            label = f"{detection.label} {detection.confidence:.0%}"
            color = (31, 226, 144)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
            self._draw_label(canvas, label, x1, y1, color)

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _draw_label(canvas: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.82
        thickness = 2
        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = text_size
        top = max(y - text_height - baseline - 12, 0)
        cv2.rectangle(canvas, (x, top), (x + text_width + 18, top + text_height + baseline + 12), color, -1)
        cv2.putText(canvas, label, (x + 9, top + text_height + 5), font, font_scale, (9, 15, 28), thickness, cv2.LINE_AA)

    def predict_frame(self, frame: np.ndarray, confidence_threshold: float) -> DetectionResult:
        """Future camera path: accept an RGB frame and reuse the same inference flow."""
        return self.predict_image(Image.fromarray(frame), confidence_threshold)
