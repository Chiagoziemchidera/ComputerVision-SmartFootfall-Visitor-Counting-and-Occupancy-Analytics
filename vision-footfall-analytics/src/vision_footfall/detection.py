from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def yolox_postprocess(
    outputs: np.ndarray, input_size: tuple[int, int], p6: bool = False
) -> np.ndarray:
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    if p6:
        strides.append(64)
    input_height, input_width = input_size
    for stride in strides:
        height_size = input_height // stride
        width_size = input_width // stride
        grid_y, grid_x = np.meshgrid(
            np.arange(height_size), np.arange(width_size), indexing="ij"
        )
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride, dtype=np.float32))
    grids = np.concatenate(grids, axis=1).astype(np.float32)
    expanded_strides = np.concatenate(expanded_strides, axis=1).astype(np.float32)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def preprocess_yolox(
    frame: np.ndarray, input_size: tuple[int, int]
) -> tuple[np.ndarray, float]:
    input_height, input_width = input_size
    frame_height, frame_width = frame.shape[:2]
    scale = min(input_height / frame_height, input_width / frame_width)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    tensor = padded.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor = np.expand_dims(tensor, axis=0)
    return (tensor, scale)


class YOLOXPersonDetector:

    def __init__(
        self,
        model_path: str,
        input_width: int,
        input_height: int,
        confidence_threshold: float,
        nms_threshold: float,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLOX model was not found: {model_path}")
        self.input_size = (int(input_height), int(input_width))
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        available_providers = ort.get_available_providers()
        preferred_providers = [
            provider
            for provider in [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            if provider in available_providers
        ]
        self.session = ort.InferenceSession(model_path, providers=preferred_providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("YOLOX loaded with providers: %s", self.session.get_providers())

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, scale = preprocess_yolox(frame, self.input_size)
        raw_output = self.session.run(None, {self.input_name: tensor})[0]
        predictions = yolox_postprocess(raw_output, self.input_size)[0]
        if predictions.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        centre_x = predictions[:, 0]
        centre_y = predictions[:, 1]
        width = predictions[:, 2]
        height = predictions[:, 3]
        boxes = np.column_stack(
            (
                centre_x - width / 2,
                centre_y - height / 2,
                centre_x + width / 2,
                centre_y + height / 2,
            )
        )
        object_confidence = predictions[:, 4]
        class_scores = predictions[:, 5:]
        if class_scores.shape[1] == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        person_confidence = object_confidence * class_scores[:, 0]
        keep = person_confidence >= self.confidence_threshold
        boxes = boxes[keep]
        person_confidence = person_confidence[keep]
        if len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        boxes = boxes / scale
        frame_height, frame_width = frame.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, frame_width - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, frame_height - 1)
        nms_boxes = []
        for x1, y1, x2, y2 in boxes:
            nms_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        indexes = cv2.dnn.NMSBoxes(
            nms_boxes,
            person_confidence.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
        )
        if len(indexes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        indexes = np.asarray(indexes).reshape(-1)
        return (
            boxes[indexes].astype(np.float32),
            person_confidence[indexes].astype(np.float32),
        )
