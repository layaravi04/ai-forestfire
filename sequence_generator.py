import cv2
import numpy as np
from typing import List, Tuple, Optional, Union, Any
from preprocessing.image_processor import ImageProcessor

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

class SequenceBuffer:
    def __init__(self, sequence_length: int = 8, target_size: Tuple[int, int] = (224, 224)):
        self.sequence_length = sequence_length
        self.processor = ImageProcessor(target_size=target_size)
        self.buffer: List[Any] = []
        self.raw_frames: List[np.ndarray] = []

    def push_frame(self, frame: Union[np.ndarray, str]) -> None:
        tensor_or_arr, rgb_frame = self.processor.preprocess_image(frame)
        if HAS_TORCH:
            self.buffer.append(tensor_or_arr.squeeze(0))
        else:
            self.buffer.append(np.squeeze(tensor_or_arr, axis=0))
        self.raw_frames.append(rgb_frame)

        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
            self.raw_frames.pop(0)

    def is_full(self) -> bool:
        return len(self.buffer) == self.sequence_length

    def get_sequence_tensor(self) -> Any:
        if not self.buffer:
            raise ValueError("Sequence buffer is empty")
        
        current_seq = list(self.buffer)
        while len(current_seq) < self.sequence_length:
            current_seq.insert(0, current_seq[0])
        
        if HAS_TORCH:
            stacked = torch.stack(current_seq, dim=0)
            return stacked.unsqueeze(0)
        else:
            stacked = np.stack(current_seq, axis=0)
            return np.expand_dims(stacked, axis=0)

    def reset(self) -> None:
        self.buffer.clear()
        self.raw_frames.clear()

def extract_video_frames(video_path: str, max_frames: int = 30, sample_rate: int = 2) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        frame_count += 1
        
    cap.release()
    return frames
