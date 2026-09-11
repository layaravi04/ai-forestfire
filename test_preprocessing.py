import pytest
import numpy as np
from preprocessing.image_processor import ImageProcessor
from preprocessing.sequence_generator import SequenceBuffer

def test_image_processor_output_shape():
    processor = ImageProcessor(target_size=(224, 224))
    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    tensor_or_arr, rgb_out = processor.preprocess_image(dummy_img)
    assert tensor_or_arr.shape == (1, 3, 224, 224)
    assert rgb_out.shape == (300, 400, 3)

def test_sequence_buffer():
    buffer = SequenceBuffer(sequence_length=8)
    for _ in range(8):
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        buffer.push_frame(dummy_img)
    
    assert buffer.is_full()
    seq_tensor = buffer.get_sequence_tensor()
    assert seq_tensor.shape == (1, 8, 3, 224, 224)
