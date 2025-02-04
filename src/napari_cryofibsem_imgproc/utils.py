import cv2
import numpy as np

dtype_to_cvtype = {
    np.dtype(np.uint8): cv2.CV_8U,
    np.dtype(np.int8): cv2.CV_8S,
    np.dtype(np.uint16): cv2.CV_16U,
    np.dtype(np.int16): cv2.CV_16S,
    np.dtype(np.int32): cv2.CV_32S,
    np.dtype(np.float16): cv2.CV_16F,
    np.dtype(np.float32): cv2.CV_32F,
    np.dtype(np.float64): cv2.CV_64F,
}

def restore_original_type(img, dtype):
    processed_slice = None
    if np.isdtype(dtype, "integral"):
        processed_slice = cv2.normalize(img, None, alpha=np.iinfo(dtype).min, beta=np.iinfo(dtype).max, norm_type=cv2.NORM_MINMAX,
                                             dtype=dtype_to_cvtype[dtype])
    elif np.isdtype(dtype, "real floating"):
        processed_slice = cv2.normalize(img, None, alpha=-0.99, beta=0.99, norm_type=cv2.NORM_MINMAX,
                                             dtype=dtype_to_cvtype[dtype])
    return processed_slice

def clip_to_dtype(img, dtype):
    processed_slice = None
    if np.isdtype(dtype, "integral"):
        processed_slice = np.clip(img, a_min=np.iinfo(dtype).min, a_max=np.iinfo(dtype).max).astype(dtype)
    elif np.isdtype(dtype, "real floating"):
        processed_slice = np.clip(img, a_min=np.finfo(dtype).min, a_max=np.finfo(dtype).max).astype(dtype)
    return processed_slice
