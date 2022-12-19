import numpy as np
import cv2


def solveZ(A: np.ndarray):
    mat = cv2.Mat(A.astype(np.float64))
    rows, cols = mat.shape
    flags = 0 if rows >= cols else cv2.SVD_FULL_UV
    _, _, vt = cv2.SVDecomp(mat, flags=flags)
    return vt[-1]
