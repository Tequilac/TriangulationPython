import cv2

from triangulation_base.triangulation_base import TriangulationBase

import numpy as np

EPS = 1e-5


class IterativeEigen(TriangulationBase):
    def triangulate_point(
            self,
            point_first: tuple[float, float],
            point_second: tuple[float, float],
    ) -> tuple[float, float, float]:
        result = np.array(4)
        w0, w1 = 1, 1
        for _ in range(10):
            matrix = np.zeros((4, 4))
            matrix[0] = (point_first[0] * self.camera_matrix_first[2] - self.camera_matrix_first[0]) / w0
            matrix[1] = (point_first[1] * self.camera_matrix_first[2] - self.camera_matrix_first[1]) / w0
            matrix[2] = (point_second[0] * self.camera_matrix_second[2] - self.camera_matrix_second[0]) / w1
            matrix[3] = (point_second[1] * self.camera_matrix_second[2] - self.camera_matrix_second[1]) / w1

            _, _, eigen_vectors = cv2.eigen(np.matmul(matrix.transpose(), matrix))
            result = eigen_vectors[eigen_vectors.shape[0] - 1].transpose()

            new_w0 = np.matmul(self.camera_matrix_first[2], result)
            new_w1 = np.matmul(self.camera_matrix_second[2], result)
            if (abs(w0 - new_w0) <= EPS) and (abs(w1 - new_w1) <= EPS):
                break
            w0, w1 = new_w0, new_w1
        return result[0] / result[3], result[1] / result[3], result[2] / result[3]
