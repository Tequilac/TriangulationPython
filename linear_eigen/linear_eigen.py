import cv2
import numpy as np

from triangulation_base.triangulation_base import TriangulationBase


class LinearEigen(TriangulationBase):
    def triangulate_point(
            self,
            point_first: tuple[float, float],
            point_second: tuple[float, float],
    ) -> tuple[float, float, float]:
        matrix = np.zeros((4, 4))
        matrix[0] = point_first[0] * self.camera_matrix_first[2] - self.camera_matrix_first[0]
        matrix[1] = point_first[1] * self.camera_matrix_first[2] - self.camera_matrix_first[1]
        matrix[2] = point_second[0] * self.camera_matrix_second[2] - self.camera_matrix_second[0]
        matrix[3] = point_second[1] * self.camera_matrix_second[2] - self.camera_matrix_second[1]
        _, _, eigen_vectors = cv2.eigen(np.matmul(matrix.transpose(), matrix))
        tmp = eigen_vectors[eigen_vectors.shape[0] - 1]
        return tmp[0]/tmp[3], tmp[1]/tmp[3], tmp[2]/tmp[3]
