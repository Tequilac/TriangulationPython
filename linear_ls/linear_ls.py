import numpy as np

from solveZ.solveZ import solveZ
from triangulation_base.triangulation_base import TriangulationBase


class LinearLS(TriangulationBase):
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
        result = solveZ(matrix)
        return result[0]/result[3], result[1]/result[3], result[2]/result[3]
