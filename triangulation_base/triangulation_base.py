from abc import ABC, abstractmethod

import numpy as np


class TriangulationBase(ABC):
    def __init__(self, camera_matrix_first: np.matrix, camera_matrix_second: np.matrix):
        self.camera_matrix_first = camera_matrix_first
        self.camera_matrix_second = camera_matrix_second

    @abstractmethod
    def triangulate_point(self, point_first: tuple[float, float], point_second: tuple[float, float]):
        ...

    def triangulate(self, points_first: list[tuple[float, float]], points_second: list[tuple[float, float]]):
        result = []
        for point_first, point_second in zip(points_first, points_second):
            result.append(self.triangulate_point(point_first, point_second))
        return result
