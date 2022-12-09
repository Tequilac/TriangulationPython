from linear_ls.linear_ls import LinearLS
from triangulation_base.triangulation_base import TriangulationBase

import numpy as np
import cv2
from numpy.linalg import inv


class PolyBase(TriangulationBase):
    def __init__(self, camera_matrix_first: np.matrix, camera_matrix_second: np.matrix, f: np.matrix = None):
        super().__init__(camera_matrix_first, camera_matrix_second)
        self.linear_ls = LinearLS(camera_matrix_first, camera_matrix_second)
        self.f = f if f is not None else self.compute_fundamental_matrix(camera_matrix_first, camera_matrix_second)

    def compute_fundamental_matrix(self, camera_matrix_first: np.matrix, camera_matrix_second: np.matrix) \
            -> np.matrix:
        K0, K1, R, T = self.set_origin_to_camera(camera_matrix_first, camera_matrix_second)

        A = K0 * R.transpose() * T

        C = np.zeros((3, 3))
        C[0, 1] = -A[2]
        C[0, 2] = A[1]
        C[1, 0] = A[2]
        C[1, 2] = -A[0]
        C[2, 0] = -A[1]
        C[2, 1] = A[0]

        return inv(K1).transpose() * R * K0.transpose()

    def set_origin_to_camera(self, camera_matrix_first: np.matrix, camera_matrix_second: np.matrix):
        K0, R0, T0, _, _, _, _ = cv2.decomposeProjectionMatrix(camera_matrix_first)
        K1, R1, T1, _, _, _, _ = cv2.decomposeProjectionMatrix(camera_matrix_second)

        M = np.identity(4)

        M[0:3, 0:3] = inv(R0)
        M[0, 3] = T0[0] / T0[3]
        M[1, 3] = T0[1 / T0[3]]
        M[2, 3] = T0[2] / T0[3]

        tmp = inv(K1) * camera_matrix_second * M

        R = tmp[0:3, 0:3]
        T = tmp[3, 0:3]

        return K0, K1, R, T

    def triangulate_point(self, point_first: tuple[float, float], point_second: tuple[float, float]) \
            -> tuple[float, float, float]:
        x0, x1 = self.compute_corrected_correspondences(point_first, point_second)
        return self.linear_ls.triangulate_point(x0, x1)

    def compute_corrected_correspondences(self, point_first: tuple[float, float], point_second: tuple[float, float]) \
            -> tuple[tuple[float, float], tuple[float, float]]:
        T0 = self.translate_to_origin(point_first)
        T1 = self.translate_to_origin(point_second)

        f = T1.transpose() * self.f * T0

        e0 = self.compute_right_epipole(f)
        e1 = self.compute_left_epipole(f)

        R0 = self.form_rotation_matrix(e0)
        R1 = self.form_rotation_matrix(e1)

        self.f = R1 * self.f * R0.transpose()

        params = (f[1, 1], f[1, 2], f[2, 1], f[2, 2], e0[2], e1[2])

        roots = self.solve(params)

        t = self.evaluate_roots(roots, params)

        l0, l1 = self.construct_lines(t, params)

        x0 = self.find_point_on_line_closest_to_origin(l0)
        x1 = self.find_point_on_line_closest_to_origin(l1)

        x0 = self.transfer_point_to_original_coordinates(x0, R0, T0)
        x1 = self.transfer_point_to_original_coordinates(x1, R1, T1)

        return (x0[0] / x0[2], x0[1] / x0[2]), (x1[0] / x1[2], x1[1] / x1[2])

    def translate_to_origin(self, point: tuple[float, float]) \
            -> np.matrix:
        result = np.identity(3)
        result[0, 2] = point[0]
        result[1, 2] = point[1]
        return np.matrix(result)

    def compute_left_epipole(self, f: np.matrix) \
            -> tuple[float, float, float]:
        _, _, VT = cv2.SVDecomp(f.transpose())
        return VT[2, 0], VT[2, 1], VT[2, 2]

    def compute_right_epipole(self, f: np.matrix) \
            -> tuple[float, float, float]:
        _, _, VT = cv2.SVDecomp(f)
        return VT[2, 0], VT[2, 1], VT[2, 2]

    def form_rotation_matrix(self, epipole: tuple[float, float, float]) \
            -> np.matrix:
        result = np.identity(3)
        result[0, 0] = epipole[0]
        result[0, 1] = epipole[1]
        result[1, 0] = -epipole[1]
        result[1, 1] = -epipole[0]
        return np.matrix(result)

    def find_polynominal_order(self, coeffs: np.ndarray) \
            -> int:
        for idx, coeff in enumerate(coeffs):
            if coeff != 0:
                return idx
        return -1

    def solve(self, params: tuple[float, float, float, float, float, float]) \
            -> list[any]:
        coeffs = self.prepare_poly_coeffs(params)
        if len(coeffs) <= 1:
            return [0]

        roots = np.roots(coeffs)

        return [(roots[i, 0], roots[i, 1]) for i in range(roots)]

    def evaluate_roots(self, roots: list[any], params: tuple[float, float, float, float, float, float]) \
            -> float:
        costs = self.evaluate_roots_costs(roots, params)
        return np.amin(costs)

    def prepare_poly_coeffs(self, params: tuple[float, float, float, float, float, float]) -> np.ndarray:
        (a, b, c, d, e, f) = params
        result = np.array([
            a * b * d * d - b * b * c * d,
            -2 * b * b * d * d * f * f + a * a * d * d - d * d * d * d * f * f * f * f - b * b * c * c - b * b * b * b,
            -4 * b * b * c * d * f * f - 2 * b * b * c * d * e * e - 4 * a * b * d * d * f * f + 2 * a * b * d * d * e * e + a * a * c * d - 4 * c * d * d * d * f * f * f * f - a * b * c * c - 4 * a * b * b * b,
            -8 * a * b * c * d * f * f - 6 * c * c * d * d * f * f * f * f - 2 * b * b * c * c * f * f - 2 * a * a * d * d * f * f - 2 * b * b * c * c * e * e + 2 * a * a * d * d * e * e - 6 * a * a * b * b,
            -4 * a * a * c * d * f * f + 2 * a * a * c * d * e * e - 4 * a * b * c * c * f * f + a * b * d * d * e * e * e * e - 2 * a * b * c * c * e * e - b * b * c * d * e * e * e * e - 4 * c * c * c * d * f * f * f * f - 4 * a * a * a * b,
            a * a * d * d * e * e * e * e - 2 * a * a * c * c * f * f - b * b * c * c * e * e * e * e - c * c * c * c * f * f * f * f - a * a * a * a,
            a * a * c * d * e * e * e * e - a * b * c * c * e * e * e * e])

        result = np.resize(result, (1, self.find_polynominal_order(result)) + 1)

        max_coeff = np.amax(result)

        if max_coeff > 0:
            lam = lambda x: x / max_coeff
            result = lam(result)
        elif max_coeff < 0:
            min_coeff = np.amin(result)
            lam = lambda x: x / max_coeff
            result = lam(result)

        return result

    def evaluate_roots_costs(self, roots: list[any], params: tuple[float, float, float, float, float, float]) \
            -> np.ndarray:
        (a, b, c, d, e, f) = params
        cost_lambda = \
            lambda t: ((t * t) / (1 + e * e * t * t)) + ((c * t + d) * (c * t + d)) / (
                    (a * t + b) * (a * t + b) + f * f * (c * t + d) * (c * t + d))

        return np.array([cost_lambda(root) for root in roots])

    def construct_lines(self, t: float, params: tuple[float, float, float, float, float, float]) \
            -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        (a, b, c, d, e, f) = params
        l0 = (t * e, 1, -t)
        l1 = (-f * (c * t * d), a * t + b, c * t + d)
        return l0, l1

    def find_point_on_line_closest_to_origin(self, l: tuple[float, float, float]) \
            -> tuple[float, float, float]:
        return -l[0] * l[2], -l[1] * l[2], l[0] * l[0] + l[1] * l[1]

    def transfer_point_to_original_coordinates(self, p: tuple[float, float, float], R: np.matrix, T: np.matrix) \
            -> tuple[float, float, float]:
        x = np.matrix((p[0], p[1], p[2]))
        x = T * R.transpose() * x
        return x[0], x[1], x[2]