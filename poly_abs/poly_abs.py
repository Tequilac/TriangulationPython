import numpy as np

from poly_base.poly_base import PolyBase


class PolyAbs(PolyBase):
    def prepare_poly_coeffs(self, params: tuple[float, float, float, float, float, float]) -> np.ndarray:
        (a, b, c, d, e, f) = params
        result = np.array([
            + 2 * a * b * b * b * c * d + 3 * b * b * d * d * d * d * f * f * f * f + 3 * b * b * b * b * d * d * f * f
            + d * d * d * d * d * d * f * f * f * f * f * f - a * a * b * b * d * d - b * b * b * b * c * c + b * b * b * b * b * b,
            + 12 * b * b * c * d * d * d * f * f * f * f + 12 * a * b * b * b * d * d * f * f + 6 * b * b * b * b * c * d * f * f
            + 6 * a * b * d * d * d * d * f * f * f * f
            + 4 * a * a * b * b * c * d + 6 * c * d * d * d * d * d * f * f * f * f * f * f - 2 * a * a * a * b * d * d
            - 2 * a * b * b * b * c * c + 6 * a * b * b * b * b * b,
            + 24 * a * b * c * d * d * d * f * f * f * f + 24 * a * b * b * b * c * d * f * f
            + 6 * a * b * b * b * c * d * e * e + 18 * b * b * c * c * d * d * f * f * f * f
            + 18 * a * a * b * b * d * d * f * f - 3 * a * a * b * b * d * d * e * e + 2 * a * a * a * b * c * d
            + 15 * c * c * d * d * d * d * f * f * f * f * f * f + 3 * a * a * d * d * d * d * f * f * f * f
            + 3 * b * b * b * b * c * c * f * f - 3 * b * b * b * b * c * c * e * e
            + 15 * a * a * b * b * b * b - a * a * b * b * c * c - a * a * a * a * d * d,
            + 36 * a * b * c * c * d * d * f * f * f * f + 36 * a * a * b * b * c * d * f * f
            + 12 * a * a * b * b * c * d * e * e + 12 * b * b * c * c * c * d * f * f * f * f
            + 12 * a * a * c * d * d * d * f * f * f * f + 12 * a * a * a * b * d * d * f * f
            + 12 * a * b * b * b * c * c * f * f - 6 * a * a * a * b * d * d * e * e - 6 * a * b * b * b * c * c * e * e + 20 * c * c * c * d * d * d * f * f * f * f * f * f + 20 * a * a * a * b * b * b,
            + 24 * a * b * c * c * c * d * f * f * f * f + 24 * a * a * a * b * c * d * f * f
            + 6 * a * b * b * b * c * d * e * e * e * e + 6 * a * a * a * b * c * d * e * e
            + 18 * a * a * c * c * d * d * f * f * f * f + 18 * a * a * b * b * c * c * f * f
            - 3 * a * a * b * b * d * d * e * e * e * e - 3 * a * a * b * b * c * c * e * e
            + 15 * c * c * c * c * d * d * f * f * f * f * f * f + 3 * b * b * c * c * c * c * f * f * f * f
            + 3 * a * a * a * a * d * d * f * f - 3 * b * b * b * b * c * c * e * e * e * e
            - 3 * a * a * a * a * d * d * e * e + 15 * a * a * a * a * b * b,
            + 12 * a * a * b * b * c * d * e * e * e * e + 12 * a * a * c * c * c * d * f * f * f * f
            + 12 * a * a * a * b * c * c * f * f - 6 * a * a * a * b * d * d * e * e * e * e
            - 6 * a * b * b * b * c * c * e * e * e * e + 6 * a * a * a * a * c * d * f * f
            + 6 * a * b * c * c * c * c * f * f * f * f + 6 * c * c * c * c * c * d * f * f * f * f * f * f
            + 6 * a * a * a * a * a * b,
            + 6 * a * a * a * b * c * d * e * e * e * e + 2 * a * b * b * b * c * d * e * e * e * e * e * e
            - 3 * a * a * b * b * c * c * e * e * e * e - 3 * a * a * a * a * d * d * e * e * e * e
            + 3 * a * a * c * c * c * c * f * f * f * f
            + 3 * a * a * a * a * c * c * f * f
            - a * a * b * b * d * d * e * e * e * e * e * e + c * c * c * c * c * c * f * f * f * f * f * f
            - b * b * b * b * c * c * e * e * e * e * e * e + a * a * a * a * a * a,
            + 4 * a * a * b * b * c * d * e * e * e * e * e * e - 2 * a * a * a * b * d * d * e * e * e * e * e * e
            - 2 * a * b * b * b * c * c * e * e * e * e * e * e,
            2 * a * a * a * b * c * d * e * e * e * e * e * e - a * a * b * b * c * c * e * e * e * e * e * e
            - a * a * a * a * d * d * e * e * e * e * e * e])

        result = np.resize(result, (1, self.find_polynominal_order(result)) + 1)

        max_coeff = np.amax(result)

        if max_coeff > 0:
            lam = lambda x: x / max_coeff
            result = lam(result)
        elif max_coeff < 0:
            min_coeff = np.amin(result)
            lam = lambda x: x / min_coeff
            result = lam(result)

        return result

    def evaluate_roots_costs(self, roots: list[any], params: tuple[float, float, float, float, float, float]) \
            -> np.ndarray:
        (a, b, c, d, e, f) = params
        cost_lambda = \
            lambda t: (abs(t) / np.sqrt(1 + e * e * t * t)) + \
                      (abs(c * t + d) / np.sqrt((a * t + b) * (a * t + b) + f * f * (c * t + d) * (c * t + d)))

        return np.array([cost_lambda(root) for root in roots])
