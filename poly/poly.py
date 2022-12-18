import numpy as np

from poly_base.poly_base import PolyBase


class Poly(PolyBase):
    def prepare_poly_coeffs(self, params: tuple[float, float, float, float, float, float]) -> np.ndarray:
        (a, b, c, d, e, f) = params
        result = np.array([
            a * b * d * d - b * b * c * d,
            -2 * b * b * d * d * f * f + a * a * d * d - d * d * d * d * f * f * f * f - b * b * c * c - b * b * b * b,
            -4 * b * b * c * d * f * f - 2 * b * b * c * d * e * e - 4 * a * b * d * d * f * f
            + 2 * a * b * d * d * e * e + a * a * c * d - 4 * c * d * d * d * f * f * f * f - a * b * c * c - 4 * a * b * b * b,
            -8 * a * b * c * d * f * f - 6 * c * c * d * d * f * f * f * f - 2 * b * b * c * c * f * f
            - 2 * a * a * d * d * f * f - 2 * b * b * c * c * e * e + 2 * a * a * d * d * e * e - 6 * a * a * b * b,
            -4 * a * a * c * d * f * f + 2 * a * a * c * d * e * e
            - 4 * a * b * c * c * f * f + a * b * d * d * e * e * e * e - 2 * a * b * c * c * e * e
            - b * b * c * d * e * e * e * e - 4 * c * c * c * d * f * f * f * f - 4 * a * a * a * b,
            a * a * d * d * e * e * e * e - 2 * a * a * c * c * f * f - b * b * c * c * e * e * e * e
            - c * c * c * c * f * f * f * f - a * a * a * a,
            a * a * c * d * e * e * e * e - a * b * c * c * e * e * e * e])

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
            lambda t: ((t * t) / (1 + e * e * t * t)) + ((c * t + d) * (c * t + d)) / (
                    (a * t + b) * (a * t + b) + f * f * (c * t + d) * (c * t + d))

        return np.array([cost_lambda(root) for root in roots])