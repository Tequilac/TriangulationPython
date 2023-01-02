import numpy as np

from iterative_eigen.iterative_eigen import IterativeEigen
from iterative_ls.iterative_ls import IterativeLS
from linear_eigen.linear_eigen import LinearEigen
from linear_ls.linear_ls import LinearLS
from poly.poly import Poly
from poly_abs.poly_abs import PolyAbs


def calculate(matrix1, matrix2, point1, point2, Method):
    method = Method(matrix1, matrix2)
    res = method.triangulate_point(point1, point2)
    print(f"Result: {res} Method: {Method.__name__}")
    print()


def compute_intrinsic_parameter_matrix(f, s, a, cx, cy):
    K = np.identity(3)
    K[0, 0] = f
    K[0, 1] = s
    K[0, 2] = cx
    K[1, 1] = a * f
    K[1, 2] = cy

    return K


def Rx(omega):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(omega), -np.sin(omega)],
                     [0.0, np.sin(omega), np.cos(omega)]])


def Ry(phi):
    return np.array([[np.cos(phi), 0.0, np.sin(phi)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(phi), 0.0, np.cos(phi)]])


def Rz(kappa):
    return np.array([[np.cos(kappa), -np.sin(kappa), 0.0],
                     [np.sin(kappa), np.cos(kappa), 0.0],
                     [0.0, 0.0, 1.0]])


def camera(f, kappa, cx, cy, s, tx, ty, tz, rx, ry, rz):
    K = compute_intrinsic_parameter_matrix(f, s, kappa, cx, cy)
    # K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    print(K)
    t = np.array([[tx], [ty], [tz]], dtype=np.float64)
    R = Rx(rx) @ Ry(ry) @ Rz(rz)
    Rt = np.hstack((R, t))
    return K @ Rt


p0 = camera(f=5.467110e+002,
            kappa=3.053090e-007,
            cx=4.800000e+002,
            cy=2.700000e+002,
            s=1.000730e+000,
            tx=2.119430e+001,
            ty=6.138730e+002,
            tz=3.393530e+003,
            rx=3.131889e+000,
            ry=1.352972e+000,
            rz=1.565215e+000)

p1 = camera(f=5.532470e+002,
            kappa=6.184690e-007,
            cx=4.800000e+002,
            cy=2.700000e+002,
            s=9.962450e-001,
            tx=-6.209040e+001,
            ty=6.183890e+002,
            tz=5.207650e+003,
            rx=-1.702316e+000,
            ry=-1.144498e-002,
            rz=-3.137334e+000)

p2 = camera(f=5.362660e+002,
            kappa=3.438760e-007,
            cx=4.800000e+002,
            cy=2.700000e+002,
            s=1.002330e+000,
            tx=7.118630e+000,
            ty=5.031500e+002,
            tz=3.684760e+003,
            rx=-3.067590e+000,
            ry=-1.345954e+000,
            rz=-1.629773e+000)

p3 = camera(f=5.523020e+002,
            kappa=5.218370e-007,
            cx=4.800000e+002,
            cy=2.700000e+002,
            s=9.962750e-001,
            tx=1.513780e+002,
            ty=4.692570e+002,
            tz=5.216930e+003,
            rx=1.741505e+000,
            ry=-3.491322e-003,
            rz=-8.690605e-003)

print(p0)
print(p1)
print(p2)
print(p3)

point0 = (481, 368)

point1 = (484, 338)

point2 = (483, 356)

point3 = (482, 324)

# point0 = (752, 110)
# point1 = (483, 152)
# point2 = (222, 104)
# point3 = (484, 156)

calculate(p0, p1, point0, point1, Poly)
calculate(p0, p2, point0, point2, Poly)
calculate(p0, p3, point0, point3, Poly)
print()

calculate(p0, p1, point0, point1, PolyAbs)
calculate(p0, p2, point0, point2, PolyAbs)
calculate(p0, p3, point0, point3, PolyAbs)
print()

calculate(p0, p1, point0, point1, LinearLS)
calculate(p0, p2, point0, point2, LinearLS)
calculate(p0, p3, point0, point3, LinearLS)
print()

calculate(p0, p1, point0, point1, LinearEigen)
calculate(p0, p2, point0, point2, LinearEigen)
calculate(p0, p3, point0, point3, LinearEigen)
print()

calculate(p0, p1, point0, point1, IterativeLS)
calculate(p0, p2, point0, point2, IterativeLS)
calculate(p0, p3, point0, point3, IterativeLS)
print()

calculate(p0, p1, point0, point1, IterativeEigen)
calculate(p0, p2, point0, point2, IterativeEigen)
calculate(p0, p3, point0, point3, IterativeEigen)
print()

calculate(p0, p0, point0, point0, IterativeEigen)
