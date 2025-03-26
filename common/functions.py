"""NCP and AD functions."""

import porepy as pp
from functools import partial
import numpy as np
from porepy.numerics.ad.forward_mode import AdArray
from typing import TypeVar

FloatType = TypeVar("FloatType", AdArray, np.ndarray, float)

# Utils for NCP functions


def ncp_min(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Min function."""
    assert np.isclose(mu, 0.0), "mu not implemented"
    # regularized min-NCP: min(a,b) = 0.5 * (a+b - ((a-b)^2)^0.5
    # equation: pp.ad.Operator = pp.ad.Scalar(0.5) * (
    #    force + gap - ((force - gap) ** 2 + mu) ** 0.5
    # )
    f_max = pp.ad.Function(pp.ad.maximum, "max_function")
    return pp.ad.Scalar(-1.0) * f_max(pp.ad.Scalar(-1.0) * a, pp.ad.Scalar(-1.0) * b)


def ncp_fb(a: pp.ad.Operator, b: pp.ad.Operator, mu: float = 0.0) -> pp.ad.Operator:
    """Fischer-Burmeister function."""
    assert np.isclose(mu, 0.0), "mu not implemented"
    # Fischer-Burmeister: (a**2 + b**2)**0.5 - (a + b)
    # equation: pp.ad.Operator = (force + gap) - (force**2 + gap**2 + mu) ** 0.5
    return pp.ad.Scalar(0.5) * ((a + b) - (a**2 + b**2) ** 0.5)


def ncp_min_regularized_fb(
    a: pp.ad.Operator, b: pp.ad.Operator, tol: float = 1e-10
) -> pp.ad.Operator:
    """Fischer-Burmeister function regularized by min function."""
    f_characteristic_fb = pp.ad.Function(
        partial(pp.ad.functions.characteristic_function, tol),
        "characteristic_function_for_zero_normal_traction",
    )
    char_val = f_characteristic_fb(a**2 + b**2)
    min_ncp_equation: pp.ad.Operator = ncp_min(a, b)
    fb_ncp_equation = ncp_fb(a, b)
    return (
        char_val * min_ncp_equation + (pp.ad.Scalar(1.0) - char_val) * fb_ncp_equation
    )


# Extensions of porepy/numerics/ad/functions.py


def nan_to_num(var: FloatType) -> FloatType:
    if isinstance(var, AdArray):
        val = np.nan_to_num(var.val)
        jac = var._diagvec_mul_jac(np.zeros_like(var.val))
        return AdArray(val, 0 * jac)
    else:
        return np.nan_to_num(var)


def sign(var: FloatType) -> FloatType:
    tol = -1e-12
    if isinstance(var, AdArray):
        val = np.ones_like(var.val, dtype=var.val.dtype)
        neg_inds = var.val < tol
        val[neg_inds] = -1
        jac = var._diagvec_mul_jac(np.sign(var.val))
        return AdArray(val, 0 * jac)
    else:
        return np.sign(var)
