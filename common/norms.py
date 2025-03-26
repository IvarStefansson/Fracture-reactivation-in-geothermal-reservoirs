import logging

# Module-wide logger
logger = logging.getLogger(__name__)

import numpy as np

from .metric import EuclideanMetric, LebesgueMetric


class EuclideanConvergenceMetrics:
    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray, split: bool = False
    ) -> float:
        return EuclideanMetric.norm(self, nonlinear_increment, split)

    def compute_residual_norm(self, residual: np.ndarray, split: bool = False) -> float:
        return EuclideanMetric.norm(self, residual, split)


class LebesgueConvergenceMetrics:
    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray, split: bool = False
    ) -> float:
        return LebesgueMetric.variable_norm(
            self, values=nonlinear_increment, variables=None, split=split
        )

    def compute_residual_norm(self, residual: np.ndarray, split: bool = False) -> float:
        return LebesgueMetric.residual_norm(self, equations=None, split=split)
