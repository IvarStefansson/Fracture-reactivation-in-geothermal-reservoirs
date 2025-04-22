"""Predefined model for the Bedretto tunnel geometry."""

from .geometry import BedrettoValter_Geometry
from .physics import BedrettoValter_Physics
from ncp import (
    ReverseElasticModuli,
    AuxiliaryContact,
    FractureStates,
    IterationExporting,
    LebesgueConvergenceMetrics,
    LogPerformanceDataVectorial,
)


class BedrettoValterModel(
    BedrettoValter_Geometry,  # Geometry
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based contact states for output only
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    ReverseElasticModuli,  # Characteristic displacement from traction
    BedrettoValter_Physics,  # Basic model, BC and IC
):
    """Bedretto model"""
