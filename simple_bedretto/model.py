"""Predefined model for the Bedretto tunnel geometry."""

from simple_bedretto.geometry import SimpleBedrettoTunnel_Geometry
from simple_bedretto.physics import SimpleBedrettoTunnel_Physics
from ncp import (
    ReverseElasticModuli,
    AuxiliaryContact,
    FractureStates,
    IterationExporting,
    LebesgueConvergenceMetrics,
    LogPerformanceDataVectorial,
)


class SimpleBedrettoTunnel_Model(
    SimpleBedrettoTunnel_Geometry,  # Geometry
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based contact states for output only
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    ReverseElasticModuli,  # Characteristic displacement from traction
    SimpleBedrettoTunnel_Physics,  # Basic model, BC and IC
):
    """Simple Bedretto model solved with Huebers nonlinear radial return formulation."""
