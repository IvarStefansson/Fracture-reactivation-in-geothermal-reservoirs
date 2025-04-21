"""Runable PorePy models of the two-dimensional fracture stimulation
experiments inspired by the work:

Vaezi et al (2024), "Numerical modeling of hydraulic stimulation of
fractured crystalline rock at the bedretto underground laboratory
for geosciences and geoenergies", International Journal of Rock Mechanics
and Mining Sciences, 176, 105689.

The geometry is adapted from the figures provided in the paper. The model
is resolves fractures explicitly and models fracture deformation through
Coulomb friction and classical contact mechanics, in constrast to (visco-)
elastic modeling of embeddings, cf. Vaezi et al (2024).

"""
from .geometry import BedrettoMB1_Geometry
from .physics import BedrettoMB1_Physics
from ncp import (
    ReverseElasticModuli,
    AuxiliaryContact,
    FractureStates,
    IterationExporting,
    LebesgueConvergenceMetrics,
    LogPerformanceDataVectorial,
)
class BedrettoMB1_Model(
    BedrettoMB1_Geometry,  # Geometry
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based contact states for output only
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    ReverseElasticModuli,  # Characteristic displacement from traction
    BedrettoMB1_Physics,  # Basic model, BC and IC
):
    ...