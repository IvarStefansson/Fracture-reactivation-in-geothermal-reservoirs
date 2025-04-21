"""Aim to replicate Vaezi et al. (2024)."""

import porepy as pp
import egc

from .geometry import BedrettoMB1_Geometry
from ncp import (
    ReverseElasticModuli,
    AuxiliaryContact,
    FractureStates,
    IterationExporting,
    LebesgueConvergenceMetrics,
    LogPerformanceDataVectorial,
)
# ! ---- MATERIAL PARAMETERS ----

fluid_parameters: dict[str, float] = {
    "compressibility": 4.6e-10,  # 25 deg C
    "viscosity": 0.89e-3,  # 25 deg C
    "density": 998.2e0,
}

# Values from Multi-disciplinary characterizations of the BedrettoLab
solid_parameters: dict[str, float] = {
    # Guessed
    "dilation_angle": 0.0,  # guessed - TODO increase with time values (20 deg Vaezi et al?)
    "biot_coefficient": 1,  # Vaezi et al.
    "permeability": 2.5e-18,  # Vaezi et al.
    "residual_aperture": 2.25e-6,  # Vaezi et al.
    "porosity": 0.005,  # Vaezi et al.
    "shear_modulus": 16.8 * pp.GIGA,  # Vaezi et al
    "lame_lambda": 47.78 * pp.GIGA,  # X.Ma et al.
    "density": 2653,  # X.Ma et al.
    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
    "friction_coefficient": 0.6,  # X.Ma et al.
    "maximum_elastic_fracture_opening": 0e-3,  # Not used
    "fracture_normal_stiffness": 1e3,  # Not used
    "fracture_tangential_stiffness": -1,  # Not used
}

injection_schedule = {
    "time": [pp.DAY, 2 * pp.DAY],
    "pressure": [0, 0],
    "reference_pressure": 1 * pp.MEGA,
}

numerics_parameters: dict[str, float] = {
    "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_contact_traction": injection_schedule["reference_pressure"],
    "contact_mechanics_scaling": 1.0,
}


class BedrettoMB1_Initialization_Physics(
    egc.HydrostaticPressureInitialCondition,
    egc.BackgroundStress,
    egc.HydrostaticPressureBC,
    egc.LithostaticPressureBC,
    egc.HydrostaticPressureInitialization,
    pp.constitutive_laws.GravityForce,
    egc.ScalarPermeability,
    egc.NormalPermeabilityFromHigherDimension,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
): ...


class BedrettoMB1_Initialization_Model(
    BedrettoMB1_Geometry,  # Geometry
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based contact states for output only
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    ReverseElasticModuli,  # Characteristic displacement from traction
    BedrettoMB1_Initialization_Physics,  # Basic model, BC and IC
): ...
