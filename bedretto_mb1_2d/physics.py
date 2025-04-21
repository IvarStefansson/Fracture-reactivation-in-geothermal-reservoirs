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

import numpy as np
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
from copy import deepcopy

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
    "time": [i * pp.DAY for i in range(6)],
    "pressure": [0, 3 * pp.MEGA, 5 * pp.MEGA, 10 * pp.MEGA, 5 * pp.MEGA, 5 * pp.MEGA],
    "reference_pressure": 1 * pp.MEGA,
}

numerics_parameters: dict[str, float] = {
    "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_contact_traction": injection_schedule["reference_pressure"],
    "contact_mechanics_scaling": 1.0,
}


class PressureConstraintWell:
    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        current_injection_pressure = np.interp(
            self.time_manager.time,
            injection_schedule["time"],
            injection_schedule["pressure"],
            left=0.0,
        )
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_pressure",
                values=np.array(
                    [self.units.convert_units(current_injection_pressure, "Pa")]
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        std_eq = super().mass_balance_equation(subdomains)

        # Need to embedd in full domain
        sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]

        # Pick the only subdomain
        fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]

        # Leave equation unmodified if no fractures present
        if len(fracture_sds) == 0:
            return std_eq

        # Pick a single fracture (the stimulated one is fracture #14 - first segment)
        well_sd = fracture_sds[self.stimulated_fracture_id]

        for i, sd in enumerate(subdomains):
            if sd == well_sd:
                # Pick the center (hardcoded)
                depth = self._domain.bounding_box["ymin"]
                well_loc = np.array(
                    [
                        self.units.convert_units(72, "m"),
                        self.units.convert_units(depth - 270, "m"),
                    ]
                ).reshape((2, 1))

                well_loc_ind = sd.closest_cell(well_loc)

                sd_indicator[i][well_loc_ind] = 1

        # Characteristic functions
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1 - indicator

        current_injection_pressure = pp.ad.TimeDependentDenseArray(
            "current_injection_pressure", [self.mdg.subdomains()[0]]
        )
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = (
            self.pressure(subdomains)
            - current_injection_pressure
            - hydrostatic_pressure
        )

        eq_with_pressure_constraint = (
            pp.ad.DenseArray(reverse_indicator) * std_eq
            + pp.ad.DenseArray(indicator) * constrained_eq
        )
        eq_with_pressure_constraint.set_name(
            "mass_balance_equation_with_constrained_pressure"
        )

        return eq_with_pressure_constraint


class BedrettoMB1_Physics(
    egc.InitialConditionFromParameters,
    egc.BackgroundStress,
    egc.HydrostaticPressureBC,
    egc.LithostaticPressureBC,
    egc.HydrostaticPressureInitialization,
    PressureConstraintWell,
    pp.constitutive_laws.GravityForce,
    egc.ScalarPermeability,
    egc.NormalPermeabilityFromHigherDimension,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
): ...


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

# Add simplified model and parameters for initialization
fluid_parameters_initialization = deepcopy(fluid_parameters)
solid_parameters_initialization = deepcopy(solid_parameters)
numerics_parameters_initialization = deepcopy(numerics_parameters)
solid_parameters_initialization["dilation_angle"] = 0.0

injection_schedule_initialization = {
    "time": [0 * pp.DAY, 2 * pp.DAY],
    "pressure": [0, 0],
    "reference_pressure": 1 * pp.MEGA,
}


class BedrettoMB1_Model_Initialization(
    egc.HydrostaticPressureInitialCondition,
    BedrettoMB1_Model,
): ...