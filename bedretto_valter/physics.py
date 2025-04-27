import numpy as np
import porepy as pp
import egc

# Based on publications on BedrettoLab - a much harder problem

# ! ---- MATERIAL PARAMETERS ----

fluid_parameters: dict[str, float] = {
    "compressibility": 4.6e-10,  # 25 deg C
    "viscosity": 0.89e-3,  # 25 deg C
    "density": 998.2e0,
}

# Values from Multi-disciplinary characterizations of the BedrettoLab
solid_parameters: dict[str, float] = {
    # Guessed
    "dilation_angle": 0.0,  # guessed
    # Literature values
    "biot_coefficient": 1,  # guessed by Vaezi et al.
    "permeability": 4.35e-6 * pp.DARCY,  # X.Ma et al.
    "residual_aperture": 1e-4,  # Computed from transmissivities (X. Ma et al.) and cubic law
    "porosity": 1.36e-2,  # X.Ma et al.
    "shear_modulus": 16.8e9,  # X.Ma et al.
    "lame_lambda": 60.5e9 - 2 / 3 * 16.8e9,  # X.Ma et al.
    "density": 2653,  # X.Ma et al.
    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
    "friction_coefficient": 0.6,  # X.Ma et al.
    "maximum_elastic_fracture_opening": 0e-3,  # Not used
    "fracture_normal_stiffness": 1e3,  # Not used
    "fracture_tangential_stiffness": -1,  # Not used
}

# Testing grounds
overpressure_initialization = [(pp.DAY, 0), (2 * pp.DAY, 0)]
offset = overpressure_initialization[-1][0]
overpressure_schedule = [
    (pp.DAY, 3 * pp.MEGA),
    (2 * pp.DAY, 5 * pp.MEGA),
    (3 * pp.DAY, 10 * pp.MEGA),
    (4 * pp.DAY, 5 * pp.MEGA),
    (5 * pp.DAY, 5 * pp.MEGA),
]
injection_schedule = {
    "time": [t for t, _ in overpressure_initialization]
    + [t + offset for t, _ in overpressure_schedule],
    "overpressure": [p for _, p in overpressure_initialization]
    + [p for _, p in overpressure_schedule],
    "reference_pressure": 1 * pp.MEGA,
}

# Inspired by Vaezi et al.
_overpressure_initialization = [
    (0, 0),
    (1 * pp.DAY, 0),
    (2 * pp.DAY, 0),
    (3 * pp.DAY, 0),
    (4 * pp.DAY, 0),
    (5 * pp.DAY, 0),
]
_offset = _overpressure_initialization[-1][0]
_overpressure_schedule = [
    (0.01 * pp.HOUR, 2 * pp.MEGA),
    (0.2 * pp.HOUR, 2 * pp.MEGA),
    (0.21 * pp.HOUR, 4 * pp.MEGA),
    (0.4 * pp.HOUR, 4 * pp.MEGA),
    (0.41 * pp.HOUR, 6 * pp.MEGA),
    (0.6 * pp.HOUR, 6 * pp.MEGA),
    (0.61 * pp.HOUR, 8 * pp.MEGA),
    (0.8 * pp.HOUR, 8 * pp.MEGA),
    (0.81 * pp.HOUR, 10 * pp.MEGA),
    (1.0 * pp.HOUR, 10 * pp.MEGA),
    (1.01 * pp.HOUR, 12 * pp.MEGA),
    (1.2 * pp.HOUR, 12 * pp.MEGA),
    (1.21 * pp.HOUR, 14 * pp.MEGA),
    (1.4 * pp.HOUR, 14 * pp.MEGA),
    (1.41 * pp.HOUR, 12 * pp.MEGA),
    (1.6 * pp.HOUR, 12 * pp.MEGA),
    (1.61 * pp.HOUR, 14 * pp.MEGA),
    (1.8 * pp.HOUR, 14 * pp.MEGA),
    (1.81 * pp.HOUR, 16 * pp.MEGA),
    (2.0 * pp.HOUR, 16 * pp.MEGA),
    (2.01 * pp.HOUR, 20 * pp.MEGA),
    (2.2 * pp.HOUR, 20 * pp.MEGA),
    (2.21 * pp.HOUR, 16 * pp.MEGA),
    (2.4 * pp.HOUR, 8 * pp.MEGA),
    (2.41 * pp.HOUR, 20 * pp.MEGA),
    (2.6 * pp.HOUR, 20 * pp.MEGA),
    (2.61 * pp.HOUR, 16 * pp.MEGA),
    (2.8 * pp.HOUR, 10 * pp.MEGA),
    (2.81 * pp.HOUR, 20 * pp.MEGA),
    (3.0 * pp.HOUR, 20 * pp.MEGA),
    (3.01 * pp.HOUR, 16 * pp.MEGA),
    (3.2 * pp.HOUR, 11 * pp.MEGA),
    (3.21 * pp.HOUR, 20 * pp.MEGA),
    (3.4 * pp.HOUR, 20 * pp.MEGA),
    (3.41 * pp.HOUR, 16 * pp.MEGA),
    (3.6 * pp.HOUR, 12 * pp.MEGA),
    (3.61 * pp.HOUR, 20 * pp.MEGA),
    (3.8 * pp.HOUR, 20 * pp.MEGA),
    (3.81 * pp.HOUR, 16 * pp.MEGA),
    (4.0 * pp.HOUR, 12 * pp.MEGA),
    (4.01 * pp.HOUR, 20 * pp.MEGA),
    (4.2 * pp.HOUR, 20 * pp.MEGA),
    (4.21 * pp.HOUR, 16 * pp.MEGA),
    (4.4 * pp.HOUR, 12 * pp.MEGA),
    (4.41 * pp.HOUR, 20 * pp.MEGA),
    (4.6 * pp.HOUR, 20 * pp.MEGA),
    (4.61 * pp.HOUR, 16 * pp.MEGA),
    (4.8 * pp.HOUR, 12 * pp.MEGA),
    (4.81 * pp.HOUR, 20 * pp.MEGA),
    (5.0 * pp.HOUR, 20 * pp.MEGA),
    (5.01 * pp.HOUR, 16 * pp.MEGA),
    (5.2 * pp.HOUR, 12 * pp.MEGA),
    (5.21 * pp.HOUR, 20 * pp.MEGA),
    (5.4 * pp.HOUR, 20 * pp.MEGA),
    (5.41 * pp.HOUR, 16 * pp.MEGA),
    (5.6 * pp.HOUR, 12 * pp.MEGA),
    (5.61 * pp.HOUR, 20 * pp.MEGA),
    (5.8 * pp.HOUR, 20 * pp.MEGA),
    (5.81 * pp.HOUR, 16 * pp.MEGA),
    (6.0 * pp.HOUR, 12 * pp.MEGA),
    (6.01 * pp.HOUR, 20 * pp.MEGA),
    (6.2 * pp.HOUR, 20 * pp.MEGA),
    (6.21 * pp.HOUR, 16 * pp.MEGA),
    (6.4 * pp.HOUR, 12 * pp.MEGA),
    (6.41 * pp.HOUR, 20 * pp.MEGA),
    (6.6 * pp.HOUR, 20 * pp.MEGA),
    (6.61 * pp.HOUR, 16 * pp.MEGA),
    (6.8 * pp.HOUR, 12 * pp.MEGA),
    (6.81 * pp.HOUR, 20 * pp.MEGA),
    (7.0 * pp.HOUR, 20 * pp.MEGA),
    (7.01 * pp.HOUR, 16 * pp.MEGA),
    (7.2 * pp.HOUR, 12 * pp.MEGA),
    (7.21 * pp.HOUR, 20 * pp.MEGA),
    (7.4 * pp.HOUR, 20 * pp.MEGA),
    (7.41 * pp.HOUR, 16 * pp.MEGA),
    (7.6 * pp.HOUR, 12 * pp.MEGA),
    (7.61 * pp.HOUR, 20 * pp.MEGA),
    (7.8 * pp.HOUR, 20 * pp.MEGA),
    (7.81 * pp.HOUR, 16 * pp.MEGA),
    (8.0 * pp.HOUR, 12 * pp.MEGA),
]
_injection_schedule = {
    "time": [t for t, _ in _overpressure_initialization]
    + [t + _offset for t, _ in _overpressure_schedule],
    "overpressure": [p for _, p in _overpressure_initialization]
    + [p for _, p in _overpressure_schedule],
    "reference_pressure": 10 * pp.MEGA,
}
# import matplotlib.pyplot as plt
# plt.figure("pressure schedule")
# plt.plot((np.array(_injection_schedule["time"][6:]) -_offset) / 3600, np.array(_injection_schedule["overpressure"][6:]) / 1e6)
# plt.xlabel("time [hr]")
# plt.ylabel("over pressure [MPa]")
# plt.title("Injection schedule")
# plt.grid()
# plt.show()

numerics_parameters: dict[str, float] = {
    "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_contact_traction": injection_schedule["reference_pressure"],
}


class HorizontalBackgroundStress(egc.BackgroundStress):
    def horizontal_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Horizontal background stress

        Values are based on the following paper:
        Hetrich et al. (2021) "Characterization, hydraulic stimulation, and fluid
        circulation experiments in the Bedretto Underground Laboratory for
        Geosciences and Geoenergies", ARMA 21-1895

        Assume homogeneous stress field for fixed depth. With increasing depth
        both the lithostatic and horizontal stress increase with the same factor.

        """
        principal_background_stress_max_factor = (
            19.8 / 26.5
        )  # 19.8 MPa vs 26.5 MPa at top
        principal_background_stress_min_factor = (
            11.2 / 26.5
        )  # 11.2 MPa vs 26.5 MPa at top
        background_stress_deg = 112 * (np.pi / 180)  # N112 degrees East

        s_v = self.vertical_background_stress(grid)
        s_h = np.zeros((self.nd - 1, self.nd - 1, grid.num_cells))
        principal_stress_factor = np.array(
            [
                [principal_background_stress_max_factor, 0],
                [0, principal_background_stress_min_factor],
            ]
        )
        rotation = np.array(
            [
                [np.cos(background_stress_deg), -np.sin(background_stress_deg)],
                [np.sin(background_stress_deg), np.cos(background_stress_deg)],
            ]
        )
        scaling = rotation @ principal_stress_factor @ rotation.T
        for i, j in np.ndindex(self.nd - 1, self.nd - 1):
            s_h[i, j] = scaling[i, j] * s_v
        return s_h


class PressureConstraintWell:
    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        current_injection_overpressure = np.interp(
            self.time_manager.time,
            injection_schedule["time"],
            injection_schedule["overpressure"],
            left=0.0,
        )
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_overpressure",
                values=np.array(
                    [self.units.convert_units(current_injection_overpressure, "Pa")]
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

        if len(fracture_sds) == 0:
            return std_eq

        # Pick a single fracture
        well_sd = fracture_sds[7]

        for i, sd in enumerate(subdomains):
            if sd == well_sd:
                # Pick the center (hardcoded)
                well_loc = np.array(
                    [
                        self.units.convert_units(0, "m"),
                        self.units.convert_units(0, "m"),
                        self.units.convert_units(-150, "m"),
                    ]
                ).reshape((3, 1))

                well_loc_ind = sd.closest_cell(well_loc)

                sd_indicator[i][well_loc_ind] = 1

        # Characteristic functions
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1 - indicator

        current_injection_overpressure = pp.ad.TimeDependentDenseArray(
            "current_injection_overpressure", [self.mdg.subdomains()[0]]
        )
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = (
            self.pressure(subdomains)
            - current_injection_overpressure
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


class BedrettoValter_Physics(
    # egc.HydrostaticPressureInitialCondition,
    HorizontalBackgroundStress,
    egc.HydrostaticPressureBC,
    egc.LithostaticPressureBC,
    egc.HydrostaticPressureInitialization,
    PressureConstraintWell,
    pp.constitutive_laws.GravityForce,
    egc.ScalarPermeability,
    egc.NormalPermeabilityFromHigherDimension,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    # egc.LinearFluidCompressibility, # Replaces exponential law
    pp.poromechanics.Poromechanics,  # Basic model
): ...
