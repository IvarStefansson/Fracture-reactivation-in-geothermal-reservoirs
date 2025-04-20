import numpy as np
import porepy as pp
import egc

if False:
    # Used so far but based on previous works by PMG

    # ! ---- MATERIAL PARAMETERS ----

    fluid_parameters: dict[str, float] = {
        "compressibility": 0,
        "viscosity": 1e-3,
        "density": 998.2e0,
    }

    solid_parameters: dict[str, float] = {
        "biot_coefficient": 1.0,
        "permeability": 1e-14,
        "porosity": 1.0e-2,
        "shear_modulus": 1e14,
        "lame_lambda": 1e14,
        "residual_aperture": 1e-3,
        "density": 2600,
        "maximum_elastic_fracture_opening": 0e-3,  # Not used
        "fracture_normal_stiffness": 1e3,  # Not used
        "fracture_tangential_stiffness": -1,
        "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
        "dilation_angle": 0.1,
        "friction_coefficient": 0.8,
    }

    injection_schedule = {
        "time": [pp.DAY, 2 * pp.DAY] + [(3 + i) * pp.DAY for i in range(5)],
        "pressure": [0, 0] + [3e7, 5e7, 10e7, 5e7, 5e7],
        "reference_pressure": 1e7,
    }

    numerics_parameters: dict[str, float] = {
        "open_state_tolerance": 1e-10,  # Numerical method parameter
        "characteristic_contact_traction": injection_schedule["reference_pressure"],
        "contact_mechanics_scaling": 1.0,
    }

else:
    # Based on publications on BedrettoLab - a much harder problem

    # ! ---- MATERIAL PARAMETERS ----

    fluid_parameters: dict[str, float] = {
        "compressibility": 0,  # 4.6e-10, # 25 deg C
        "viscosity": 0.89e-3,  # 25 deg C
        "density": 998.2e0,
    }

    # Values from Multi-disciplinary characterizations of the BedrettoLab
    solid_parameters: dict[str, float] = {
        # Guessed
        "biot_coefficient": 1,  # guessed by Vaezi et al.
        "normal_permeability": 4.35e-6 * pp.DARCY,  # guessed
        "dilation_angle": 0.1,  # guessed
        # Literature values
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

    injection_schedule = {
        "time": [pp.DAY, 2 * pp.DAY] + [(3 + i) * pp.DAY for i in range(5)],
        "pressure": [0, 0] + [3e7, 5e7, 10e7, 5e7, 5e7],
        "reference_pressure": 1e7,
    }

    numerics_parameters: dict[str, float] = {
        "open_state_tolerance": 1e-10,  # Numerical method parameter
        "characteristic_contact_traction": injection_schedule["reference_pressure"],
        "contact_mechanics_scaling": 1.0,
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

        if len(fracture_sds) == 0:
            return std_eq

        # Pick a single fracture
        well_sd = fracture_sds[0]

        for i, sd in enumerate(subdomains):
            if sd == well_sd:
                # Pick the center (hardcoded)
                well_loc = np.array(
                    [
                        self.units.convert_units(0, "m"),
                        self.units.convert_units(0, "m"),
                        self.units.convert_units(-3000, "m"),
                    ]
                ).reshape((3, 1))

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


class Physics(
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
    pp.poromechanics.Poromechanics,  # Basic model
): ...
