import numpy as np
import porepy as pp
import egc

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
    egc.HydrostaticPressureInitialCondition,
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
