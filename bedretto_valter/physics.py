import numpy as np
import porepy as pp
import egc
from icecream import ic

# Based on publications on BedrettoLab

# ! ---- MATERIAL PARAMETERS ----

fluid_parameters: dict[str, float] = {
    "compressibility": 0,  # 4.6e-10,  # 25 deg C # TODO increase
    "viscosity": 0.89e-3,  # 25 deg C
    "density": 998.2e0,
}

# Values from Multi-disciplinary characterizations of the BedrettoLab
solid_parameters: dict[str, float] = {
    # Guessed
    "dilation_angle": 0.0,  # guessed # TODO
    # Literature values
    "biot_coefficient": 1,  # guessed by Vaezi et al.
    "permeability": 4.35e-6 * pp.DARCY,  # X.Ma et al.
    "normal_permeability": 4.35e-6 * pp.DARCY,  # X.Ma et al. # TODO - not used at the moment
    "residual_aperture": 1e-4,  # Computed from transmissivities (X. Ma et al.) and cubic law
    "porosity": 1.36e-2,  # X.Ma et al.
    "shear_modulus": 16.8 * pp.GIGA,  # X.Ma et al.
    "lame_lambda": 19.73 * pp.GIGA,  # X.Ma et al.
    "density": 2653,  # X.Ma et al.
    "friction_coefficient": 0.6,  # X.Ma et al.
}
# fracture_permeability = 1e-8 / 12
# intersection_permeability = 1e-8 / 12

# Testing grounds
if False:
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
        "reference_pressure": 3 * pp.MEGA,
    }
else:
    # Inspired by Vaezi et al.
    overpressure_initialization = [(i * pp.DAY, 0) for i in [0.25, 0.5, 0.75, 1]]
    offset = overpressure_initialization[-1][0]
    overpressure_schedule = [
        (0.05 * pp.HOUR, 2 * pp.MEGA),
        (0.2 * pp.HOUR, 2 * pp.MEGA),
        (0.25 * pp.HOUR, 4 * pp.MEGA),
        (0.4 * pp.HOUR, 4 * pp.MEGA),
        (0.45 * pp.HOUR, 6 * pp.MEGA),
        (0.6 * pp.HOUR, 6 * pp.MEGA),
        (0.65 * pp.HOUR, 8 * pp.MEGA),
        (0.8 * pp.HOUR, 8 * pp.MEGA),
        (0.85 * pp.HOUR, 10 * pp.MEGA),
        (1.0 * pp.HOUR, 10 * pp.MEGA),
        (1.05 * pp.HOUR, 12 * pp.MEGA),
        (1.2 * pp.HOUR, 12 * pp.MEGA),
        (1.25 * pp.HOUR, 14 * pp.MEGA),
        (1.4 * pp.HOUR, 14 * pp.MEGA),
        (1.45 * pp.HOUR, 12 * pp.MEGA),
        (1.6 * pp.HOUR, 12 * pp.MEGA),
        (1.65 * pp.HOUR, 14 * pp.MEGA),
        (1.8 * pp.HOUR, 14 * pp.MEGA),
        (1.85 * pp.HOUR, 16 * pp.MEGA),
        (2.0 * pp.HOUR, 16 * pp.MEGA),
        (2.05 * pp.HOUR, 20 * pp.MEGA),
        (2.2 * pp.HOUR, 20 * pp.MEGA),
        (2.25 * pp.HOUR, 16 * pp.MEGA),
        (2.4 * pp.HOUR, 8 * pp.MEGA),
        (2.45 * pp.HOUR, 20 * pp.MEGA),
        (2.6 * pp.HOUR, 20 * pp.MEGA),
        (2.65 * pp.HOUR, 16 * pp.MEGA),
        (2.8 * pp.HOUR, 10 * pp.MEGA),
        (2.85 * pp.HOUR, 20 * pp.MEGA),
        (3.0 * pp.HOUR, 20 * pp.MEGA),
        (3.05 * pp.HOUR, 16 * pp.MEGA),
        (3.2 * pp.HOUR, 11 * pp.MEGA),
        (3.25 * pp.HOUR, 20 * pp.MEGA),
        (3.4 * pp.HOUR, 20 * pp.MEGA),
        (3.45 * pp.HOUR, 16 * pp.MEGA),
        (3.6 * pp.HOUR, 12 * pp.MEGA),
        (3.65 * pp.HOUR, 20 * pp.MEGA),
        (3.8 * pp.HOUR, 20 * pp.MEGA),
        (3.85 * pp.HOUR, 16 * pp.MEGA),
        (4.0 * pp.HOUR, 12 * pp.MEGA),
        (4.05 * pp.HOUR, 20 * pp.MEGA),
        (4.2 * pp.HOUR, 20 * pp.MEGA),
        (4.25 * pp.HOUR, 16 * pp.MEGA),
        (4.4 * pp.HOUR, 12 * pp.MEGA),
        (4.45 * pp.HOUR, 20 * pp.MEGA),
        (4.6 * pp.HOUR, 20 * pp.MEGA),
        (4.65 * pp.HOUR, 16 * pp.MEGA),
        (4.8 * pp.HOUR, 12 * pp.MEGA),
        (4.85 * pp.HOUR, 20 * pp.MEGA),
        (5.0 * pp.HOUR, 20 * pp.MEGA),
        (5.05 * pp.HOUR, 16 * pp.MEGA),
        (5.2 * pp.HOUR, 12 * pp.MEGA),
        (5.25 * pp.HOUR, 20 * pp.MEGA),
        (5.4 * pp.HOUR, 20 * pp.MEGA),
        (5.45 * pp.HOUR, 16 * pp.MEGA),
        (5.6 * pp.HOUR, 12 * pp.MEGA),
        (5.65 * pp.HOUR, 20 * pp.MEGA),
        (5.8 * pp.HOUR, 20 * pp.MEGA),
        (5.85 * pp.HOUR, 16 * pp.MEGA),
        (6.0 * pp.HOUR, 12 * pp.MEGA),
        (6.05 * pp.HOUR, 20 * pp.MEGA),
        (6.2 * pp.HOUR, 20 * pp.MEGA),
        (6.25 * pp.HOUR, 16 * pp.MEGA),
        (6.4 * pp.HOUR, 12 * pp.MEGA),
        (6.45 * pp.HOUR, 20 * pp.MEGA),
        (6.6 * pp.HOUR, 20 * pp.MEGA),
        (6.65 * pp.HOUR, 16 * pp.MEGA),
        (6.8 * pp.HOUR, 12 * pp.MEGA),
        (6.85 * pp.HOUR, 20 * pp.MEGA),
        (7.0 * pp.HOUR, 20 * pp.MEGA),
        (7.05 * pp.HOUR, 16 * pp.MEGA),
        (7.2 * pp.HOUR, 12 * pp.MEGA),
        (7.25 * pp.HOUR, 20 * pp.MEGA),
        (7.4 * pp.HOUR, 20 * pp.MEGA),
        (7.45 * pp.HOUR, 16 * pp.MEGA),
        (7.6 * pp.HOUR, 12 * pp.MEGA),
        (7.65 * pp.HOUR, 20 * pp.MEGA),
        (7.8 * pp.HOUR, 20 * pp.MEGA),
        (7.85 * pp.HOUR, 16 * pp.MEGA),
        (8.0 * pp.HOUR, 12 * pp.MEGA),
    ]
    injection_schedule = {
        "time": [t for t, _ in overpressure_initialization]
        + [t + offset for t, _ in overpressure_schedule],
        "overpressure": [p for _, p in overpressure_initialization]
        + [p for _, p in overpressure_schedule],
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


class PorePressure:
    """Generalize the hydrostatic pressure, X. Ma et al."""

    def hydrostatic_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_m1000 = self.units.convert_units(2.0 * pp.MEGA, "Pa")  # minus 1000 m
        z = sd.cell_centers[self.nd - 1]
        slope = self.units.convert_units((5.6 - 2.0) * pp.MEGA / 300.0, "Pa*m^-1")
        pressure = p_m1000 - slope * (z - self.units.convert_units(-1000, "m"))
        return pressure


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
    """Pressurize specific fractures in their center."""

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
        ic(self.time_manager.time - pp.DAY, current_injection_overpressure)
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
        pressurized_interval = [fracture_sds[i] for i in self.pressurized_fractures]
        injection_coord = dict(
            zip(
                pressurized_interval,
                self.units.convert_units(self.fracture_centers, "m"),
            )
        )

        for i, sd in enumerate(subdomains):
            if sd in pressurized_interval:
                well_loc = injection_coord[sd]
                print(well_loc)
                well_loc_ind = sd.closest_cell(well_loc)
                sd_indicator[i][well_loc_ind] = 1

        # Characteristic functions
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1.0 - indicator

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
        eq_with_pressure_constraint.set_name(std_eq.name)

        return eq_with_pressure_constraint


# class CustomFracturePermeability(
#     pp.models.constitutive_laws.DimensionDependentPermeability
# ):
#     def fracture_permeability(self, subdomains):
#         # fracture_permeability = self.params["fracture_permeability"]
#         size = sum(sd.num_cells for sd in subdomains)
#         permeability = pp.wrap_as_dense_ad_array(
#             fracture_permeability, size, name="fracture permeability"
#         )
#         return self.isotropic_second_order_tensor(subdomains, permeability)
#
#     def intersection_permeability(self, subdomains):
#         # intersection_permeability = self.params["intersection_permeability"]
#         size = sum(sd.num_cells for sd in subdomains)
#         permeability = pp.wrap_as_dense_ad_array(
#             intersection_permeability, size, name="intersection permeability"
#         )
#         return self.isotropic_second_order_tensor(subdomains, permeability)


class SimpleTPFAFlow:
    """Simplified Flow discretization:

    * TPFA for flow.
    * Constant aperture in the normal flow.

    """

    # def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
    #    """Discretization object for the Darcy flux term.

    #    Parameters:
    #        subdomains: List of subdomains where the Darcy flux is defined.

    #    Returns:
    #        Discretization of the Darcy flux.

    #    """
    #    return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def interface_darcy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Darcy flux on interfaces.

        The units of the Darcy flux are [m^2 Pa / s], see note in :meth:`darcy_flux`.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        # Gradient operator in the normal direction. The collapsed distance is
        # :math:`\frac{a}{2}` on either side of the fracture.
        # We assume here that :meth:`aperture` is implemented to give a meaningful value
        # also for subdomains of co-dimension > 1.
        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg()
            @ self.aperture(subdomains).previous_iteration() ** pp.ad.Scalar(-1)
        )
        normal_gradient.set_name("normal_gradient")

        # Project the two pressures to the interface and multiply with the normal
        # diffusivity.
        pressure_l = projection.secondary_to_mortar_avg() @ self.pressure(subdomains)
        pressure_h = projection.primary_to_mortar_avg() @ self.pressure_trace(
            subdomains
        )
        eq = self.interface_darcy_flux(interfaces) - self.volume_integral(
            self.normal_permeability(interfaces)
            * (
                normal_gradient * (pressure_h - pressure_l)
                + self.interface_vector_source_darcy_flux(interfaces)
            ),
            interfaces,
            1,
        )
        eq.set_name("interface_darcy_flux_equation")
        return eq


class BedrettoValter_Physics(
    egc.HydrostaticPressureInitialCondition,
    PorePressure,
    HorizontalBackgroundStress,
    egc.HydrostaticPressureBC,
    egc.LithostaticPressureBC,
    PressureConstraintWell,
    # egc.HydrostaticPressureInitialization,
    # egc.EquilibriumStateInitialization, # FTHM IS NOT MADE FOR THIS (TRU FOR NCP)
    egc.AlternatingDecoupling,  # FTHM IS NOT MADE FOR THIS (TRU FOR NCP)
    pp.constitutive_laws.GravityForce,
    egc.ScalarPermeability,
    egc.NormalPermeabilityFromHigherDimension,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    # CustomFracturePermeability,
    SimpleTPFAFlow,
    pp.poromechanics.Poromechanics,  # Basic model
): ...
