from dataclasses import dataclass
from typing import Callable, ClassVar
from warnings import warn

import numpy as np
import porepy as pp

# ! ---- MATERIAL PARAMETERS ----

fluid_parameters: dict[str, float] = {
    "compressibility": 0,  # 1e-16,
    "viscosity": 1e-3,
    "density": 998.2e0,
}

solid_parameters: dict[str, float] = {
    "biot_coefficient": 1.0,
    "permeability": 1e-14,
    "normal_permeability": 1e-14,  # 1e-4, # Ivar: 1e-6
    "porosity": 1.0e-2,
    "shear_modulus": 1e14,
    "lame_lambda": 1e14,
    "residual_aperture": 1e-3,  # 1e-2, # Ivar: 1e-3
    "density": 2600,
    "maximum_elastic_fracture_opening": 0e-3,  # Not used
    "fracture_normal_stiffness": 1e3,  # Not used
    "fracture_tangential_stiffness": -1,
    "fracture_gap": 0e-3,  # Equals the maximum fracture closure.
    "dilation_angle": 0.1,
    "friction_coefficient": 0.8,
}

# TODO!
injection_schedule = {
    "time": [pp.DAY * 0.2, pp.DAY * 0.4, pp.DAY * 0.6, pp.DAY * 0.8, pp.DAY],
    "pressure": [0e6] * 5,
    "reference_pressure": 1e7,
}

# TODO!
principal_background_stress_max_factor = 0.0  # 1.3  # 24e6  # 24 MPa
principal_background_stress_min_factor = 0.0  # 0.8  # 14e6  # 14 MPa
background_stress_deg = 100 * (np.pi / 180)  # N100 degrees East

# TODO!
numerics_parameters: dict[str, float] = {
    # "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_displacement": 1.0,
    "characteristic_contact_traction": 1.0,
}


@dataclass(kw_only=True, eq=False)
class ExtendedNumericalConstants(pp.NumericalConstants):
    contact_mechanics_scaling_t: float

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "characteristic_displacement": "m",
            "characteristic_contact_traction": "Pa",
            "open_state_tolerance": "-",
            "contact_mechanics_scaling": "-",
            "contact_mechanics_scaling_t": "-",
        }
    )

    @property
    def default_constants(self):
        default_constants = super().default_constants
        default_constants.update({"contact_mechanics_scaling_t": 1.0})
        return default_constants

    def contact_mechanics_scaling_t(self):
        return self.constants["contact_mechanics_scaling_t"]


class ADTime:
    def time(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """AD variant of time .

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for time.

        """
        return pp.ad.TimeDependentDenseArray("time", [self.mdg.subdomains()[0]])

    def update_time_ones(self) -> None:
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="time",
                values=self.units.convert_units(
                    np.array([self.time_manager.time]), "s"
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        self.update_time_ones()


class RampedGravity:
    def update_time_dependent_ad_arrays(self) -> None:
        super().update_time_dependent_ad_arrays()
        if not hasattr(self, "ref_gravity"):
            self.ref_gravity = pp.GRAVITY_ACCELERATION
        ramp = 10
        if self.time_manager.time < ramp * pp.DAY:
            pp.GRAVITY_ACCELERATION = (
                self.time_manager.time / (ramp * pp.DAY) * self.ref_gravity
            )
        else:
            pp.GRAVITY_ACCELERATION = self.ref_gravity


class HydrostaticPressureBC:
    """Hydrostatic pressure boundary condition active on all sides of the domain."""

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
    is_well: Callable[[pp.Grid], bool]
    fluid: pp.FluidComponent

    def _fluid_pressure_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Auxiliary method to identify all Dirichlet/pressure boundaries."""
        if sd.dim == self.nd:
            domain_sides = self.domain_boundary_sides(sd)
            faces = domain_sides.all_bf
        else:
            faces = np.array([], dtype=int)
        return faces

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    @property
    def onset(self) -> bool:
        return self.time_manager.time > self.time_manager.schedule[0] + 1e-5

    def hydrostatic_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_atm = 0
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        fluid_density = self.fluid.density([sd]).value(self.equation_system)
        rho_g = fluid_density * gravity
        z = sd.cell_centers[-1]
        pressure = p_atm - rho_g * z
        return pressure

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        pressure = np.zeros(boundary_grid.num_cells)

        # Apply hydrostatic pressure on all sides of the domain.
        if boundary_grid.dim == self.nd - 1 and self.onset:
            sides = self.domain_boundary_sides(boundary_grid)
            pressure[sides.all_bf] = self.hydrostatic_pressure(boundary_grid)[
                sides.all_bf
            ]

        return pressure


class BackgroundStress:
    def vertical_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Vertical background stress."""
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        bulk_density = (
            self.solid.porosity * self.fluid.density([grid]).value(self.equation_system)
            + (1 - self.solid.porosity) * self.solid.density
        )
        rho_g = bulk_density * gravity
        z = grid.cell_centers[-1]
        s_v = -rho_g * z
        return s_v

    def horizontal_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Horizontal background stress."""
        s_v = self.vertical_background_stress(grid)
        s_h = np.zeros((self.nd - 1, self.nd - 1, grid.num_cells))
        principal_stress_factor = np.array(
            [
                [principal_background_stress_max_factor, 0],
                [0, principal_background_stress_min_factor],
            ]
        )
        orientation = np.array(
            [
                [np.cos(background_stress_deg), np.sin(background_stress_deg)],
                [-np.sin(background_stress_deg), np.cos(background_stress_deg)],
            ]
        )
        scaling = orientation @ principal_stress_factor @ orientation.T
        for i, j in np.ndindex(2, 2):
            s_h[i, j] = scaling[i, j] * s_v
        return s_h

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Combination of vertical (lithostatic) and horizontal stress."""

        s_h = self.horizontal_background_stress(grid)
        s_v = self.vertical_background_stress(grid)
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        for i, j in np.ndindex(2, 2):
            s[i, j] = s_h[i, j]
        s[-1, -1] = s_v
        return s


class LithostaticPressureBC:
    """Mechanical boundary conditions.

    * Zero displacement boundary condition active on the bottom of the domain.
    * Lithostatic pressure on remaining boundaries.
    * Additional background stress applied to the domain.

    """

    solid: pp.SolidConstants

    nd: int

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    time_manager: pp.TimeManager

    onset: bool

    def _displacement_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        return self.domain_boundary_sides(sd).bottom

    def _stress_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        all_faces = self.domain_boundary_sides(sd).all_bf
        return np.logical_and(
            all_faces, np.logical_not(self._displacement_boundary_faces(sd))
        )

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bc = pp.BoundaryConditionVectorial(
            sd, self._displacement_boundary_faces(sd), "dir"
        )
        # Default internal BC is Neumann. We change to Dirichlet, i.e., the
        # mortar variable represents the displacement on the fracture faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Background stress applied to boundary."""
        vals = np.zeros((self.nd, boundary_grid.num_cells))

        # Lithostatic stress on the domain boundaries.
        if boundary_grid.dim == self.nd - 1 and self.onset:
            normals = boundary_grid.cell_normals
            background_stress_tensor = self.background_stress(boundary_grid)
            surface_stress = np.einsum("ijk,jk->ik", background_stress_tensor, normals)
            neumann_faces = self._stress_boundary_faces(boundary_grid)
            vals[:, neumann_faces] = surface_stress[:, neumann_faces]

        return vals.ravel("F")


class NonzeroInitialCondition:
    """Start in equilibrium:

    * Zero displacement.
    * Zero interface displacement.
    * Hydrostatic pressure.
    * Zero interface flux.
    * Compatible contact traction (lithostatic and hydrostatic pressure).

    """

    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        super().initial_condition()
        for var in self.equation_system.variables:
            if hasattr(self, "initial_" + var.name):
                values = getattr(self, "initial_" + var.name)([var.domain])
                self.equation_system.set_variable_values(
                    values, [var], iterate_index=0, time_step_index=0
                )

    def initial_pressure(self, sd=None):
        if sd is None:
            return self.reference_variable_values.pressure
        else:
            return np.concatenate(
                [self.hydrostatic_pressure(sd[i]) for i in range(len(sd))]
            )

    def initial_displacement(self, sd=None):
        # Want to actually solve a mechanics problem alone with glued fractures.
        if sd is None:
            return np.zeros((self.nd, self.mdg.num_cells))
        else:
            return np.zeros((self.nd, sd.num_cells))

    def initial_contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Initial contact traction [Pa].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for initial contact traction.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]
        traction_vals = np.zeros((self.nd, sd.num_cells))
        # traction_vals[-1] = -1
        return traction_vals.ravel("F")


class GlobalHydrostaticPressure:
    def update_time_dependent_ad_arrays(self) -> None:
        """Set hydrostatic pressure for current gravity."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        for sd in self.mdg.subdomains(return_data=False):
            hydrostatic_pressure = self.hydrostatic_pressure(sd)
            pp.set_solution_values(
                name="hydrostatic_pressure",
                values=np.array(hydrostatic_pressure),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

            pp.set_solution_values(
                name="pressure_constraint_indicator",
                values=np.array([self.time_manager.time < 2 * pp.DAY + 1e-5]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = self.pressure(subdomains) - hydrostatic_pressure
        constrained_eq.set_name("mass_balance_equation_with_constrained_pressure")

        indicator = pp.ad.TimeDependentDenseArray(
            "pressure_constraint_indicator", [self.mdg.subdomains()[0]]
        )

        eq = super().mass_balance_equation(subdomains)

        combined_eq = indicator * constrained_eq + (pp.ad.Scalar(1.0) - indicator) * eq
        return combined_eq


# class PressureConstraintWell:
#     def update_time_dependent_ad_arrays(self) -> None:
#         """Set current injection pressure."""
#         super().update_time_dependent_ad_arrays()
#
#         # Update injection pressure
#         current_injection_pressure = np.interp(
#             self.time_manager.time,
#             injection_schedule["time"],
#             injection_schedule["pressure"],
#             left=0.0,
#         )
#         for sd in self.mdg.subdomains(return_data=False):
#             pp.set_solution_values(
#                 name="current_injection_pressure",
#                 values=np.array([current_injection_pressure]),
#                 data=self.mdg.subdomain_data(sd),
#                 iterate_index=0,
#             )
#
#     def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#         std_eq = super().mass_balance_equation(subdomains)
#
#         # Need to embedd in full domain
#         sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]
#
#         # Pick the only subdomain
#         fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]
#
#         if len(fracture_sds) == 0:
#             return std_eq
#
#         # Pick a single fracture
#         well_sd = fracture_sds[0]
#
#         for i, sd in enumerate(subdomains):
#             if sd == well_sd:
#                 # Pick the center (hardcoded)
#                 well_loc = np.array(
#                     [
#                         self.units.convert_units(1500, "m"),
#                         self.units.convert_units(1500, "m"),
#                         0,
#                     ]
#                 ).reshape((3, 1))
#
#                 well_loc_ind = sd.closest_cell(well_loc)
#
#                 sd_indicator[i][well_loc_ind] = 1
#
#         # Characteristic functions
#         indicator = np.concatenate(sd_indicator)
#         reverse_indicator = 1 - indicator
#
#         current_injection_pressure = pp.ad.TimeDependentDenseArray(
#             "current_injection_pressure", [self.mdg.subdomains()[0]]
#         )
#         constrained_eq = self.pressure(subdomains) - current_injection_pressure
#
#         # pp.ad.Scalar(self.units.convert_units(injection_pressure, "Pa"))
#
#         eq_with_pressure_constraint = (
#             pp.ad.DenseArray(reverse_indicator) * std_eq
#             + pp.ad.DenseArray(indicator) * constrained_eq
#         )
#         eq_with_pressure_constraint.set_name(
#             "mass_balance_equation_with_constrained_pressure"
#         )
#
#         return eq_with_pressure_constraint


class Physics(
    ADTime,
    BackgroundStress,
    HydrostaticPressureBC,
    LithostaticPressureBC,
    # NonzeroInitialCondition,
    GlobalHydrostaticPressure,
    # PressureConstraintWell,
    pp.constitutive_laws.GravityForce,
): ...
