import porepy as pp
from typing import Callable
import numpy as np
from porepy.models.fluid_mass_balance import FluidMassBalanceEquations
import egc


class HydrostaticPressure:
    """Utility class to compute (generalized) hydrostatic pressure."""

    fluid: pp.FluidComponent

    def hydrostatic_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_atm = 0
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        fluid_density = self.fluid.density([sd]).value(self.equation_system)
        rho_g = fluid_density * gravity
        z = sd.cell_centers[self.nd - 1]
        pressure = p_atm - rho_g * z
        return pressure

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


class HydrostaticPressureBC(HydrostaticPressure):
    """Hydrostatic pressure boundary condition active on all sides of the domain."""

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    def _fluid_pressure_boundary_faces(self, sd: pp.Grid) -> np.ndarray:
        """Auxiliary method to identify all Dirichlet/pressure boundaries."""
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.all_bf

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self._fluid_pressure_boundary_faces(sd), "dir")

    @property
    def onset(self) -> bool:
        return self.time_manager.time > self.time_manager.schedule[0] + 1e-5

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        pressure = np.zeros(boundary_grid.num_cells)

        # Apply hydrostatic pressure on all sides of the domain.
        if boundary_grid.dim == self.nd - 1 and self.onset:
            sides = self.domain_boundary_sides(boundary_grid)
            pressure[sides.all_bf] = self.hydrostatic_pressure(boundary_grid)[
                sides.all_bf
            ]

        return pressure


class HydrostaticPressureInitialCondition(egc.InitialCondition, HydrostaticPressure):
    """Hydrostatic pressure based on cell coordinates as initial condition for the pressure."""

    def initial_pressure(self, sd=None):
        if sd is None:
            return self.reference_variable_values.pressure
        else:
            return np.concatenate(
                [self.hydrostatic_pressure(sd[i]) for i in range(len(sd))]
            )


class HydrostaticPressureInitialization(HydrostaticPressure):
    """Fix pressure to be the hydrostatic pressure for the first day."""

    def update_time_dependent_ad_arrays(self) -> None:
        """Deactivate pressure equations and set hydrostatic pressure for current gravity."""

        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        for sd in self.mdg.subdomains(return_data=False):
            active_indicator = float(self.time_manager.time_index == 1)
            pp.set_solution_values(
                name="pressure_constraint_indicator",
                values=active_indicator * np.ones(sd.num_cells, dtype=float),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites mass balance equation to set the pressure equal to the hydrostatic pressure."""

        eq = super().mass_balance_equation(subdomains)

        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = self.pressure(subdomains) - hydrostatic_pressure
        constrained_eq.set_name("mass_balance_equation_with_constrained_pressure")

        indicator = pp.ad.TimeDependentDenseArray(
            "pressure_constraint_indicator", subdomains
        )

        combined_eq = eq + indicator * (constrained_eq - eq)
        # YZ: Iterative linear solver relies on this name to find this equation.
        combined_eq.set_name(FluidMassBalanceEquations.primary_equation_name())
        return combined_eq
