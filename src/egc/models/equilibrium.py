import porepy as pp
import porepy as pp

from porepy.models.fluid_mass_balance import FluidMassBalanceEquations
import logging
logger = logging.getLogger(__name__)

class ConstrainedSinglePhaseFlow:
    """Deactivate momentum balance equation."""

    def inactive_single_phase_flow(self) -> bool:
        """Define condition when to deactivate single phase flow."""
        return False

    def update_time_dependent_ad_arrays(self) -> None:
        """Define indicator for constraining flow equations."""

        super().update_time_dependent_ad_arrays()

        # Define indicator function for constraining flow equations
        indicator = self.inactive_single_phase_flow()
        if not hasattr(self, "single_phase_flow_constraint_indicator"):
            self.single_phase_flow_constraint_indicator = pp.ad.Scalar(indicator)
        else:
            self.single_phase_flow_constraint_indicator.set_value(indicator)

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites mass balance equation to constrain the pressure."""

        eq = super().mass_balance_equation(subdomains)
        constrained_eq = self.pressure(subdomains) - self.constrained_pressure(
            subdomains
        )
        indicator = self.single_phase_flow_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        # YZ: Iterative linear solver relies on this name to find this equation.
        combined_eq.set_name(FluidMassBalanceEquations.primary_equation_name())
        return combined_eq

    def interface_darcy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Overwrites interface darcy flux equation to constrain the interface Darcy flux."""

        eq = super().interface_darcy_flux_equation(interfaces)
        constrained_eq = self.interface_darcy_flux(
            interfaces
        ) - self.constrained_interface_darcy_flux(interfaces)
        indicator = self.single_phase_flow_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq

    def well_flux_equation(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Overwrites well flux equation to constrain the well flux."""

        eq = super().well_flux_equation(interfaces)
        constrained_eq = self.well_flux(interfaces) - self.constrained_well_flux(
            interfaces
        )
        indicator = self.single_phase_flow_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq


class ConstrainedMomentumBalance:
    """Deactivate momentum balance equation."""

    def inactive_momentum_balance(self) -> bool:
        """Define condition when to deactivate momentum balance."""
        return False

    def update_time_dependent_ad_arrays(self) -> None:
        """Define indicator for constraining momentum balance."""

        super().update_time_dependent_ad_arrays()

        # Define indicator function for constraining flow equations
        indicator = self.inactive_momentum_balance()
        if not hasattr(self, "momentum_balance_constraint_indicator"):
            self.momentum_balance_constraint_indicator = pp.ad.Scalar(indicator)
        else:
            self.momentum_balance_constraint_indicator.set_value(indicator)

    def momentum_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites momentum balance equation to constrain the displacement."""

        eq = super().momentum_balance_equation(subdomains)
        constrained_eq = self.displacement(subdomains) - self.constrained_displacement(
            subdomains
        )
        indicator = self.momentum_balance_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq

    def interface_force_balance_equation(
        self,
        interfaces: list[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """Overwrites interface force balance equation to constrain the interface displacement."""

        eq = super().interface_force_balance_equation(interfaces)
        constrained_eq = self.interface_displacement(
            interfaces
        ) - self.constrained_interface_displacement(interfaces)
        indicator = self.momentum_balance_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq

    def displacement_divergence(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        indicator = self.momentum_balance_constraint_indicator
        return super().displacement_divergence(subdomains) * (
            pp.ad.Scalar(1.0) - indicator
        )


class ConstrainedContactMechanics:
    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Overwrites normal fracture deformation equation to constrain the normal contact traction."""

        eq = super().normal_fracture_deformation_equation(subdomains)
        constrained_eq = self.normal_component(subdomains) @ self.contact_traction(
            subdomains
        ) - self.constrained_normal_contact_traction(subdomains)
        indicator = self.momentum_balance_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq

    def tangential_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Overwrites tangential fracture deformation equation to constrain the tangential contact traction."""

        eq = super().tangential_fracture_deformation_equation(subdomains)
        constrained_eq = self.tangential_component(subdomains) @ self.contact_traction(
            subdomains
        ) - self.constrained_tangential_contact_traction(subdomains)
        indicator = self.momentum_balance_constraint_indicator
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq


class EquilibriumStateInitialization(
    ConstrainedSinglePhaseFlow, ConstrainedMomentumBalance, ConstrainedContactMechanics
):
    """Initialization of flow and momentum balance equations through decoupling."""

    def inactive_momentum_balance(self):
        return self.time_manager.time_index == 0

    def inactive_single_phase_flow(self):
        return self.time_manager.time_index == 1

    def constrained_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.displacement(subdomains).previous_iteration()

    def constrained_interface_displacement(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        return self.interface_displacement(interfaces).previous_iteration()

    def constrained_normal_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        return (
            self.normal_component(subdomains)
            @ self.contact_traction(subdomains).previous_iteration()
        )

    def constrained_tangential_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        return (
            self.tangential_component(subdomains)
            @ self.contact_traction(subdomains).previous_iteration()
        )

    def constrained_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.pressure(subdomains).previous_iteration()

    def constrained_interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        return self.interface_darcy_flux(interfaces).previous_iteration()

    def constrained_well_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        return self.well_flux(interfaces).previous_iteration()

class CacheReferenceState:

    def cache_reference_state(self) -> None:
        """Caching of reference states."""

        if not hasattr(self, "has_reference_flow_state"):
            for sd in self.mdg.subdomains(return_data=False):
                pp.set_solution_values(
                    name="reference_pressure",
                    values=self.pressure([sd]).value(self.equation_system),
                    data=self.mdg.subdomain_data(sd),
                    iterate_index=0,
                )
            self.has_reference_flow_state = True

        if not hasattr(self, "has_reference_momentum_state"):
            for sd in self.mdg.subdomains(return_data=False, dim=self.nd):
                pp.set_solution_values(
                    name="reference_displacement",
                    values=self.displacement([sd]).value(self.equation_system),
                    data=self.mdg.subdomain_data(sd),
                    iterate_index=0,
                )
            for intf in self.mdg.interfaces(return_data=False, dim=self.nd - 1):
                pp.set_solution_values(
                    name="reference_interface_displacement",
                    values=self.interface_displacement([intf]).value(
                        self.equation_system
                    ),
                    data=self.mdg.interface_data(intf),
                    iterate_index=0,
                )
            self.has_reference_momentum_state = True

    def update_time_dependent_ad_arrays(self) -> None:
        """Cache reference state in data dictionaries."""
        super().update_time_dependent_ad_arrays()
        if self.time_manager.time_index == 2:
            self.cache_reference_state()
            logger.info("Caching reference state")


class AlternatingDecouplingInTime(EquilibriumStateInitialization):
    """Initialization of flow and momentum balance equations through decoupling."""

    def inactive_momentum_balance(self):
        return self.time_manager.time_index % 2 == 0

    def inactive_single_phase_flow(self):
        return self.time_manager.time_index % 2 == 1

class AlternatingDecouplingInNewton(EquilibriumStateInitialization):
    """Initialization of flow and momentum balance equations through decoupling."""

    def inactive_momentum_balance(self):
        return self.nonlinear_solver_statistics.num_iteration % 2 == 0

    def inactive_single_phase_flow(self):
        return self.nonlinear_solver_statistics.num_iteration % 2 == 1