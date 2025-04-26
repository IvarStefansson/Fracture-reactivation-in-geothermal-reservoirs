import porepy as pp
import porepy as pp
import numpy as np

from porepy.models.fluid_mass_balance import FluidMassBalanceEquations

class ConstrainedSinglePhaseFlow:
    """Deactivate momentum balance equation."""

    def active_equilibrium_momentum_balance(self) -> bool:
        """Deactivate momentum balance equation."""
        return False
    
    def update_time_dependent_ad_arrays(self) -> None:
        """Deactivate pressure equations and set hydrostatic pressure for current gravity."""

        super().update_time_dependent_ad_arrays()

        # Define indicator function for constraining flow equations
        indicator = self.active_equilibrium_momentum_balance()
        
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="flow_constraint_indicator",
                values=indicator * np.ones(sd.num_cells, dtype=float),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

        for intf in self.mdg.interfaces(return_data=False):
            pp.set_solution_values(
                name="flow_constraint_indicator",
                values=indicator * np.ones(intf.num_cells, dtype=float),
                data=self.mdg.interface_data(intf),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Overwrites mass balance equation to set the pressure equal to the hydrostatic pressure."""

        eq = super().mass_balance_equation(subdomains)

        constrained_pressure = pp.ad.TimeDependentDenseArray(
            "constrained_pressure", subdomains
        )
        constrained_eq = self.pressure(subdomains) - constrained_pressure
        constrained_eq.set_name("mass_balance_equation_with_constrained_pressure")

        indicator = pp.ad.TimeDependentDenseArray(
            "flow_constraint_indicator", subdomains
        )

        combined_eq = eq + indicator * (constrained_eq - eq)
        # YZ: Iterative linear solver relies on this name to find this equation.
        combined_eq.set_name(FluidMassBalanceEquations.primary_equation_name())
        return combined_eq
    
    def interface_darcy_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        eq = super().interface_darcy_flux_equation(interfaces)

        constrained_interface_darcy_flux = pp.ad.TimeDependentDenseArray(
            "constrained_interface_darcy_flux", self.mdg.interfaces()
        )
        constrained_eq = self.interface_darcy_flux(interfaces) - constrained_interface_darcy_flux
        constrained_eq.set_name("interface_darcy_flux_equation_with_constrained_interface_darcy_flux")
        indicator = pp.ad.TimeDependentDenseArray(
            "flow_constraint_indicator", self.mdg.interfaces()
        )
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq
        
    def well_flux_equation(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        eq = super().well_flux_equation(interfaces)

        constrained_well_flux = pp.ad.TimeDependentDenseArray(
            "constrained_well_flux", self.mdg.interfaces()
        )
        constrained_eq = self.well_flux(interfaces) - constrained_well_flux
        constrained_eq.set_name("well_flux_equation_with_constrained_well_flux")
        indicator = pp.ad.TimeDependentDenseArray(
            "flow_constraint_indicator", self.mdg.interfaces()
        )
        combined_eq = eq + indicator * (constrained_eq - eq)
        combined_eq.set_name(eq.name)
        return combined_eq

