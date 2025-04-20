"""Simple linear fluid compressibility model.

Purpose: If Newton's method generates large pressures throughout
the iterative process, the exponential term in the default compressibility
model can cause numerical issues. This model uses a linear approximation
of the compressibility term, which is valid for small perturbations.
It shall mostly serve for numerical troubleshooting.

"""
import porepy as pp
from typing import cast

class LinearFluidCompressibility:

    def density_of_phase(self, phase: pp.Phase):

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            rho_ref = pp.ad.Scalar(
                self.fluid.reference_component.density, "reference_fluid_density"
            )
            rho_ = rho_ref + self.linear_pressure_deviation(cast(list[pp.Grid], domains))
            rho_.set_name("linear_fluid_density")
            return rho_

        return rho

    def linear_pressure_deviation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        c = self.fluid_compressibility(subdomains)
        dp = self.perturbation_from_reference("pressure", subdomains)
        linear_pressure_deviation = c * dp
        linear_pressure_deviation.set_name("linear_pressure_deviation")
        return linear_pressure_deviation