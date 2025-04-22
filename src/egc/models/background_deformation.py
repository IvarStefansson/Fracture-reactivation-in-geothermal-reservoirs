"""Functionality for defining displacements in a reference coordinate system."""

import porepy as pp
from typing import Sequence, cast


class BackgroundDeformation:
    """Redefinition of the mechanical stress taking background deformation into account.

    Background deformation is controlled by the parameters and is constant in time. It
    effectively represents the background stress.

    """

    def update_time_dependent_ad_arrays(self) -> None:
        """Fetch background displacement and interface displacement from parameters"""
        super().update_time_dependent_ad_arrays()

        background_deformation = self.params.get("background_deformation")
        assert background_deformation is not None
        for (var_name, var_domain), value in background_deformation.items():
            pp.set_solution_values(
                name="background_" + var_name,
                values=value,
                data=self.equation_system._get_data(var_domain),
                iterate_index=0,
                time_step_index=0,
            )

    def background_displacement(self, domains):
        return pp.ad.TimeDependentDenseArray("background_u", domains)

    def background_interface_displacement(self, interfaces):
        return pp.ad.TimeDependentDenseArray("background_u_interface", interfaces)

    def mechanical_stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Linear elastic mechanical stress enhanced by background deformation.

        .. note::
            Runs the superclass method to get the mechanical stress. Thus, the safety
            checks are already done.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """
        # Fetch the underlying mechanical stress operator
        stress = super().mechanical_stress(domains)

        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.stress_keyword, domains=domains
            )

        # By now we know that subdomains is a list of grids, so we can cast it as such
        # (in the typing sense). Also fetch the interfaces, and associated projections.
        domains = cast(list[pp.Grid], domains)
        interfaces = self.subdomains_to_interfaces(domains, [1])
        proj = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=self.nd)

        # Fetch the stress discretization
        discr = self.stress_discretization(domains)

        # External boundaries
        stress = stress + (
            discr.stress() @ self.background_displacement(domains)
            + discr.bound_stress()
            @ proj.mortar_to_primary_avg()
            @ self.background_interface_displacement(interfaces)
        )
        stress.set_name("mechanical_stress")
        return stress
