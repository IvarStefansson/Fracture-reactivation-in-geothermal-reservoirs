import porepy as pp


class TPFAFlow:
    """Simplified Flow discretization:

    * TPFA for flow.
    * Constant aperture in the normal flow.

    """

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        if not self.params.get("use_tpfa_flow", False):
            return super().darcy_flux_discretization(subdomains)
        else:
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)


class SimpleFlow:
    """Simplified Flow discretization:

    * TPFA for flow.
    * Constant aperture in the normal flow.

    """

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
        if not self.params.get("use_simple_flow", False):
            return super().interface_darcy_flux_equation(interfaces)

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
