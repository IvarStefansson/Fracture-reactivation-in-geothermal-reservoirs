# TODO: Integrate into NCP after experimenting

import numpy as np
import scipy.sparse as sps
import porepy as pp

import logging

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScaledNCPAdapters:
    def characteristic_distance(self, subdomains):
        return self.characteristic_displacement(subdomains)


class AdaptiveDarcysLawAd:  # (pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        no_contact_states_changes = self.no_contact_states_change(subdomains).value(
            self.equation_system
        )[0]

        if all([sd.dim < self.nd for sd in subdomains]) and np.isclose(
            no_contact_states_changes, 1
        ):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)


class DarcysLawAd(pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)


class ReverseElasticModuli:
    """Same as ElasticModuli, but with reversed assignment of characteristic values."""

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Characteristic traction [Pa].

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """
        t_char = pp.ad.Scalar(self.numerical.characteristic_contact_traction)
        t_char.set_name("characteristic_contact_traction")
        return t_char

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Characteristic displacement [m].

        Parameters:
            subdomains: List of subdomains where the characteristic displacement is
                defined.

        Returns:
            Scalar operator representing the characteristic displacement.

        """
        size = pp.ad.Scalar(np.max(self.domain.side_lengths()))
        u_char = (
            self.characteristic_contact_traction(subdomains)
            * size
            / self.youngs_modulus(subdomains)
        )
        u_char.set_name("characteristic_displacement")
        return u_char