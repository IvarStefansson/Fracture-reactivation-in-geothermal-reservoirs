"""Provide mixin to inherit the normal permeability from the neighbouring
objects.

"""

import porepy as pp


class ScalarPermeability:
    """NOTE: Assumes cubic law in fractures and intersections."""

    def scalar_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        projection = pp.ad.SubdomainProjections(subdomains)
        matrix: list[pp.Grid] = [sd for sd in subdomains if sd.dim == self.nd]
        fractures: list[pp.Grid] = [sd for sd in subdomains if sd.dim == self.nd - 1]
        intersections: list[pp.Grid] = [sd for sd in subdomains if sd.dim < self.nd - 1]

        scalar_permeability = (
            projection.cell_prolongation(matrix)
            @ self.scalar_matrix_permeability(matrix)
            + projection.cell_prolongation(fractures)
            @ self.scalar_fracture_permeability(fractures)
            + projection.cell_prolongation(intersections)
            @ self.scalar_intersection_permeability(intersections)
        )
        return scalar_permeability

    def scalar_matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Get the scalar permeability for the subdomains."""
        size = sum(sd.num_cells for sd in subdomains)
        scalar_permeability = pp.wrap_as_dense_ad_array(
            self.solid.permeability, size, name="scalar_permeability"
        )
        return scalar_permeability

    def scalar_fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Get the scalar permeability for the subdomains."""
        aperture = self.aperture(subdomains)
        scalar_permeability = (aperture ** pp.ad.Scalar(2)) / pp.ad.Scalar(12.0)
        # TODO: Check if safer:
        # scalar_permeability = (aperture * aperture) / pp.ad.Scalar(12.)
        scalar_permeability.set_name("scalar_fracture_permeability")
        return scalar_permeability

    def scalar_intersection_permeability(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Get the scalar permeability for the subdomains."""
        aperture = self.aperture(subdomains)
        scalar_permeability = (aperture ** pp.ad.Scalar(2)) / pp.ad.Scalar(12.0)
        # TODO: Check if safer - should be the same due to safegour:
        # scalar_permeability = (aperture * aperture) / pp.ad.Scalar(12.)
        scalar_permeability.set_name("scalar_intersection_permeability")
        return scalar_permeability


class NormalPermeabilityFromLowerDimension:
    """Deduct the normal permeability from lower dimensional subdomain."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Get the normal permeability from lower dimensional subdomain.

        NOTE: Requires a scalar permeability to be defined as the permeability
        otherwise comes as tensor.

        """

        assert hasattr(self, "scalar_permeability")

        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        normal_permeability = (
            projection.secondary_to_mortar_avg() @ self.scalar_permeability(subdomains)
        )
        normal_permeability.set_name("normal_permeability_from_lower_dimension")
        return normal_permeability


class NormalPermeabilityFromHigherDimension:
    """Deduct the normal permeability from higher dimensional subdomain."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Get the normal permeability from higher dimensional subdomain.

        NOTE: Requires a scalar permeability to be defined as the permeability
        otherwise comes as tensor.

        """

        assert hasattr(self, "scalar_permeability")

        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        trace = pp.ad.Trace(subdomains)
        trace_permeability = trace.trace @ self.scalar_permeability(subdomains)
        trace_permeability.set_name("trace_permeability")
        normal_permeability = projection.primary_to_mortar_avg() @ trace_permeability
        normal_permeability.set_name("normal_permeability_from_higher_dimension")
        return normal_permeability
