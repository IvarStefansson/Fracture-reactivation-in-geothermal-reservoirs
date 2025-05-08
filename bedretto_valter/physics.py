import numpy as np
import porepy as pp
import egc
from icecream import ic
from porepy.applications.material_values.fluid_values import water

# Based on publications on BedrettoLab

# ! ---- MATERIAL PARAMETERS ----

fluid_parameters: dict[str, float] = water

# Values from Multi-disciplinary characterizations of the BedrettoLab
solid_parameters: dict[str, float] = {
    # Guessed
    "dilation_angle": 0.1,  # guessed # TODO
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
    # Well
    "well_radius": 0.10,  # m
}

numerics_parameters: dict[str, float] = {
    "open_state_tolerance": 1e-10,  # Numerical method parameter
    "characteristic_contact_traction": 10 * pp.MEGA,
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


class BedrettoValter_Physics(
    egc.HydrostaticPressureInitialCondition,
    PorePressure,
    HorizontalBackgroundStress,
    egc.HydrostaticPressureBC,
    egc.LithostaticPressureBC,
    egc.HydrostaticPressureInitialization,
    #egc.EquilibriumStateInitialization, # FTHM IS NOT MADE FOR THIS (TRU FOR NCP)
    pp.constitutive_laws.GravityForce,
    egc.ScalarPermeability,
    egc.NormalPermeabilityFromLowerDimension,
    #egc.NormalPermeabilityFromHigherDimension,
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    egc.TPFAFlow,
    egc.SimpleFlow,
    pp.poromechanics.Poromechanics,  # Basic model
): ...














# fracture_permeability = 1e-8 / 12
# intersection_permeability = 1e-8 / 12

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

