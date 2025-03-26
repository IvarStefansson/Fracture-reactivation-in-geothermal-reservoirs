# To install the package, see ./scripts/install_fthm.sh and /scripts/install_petsc.sh

import porepy as pp
import numpy as np
from FTHM_Solver.thm_solver import THMSolver


class MyModel(THMSolver, pp.Thermoporomechanics):
    """This setup describes:

    * 2D box 1x1 meter.
    * Two intersecting fractures.
    * Injecting cold fluid in the intersection.
    * Dirichlet boundary conditions.

    """

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self.reference_variable_values.temperature

    def ic_values_pressure(self, sd):
        return np.ones(sd.num_cells) * self.reference_variable_values.pressure

    def locate_source(self, subdomains):
        num_cells = sum(sd.num_cells for sd in subdomains)
        src_location = np.zeros(num_cells)
        src_location[-1] = 1
        return src_location

    def fluid_source_mass_rate(self):
        return self.units.convert_units(0.001, "kg * s^-1")

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        return super().fluid_source(subdomains) + pp.ad.DenseArray(src)

    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        src = self.locate_source(subdomains)
        src *= self.fluid_source_mass_rate()
        cv = self.fluid.components[0].specific_heat_capacity

        # Relative to the reference temperature.
        t_inj = self.units.convert_units(-40, "K")
        src *= cv * t_inj
        return super().energy_source(subdomains) + pp.ad.DenseArray(src)

    def set_fractures(self) -> None:
        self._fractures = [
            pp.LineFracture([[0.2, 0.8], [0.5, 0.5]]),
            pp.LineFracture([[0.5, 0.5], [0.2, 0.8]]),
        ]


HOUR = 60 * 60

params = {
    "material_constants": {
        "solid": pp.SolidConstants(
            permeability=1e-13,  # [m^2]
            residual_aperture=1e-3,  # [m]
            shear_modulus=1.2e10,  # [Pa]
            lame_lambda=1.2e10,  # [Pa]
            dilation_angle=5 * np.pi / 180,  # [rad]
            normal_permeability=1e-4,
            biot_coefficient=0.47,  # [-]
            density=2683.0,  # [kg * m^-3]
            porosity=1.3e-2,  # [-]
            friction_coefficient=0.577,  # [-]
            # Thermal
            specific_heat_capacity=720.7,
            thermal_conductivity=0.1,  # Diffusion coefficient
            thermal_expansion=9.66e-6,
        ),
        "fluid": pp.FluidComponent(
            compressibility=4.559 * 1e-10,  # [Pa^-1], fluid compressibility
            density=998.2,  # [kg m^-3]
            viscosity=1.002e-3,  # [Pa s], absolute viscosity
            # Thermal
            specific_heat_capacity=4182.0,
            thermal_conductivity=0.5975,  # Diffusion coefficient
            thermal_expansion=2.068e-4,  # Density(T)
        ),
        "numerical": pp.NumericalConstants(
            characteristic_displacement=2e0,  # [m]
        ),
    },
    "reference_variable_values": pp.ReferenceVariableValues(
        pressure=3.5e7,  # [Pa]
        temperature=273 + 120,
    ),
    "grid_type": "simplex",
    "units": pp.Units(kg=1e10),
    "meshing_arguments": {
        "cell_size": 0.1,
    },
    "time_manager": pp.TimeManager(
        dt_init=10,
        schedule=[0, HOUR],
        iter_max=10,
        constant_dt=False,
    ),
    "adaptive_indicator_scaling": 1,
    "progressbars": True,
    "linear_solver_config": {
        "solver": "CPR",  # Avaliable options: CPR, SAMG, FGMRES (fastest to slowest).
        "ksp_monitor": False,  # Enable to see convergence messages from PETSc.
        "logging": False,  # Does not work well with a progress bar.
    },
}
model = MyModel(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, cell_value="temperature", plot_2d=True, fracturewidth_1d=2)


# NOTE: The following message is expected and is not an error:

# WARNING! There are options you set that were not used!
# WARNING! could be spelling mistake, etc!
# There are 3 unused database options. They are:
# Option left: name:-fieldsplit_1-2-3-4-5-6-7-8-9-10_fieldsplit_3-4-5-6-7-8-9-10_fieldsplit_5-6-7-8-9-10_sub_0_pc_fieldsplit_schur_fact_type value: upper source: code
# Option left: name:-fieldsplit_1-2-3-4-5-6-7-8-9-10_fieldsplit_3-4-5-6-7-8-9-10_fieldsplit_5-6-7-8-9-10_sub_0_pc_fieldsplit_schur_precondition value: selfp source: code
# Option left: name:-fieldsplit_1-2-3-4-5-6-7-8-9-10_fieldsplit_3-4-5-6-7-8-9-10_fieldsplit_5-6-7-8-9-10_sub_1_ksp_type value: preonly source: code
