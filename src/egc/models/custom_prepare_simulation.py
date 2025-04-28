"""Split pp.prepare_simulation into two parts."""


class TwoPartedPrepareSimulation:
    """Customized prepare_simulation method for the model.

    Purpose: Split into setup of geometry, equation system, and variables
    (prepare_simulation_1) and setup of discretization, linear solver, and
    nonlinear discretization (prepare_simulation_2). This allows to access
    the grid and thus assign e.g. TimeDepenedentAdArrays to the grid
    before the discretization is set up.

    """

    # def prepare_simulation(self) -> None:
    #    """Run at the start of simulation. Used for initialization etc."""
    #    if self.params.get("prepare_simulation_1", True):
    #        self.prepare_simulation_1()
    #    if self.params.get("prepare_simulation_2", True):
    #        self.prepare_simulation_2()

    def prepare_simulation_1(self) -> None:
        """Part 1 of prepare_simulation."""
        # Set the material and geometry of the problem. The geometry method must be
        # implemented in a ModelGeometry class.
        self.set_materials()
        self.set_geometry()

        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.create_variables()

    def prepare_simulation_2(self) -> None:
        """Part 2 of prepare_simulation."""
        # After fluid and variables are defined, we can define the secondary quantities
        # like fluid properties (which depend on variables). Creating fluid and
        # variables before defining secondary thermodynamic properties is critical in
        # the case where properties depend on some fractions. since the callables for
        # secondary variables are dynamically created during create_variables, as
        # opposed to e.g. pressure or temperature.
        self.assign_thermodynamic_properties_to_phases()
        self.initial_condition()
        self.initialize_previous_iterate_and_time_step_values()

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()
        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        # Export initial condition
        self.save_data_time_step()
