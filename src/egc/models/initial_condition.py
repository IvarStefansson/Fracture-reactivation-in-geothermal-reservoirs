import porepy as pp
import numpy as np
from typing import Optional


class InitialCondition:
    """Initial conditions:

    * Zero displacement.
    * (Zero interface displacement.)
    * Zero pressure.
    * (Zero interface flux.)
    * Zero contact traction.

    """

    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        super().initial_condition()
        for var in self.equation_system.variables:
            if hasattr(self, "initial_" + var.name):
                values = getattr(self, "initial_" + var.name)([var.domain])
                self.equation_system.set_variable_values(
                    values, [var], iterate_index=0, time_step_index=0
                )

    def initial_pressure(self, subdomains: Optional[list[pp.Grid]] = None):
        if subdomains is None:
            return 0.0
        else:
            return np.concatenate([np.zeros(sd.num_cells) for sd in subdomains])

    def initial_displacement(self, sd=None):
        # Want to actually solve a mechanics problem alone with glued fractures.
        if sd is None:
            return np.zeros((self.nd, self.mdg.num_cells))
        else:
            return np.zeros((self.nd, sd.num_cells))

    def initial_contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Initial contact traction [Pa].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for initial contact traction.

        """
        assert len(subdomains) == 1
        sd = subdomains[0]
        traction_vals = np.zeros((self.nd, sd.num_cells))
        return traction_vals.ravel("F")


class InitialConditionFromParameters:
    """Utility to set initial conditions from parameters dictionary.

    Requires "initial_condition" key in the parameters dictionary. Enables
    setting initial conditions for all variables at once, or specific
    variables.

    """

    def initial_condition(self) -> None:
        """Set the initial condition for the problem.

        Allow for setting initial conditions for specific variables or all at once.
        If "initial_condition" is a dictionary, it should contain variable names
        as keys and their corresponding values as values. If it is a numpy array,
        it should contain the full initial condition for all variables.

        """
        super().initial_condition()

        if "initial_condition" in self.params and isinstance(
            self.params["initial_condition"], dict
        ):
            # Option for setting initial conditions for specific variables

            for (var_name, var_domain), value in self.params[
                "initial_condition"
            ].items():
                pp.set_solution_values(
                    name=var_name,
                    values=value,
                    data=self.equation_system._get_data(var_domain),
                    iterate_index=0,
                    time_step_index=0,
                )

        elif "initial_condition" in self.params and isinstance(
            self.params["initial_condition"], np.ndarray
        ):
            # Option for setting the full initial condition at once
            value = self.params["initial_condition"]
            self.equation_system.set_variable_values(
                value, iterate_index=0, time_step_index=0
            )
