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

    def initial_pressure(self, subdomains: Optional[list[pp.Grid]]=None):
        if subdomains is None:
            return 0.
        else:
            return np.concatenate(
                [np.zeros(sd.num_cells) for sd in subdomains]
            )

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

