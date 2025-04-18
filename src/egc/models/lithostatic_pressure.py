import porepy as pp
from typing import Callable
import numpy as np

class BackgroundStress:
    def vertical_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Vertical background stress."""
        gravity = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        rho_g = self.solid.density * gravity
        z = grid.cell_centers[self.nd-1]
        s_v = -rho_g * z
        return -s_v

    def horizontal_background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Zero horizontal background stress."""
        s_v = self.vertical_background_stress(grid)
        s_h = np.zeros((self.nd - 1, self.nd - 1, grid.num_cells))
        scaling = 0.
        for i, j in np.ndindex(self.nd-1, self.nd-1):
            s_h[i, j] = scaling * s_v
        return s_h

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        """Combination of vertical (lithostatic) and horizontal stress."""

        s_h = self.horizontal_background_stress(grid)
        s_v = self.vertical_background_stress(grid)
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        for i, j in np.ndindex(self.nd-1, self.nd-1):
            s[i, j] = s_h[i, j]
        s[-1, -1] = s_v
        return s

class LithostaticPressureBC:
    """Mechanical boundary conditions.

    * Zero displacement boundary condition active on the bottom of the domain.
    * Lithostatic pressure on remaining boundaries.
    * Additional background stress applied to the domain.

    """

    solid: pp.SolidConstants

    nd: int

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    time_manager: pp.TimeManager

    onset: bool

    def _momentum_balance_dirichlet_sides(self, sd: pp.Grid) -> np.ndarray:
        domain_sides = self.domain_boundary_sides(sd)
        if self.nd == 2:
            return [domain_sides.south]
        elif self.nd == 3:
            return [domain_sides.bottom]
        
    def _momentum_balance_neumann_sides(self, sd: pp.Grid) -> np.ndarray:
        domain_sides = self.domain_boundary_sides(sd)
        if self.nd == 2:
            return [
                domain_sides.north,
                domain_sides.east,
                domain_sides.west,
            ]
        elif self.nd == 3:
            return [
                domain_sides.north,
                domain_sides.south,
                domain_sides.east,
                domain_sides.west,
                domain_sides.top,
            ]

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        bc.internal_to_dirichlet(sd)

        for side in self._momentum_balance_dirichlet_sides(sd):
            bc.is_dir[:, side] = True
            bc.is_neu[:, side] = False
        for side in self._momentum_balance_neumann_sides(sd):
            bc.is_dir[:, side] = False
            bc.is_neu[:, side] = True

        return bc

    def _orientations(self, boundary_grid: pp.BoundaryGrid) -> tuple:
        domain_sides = self.domain_boundary_sides(boundary_grid)
        if self.nd == 2:
            return (
                [0,0,1,1],
                [-1,1,-1,1],
                [domain_sides.west, domain_sides.east, domain_sides.south, domain_sides.north],
            )
        elif self.nd == 3:
            return (
                [0,0,1,1,2,2],
                [-1,1,-1,1,-1,1],
                [
                    domain_sides.west,
                    domain_sides.east,
                    domain_sides.south,
                    domain_sides.north,
                    domain_sides.bottom,
                    domain_sides.top,
                ],
            )

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Background stress applied to boundary."""
        vals = np.zeros((self.nd, boundary_grid.num_cells))

        # Lithostatic stress on the domain boundaries.
        if boundary_grid.dim == self.nd - 1 and self.onset:
            background_stress_tensor = self.background_stress(boundary_grid)

            # Stress times normal
            directions, orientations, sides = self._orientations(boundary_grid)
            for dir, orientation, side in zip(directions, orientations, sides):
                active_side = side[np.isin(side, np.concatenate(self._momentum_balance_neumann_sides(boundary_grid)))]
                if not active_side.any():
                    continue
                for i in range(self.nd):
                    vals[i, active_side] = (
                        orientation
                        * background_stress_tensor[i, dir, active_side]
                        * boundary_grid.cell_volumes[active_side]
                    )

        return vals.ravel("F")
