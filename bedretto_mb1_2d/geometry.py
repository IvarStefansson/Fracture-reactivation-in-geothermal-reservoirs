"""Implementation of two-dimensional Bedretto-like geometry inspired
from Vaezi et al (2024), "Numerical modeling of hydraulic stimulation of
fractured crystalline rock at the bedretto underground laboratory
for geosciences and geoenergies", International Journal of Rock Mechanics
and Mining Sciences, 176, 105689.

The geometry is adapted from the figures provided in the paper. Coordinates
are in meters and the domain is in local coordinates with y=0 at the top
and pointing downwards.

"""

import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures


class BedrettoMB1_Geometry(CubeDomainOrthogonalFractures):
    def set_domain(self) -> None:
        
        bounding_box = {
            "xmin": self.units.convert_units(-4850, "m"),
            "xmax": self.units.convert_units(5150, "m"),
            "ymin": self.units.convert_units(-1500, "m"),
            "ymax": self.units.convert_units(-9000, "m"),
        }
        self._domain = pp.Domain(bounding_box)

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        cell_size = self.params.get("cell_size", 1000)
        cell_size_fracture = self.params.get("cell_size_fracture", 500)
        mesh_args["cell_size"] = self.units.convert_units(cell_size, "m")
        mesh_args["cell_size_fracture"] = self.units.convert_units(
            cell_size_fracture, "m"
        )
        return mesh_args

    def grid_type(self) -> str:
        return "simplex"

    def set_fractures(self) -> None:

        # Add fractures one by one in local coordinates - need to subtact the height
        depth = self._domain.bounding_box["ymin"]
        # Use local coordinate system with y=0 at the top and pointing downwards
        fractures_end_points_local_coords = [
            # Fracture 1
            ((47, 113), (74, 89)),
            ((74, 89), (94, 63)),
            # Fracture 2
            ((89, 118), (107, 89)),
            ((107, 89), (140, 0)),
            # Fracture 3
            ((97, 128), (108, 103)),
            ((108, 103), (128, 50)),
            # Fracture 4
            ((38, 196), (59, 167)),
            ((59, 167), (91, 100)),
            # Fracture 5
            ((75, 137), (90, 123)),
            # Fracture 6
            ((68, 155), (91, 127)),
            # Fracture 7
            ((37, 218), (58, 179)),
            ((58, 179), (92, 131)),
            # Fracture 8
            ((54, 210), (61, 195)),
            ((61, 195), (74, 180)),
            ((74, 180), (82, 161)),
            ((82, 161), (93, 131)),
            # Fracture 9
            ((62, 198), (81, 179)),
            ((81, 179), (99, 160)),
            # Fracture 10
            ((29, 289), (48, 234)),
            ((48, 234), (62, 213)),
            ((62, 213), (86, 188)),
            ((86, 188), (93, 175)),
            # Fracture 11
            ((47, 312), (57, 274)),
            ((57, 274), (88, 218)),
            # Fracture 12
            ((78, 245), (94, 225)),
            ((94, 225), (110, 188)),
            ((110, 188), (118, 175)),
            # Fracture 13
            ((106, 222), (113, 209)),
            ((113, 209), (138, 178)),
            # Fracture 14 - stimulated fracture!
            ((59, 287), (91, 244)),
            ((91, 244), (113, 222)),
            ((113, 222), (118, 218)),
            # Fracture 15
            ((62, 291), (74, 277)),
            ((74, 277), (93, 252)),
            # Fracture 16
            ((63, 307), (69, 301)),
            # Fracture 17
            ((58, 327), (68, 317)),
            ((68, 317), (97, 256)),
            ((97, 256), (102, 250)),
            # Fracture 18
            ((63, 327), (72, 318)),
            # Fracture 19
            ((60, 368), (127, 292)),
            # Fracture 20
            ((59, 407), (77, 357)),
            # Fracture 21
            ((100, 398), (150, 347)),
            # Fracture 22
            ((94, 354), (98, 348)),
            ((98, 348), (149, 299)),
            # Fracture 23
            ((83, 336), (92, 301)),
            ((92, 301), (111,240)),
            ((111,240), (128, 208)),
            # Fracture 24
            ((132,261), (114, 246)),
            ((114, 246), (103, 229)),
        ]
        # Convert to global coordinates
        fractures_end_points = []
        for pt in fractures_end_points_local_coords:
            x, y = pt[0]
            x_end, y_end = pt[1]
            fractures_end_points.append(np.array([[x, depth-y], [x_end, depth-y_end]]))
        fractures = [pp.LineFracture(pts.T) for pts in fractures_end_points]

        # TODO: Add boreholes - need to be treated differently than fractures in the modeling - can we use open wells?
        boreholes_end_points_local = []
        boreholes_end_points = []
        for pt in fractures_end_points_local_coords:
            x, y = pt[0]
            x_end, y_end = pt[1]
            boreholes_end_points.append(np.array([[x, depth-y], [x_end, depth-y_end]]))
        boreholes = [pp.LineFracture(pts.T) for pts in boreholes_end_points]
        
        # Allow to steer number of fractures
        num_fractures = self.params.get("num_fractures", len(fractures))
        num_boreholes = self.params.get("num_boreholes", len(boreholes))
        self._fractures = fractures[:num_fractures] + boreholes[:num_boreholes]
