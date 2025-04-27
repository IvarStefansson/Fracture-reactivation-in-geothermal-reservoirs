"""Implementation of three-dimensional Bedretto-like geometry inspired
from Bröker et al (2024), "Hydromechanical characterization of a fractured
crystalline rock volume during multi-stage hydraulic stimulations at the
BedrettoLab", Geothermics, 124, 103126

The geometry aims at describing the VALTER project (Validating Technologies
for Reservoir Engineering) at the BedrettoLab, Switzerland. It focuses on the
404 meter long stimulation borehole ST1 and in particular its intervals 7-14,
which extend over 206 meters

"""

import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures


class BedrettoValter_Geometry(CubeDomainOrthogonalFractures):
    def set_domain(self) -> None:
        bounding_box = {
            "xmin": self.units.convert_units(-500, "m"),
            "xmax": self.units.convert_units(500, "m"),
            "ymin": self.units.convert_units(-500, "m"),
            "ymax": self.units.convert_units(500, "m"),
            "zmin": self.units.convert_units(-1500, "m"),
            "zmax": self.units.convert_units(-500, "m"),
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

    def cb1(self, md) -> np.ndarray:
        """TM: tunnel depth in m. (x,y)=(0,0) corresponds to the lab."""
        xyz = np.array([0, 0, self.units.convert_units(-1000 - md, "m")])
        return xyz

    def set_fractures(self) -> None:
        # Interval 14
        interval_14 = [
            pp.create_elliptic_fracture(
                center=self.cb1(90),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=237 * np.pi / 180,  # E-W
                dip_angle=63 * np.pi / 180,
                num_points=16,
                # index=0,
            )
        ]
        pressurized_14 = len(interval_14) * [False]
        center_14 = [
            self.cb1(90)
        ]

        # Interval 13
        interval_13 = [
            pp.create_elliptic_fracture(
                center=self.cb1(110),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=240 * np.pi / 180,  # E-W
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        pressurized_13 = len(interval_13) * [False]
        center_13 = [
            self.cb1(110)
        ]

        # Interval 12
        interval_12 = [
            # NOTE: The interesting one!
            # pp.create_elliptic_fracture(
            #    center=self.cb1(131),
            #    major_axis=self.units.convert_units(50, "m"),
            #    minor_axis=self.units.convert_units(50, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle= 231 * np.pi / 180,  # E-W
            #    dip_angle=50 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(133),
            #    major_axis=self.units.convert_units(50, "m"),
            #    minor_axis=self.units.convert_units(50, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle= 241 * np.pi / 180,  # E-W
            #    dip_angle=65 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(135),
            #    major_axis=self.units.convert_units(50, "m"),
            #    minor_axis=self.units.convert_units(50, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle= 251 * np.pi / 180,  # E-W
            #    dip_angle=80 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
        ]
        pressurized_12 = len(interval_12) * [False]
        center_12 = [
#            self.cb1(131),
#            self.cb1(133),
#            self.cb1(135),
        ]

        # Interval 11
        interval_11 = [
            pp.create_elliptic_fracture(
                center=self.cb1(140),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=233 * np.pi / 180,  # E-W
                dip_angle=58 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        pressurized_11 = len(interval_11) * [False]
        center_11 = [
            self.cb1(140)
        ]

        # Interval 10
        interval_10 = [
            pp.create_elliptic_fracture(
                center=self.cb1(165),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=231 * np.pi / 180,  # E-W
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        pressurized_10 = len(interval_10) * [False]
        center_10 = [self.cb1(165)]

        # Interval 9
        # TODO many more?
        interval_9 = [
            pp.create_elliptic_fracture(
                center=self.cb1(185),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=227 * np.pi / 180,  # E-W
                dip_angle=59 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        pressurized_9 = len(interval_9) * [True]
        center_9 = [
            self.cb1(185)
        ]

        # Interval 8
        interval_8 = [
            pp.create_elliptic_fracture(
                center=self.cb1(208),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=231 * np.pi / 180,  # N-S
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        pressurized_8 = len(interval_8) * [False]
        center_8 = [
            self.cb1(208)
        ]

        # Interval 7
        interval_7 = [
            pp.create_elliptic_fracture(
                center=self.cb1(219),
                major_axis=self.units.convert_units(50, "m"),
                minor_axis=self.units.convert_units(50, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=240 * np.pi / 180,  # N-S
                dip_angle=60 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(221),
            #    major_axis=self.units.convert_units(50, "m"),
            #    minor_axis=self.units.convert_units(50, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle=240 * np.pi / 180,  # N-S
            #    dip_angle=60 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
        ]
        pressurized_7 = len(interval_7) * [False]
        center_7 = [
            self.cb1(219)
        ]

        self._fractures = (
            interval_14
            + interval_13
            + interval_12
            + interval_11
            + interval_10
            + interval_9
            + interval_8
            + interval_7
        )
        mask_pressurized = (
            pressurized_14
            + pressurized_13
            + pressurized_12
            + pressurized_11
            + pressurized_10
            + pressurized_9
            + pressurized_8
            + pressurized_7
        )
        self.pressurized_fractures = np.where(
            pressurized_14
            + pressurized_13
            + pressurized_12
            + pressurized_11
            + pressurized_10
            + pressurized_9
            + pressurized_8
            + pressurized_7
        )[0]
        self.fracture_centers = np.array(
            center_14
            + center_13
            + center_12
            + center_11
            + center_10
            + center_9
            + center_8
            + center_7
        )[np.array(mask_pressurized, dtype=bool)]
