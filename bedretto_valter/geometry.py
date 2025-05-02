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


disk_radius = 50  # NOTE: 20 results in not-connected fractures


class BedrettoValter_Geometry(CubeDomainOrthogonalFractures):
    def set_domain(self) -> None:
        bounding_box = {
            "xmin": self.units.convert_units(-350, "m"),
            "xmax": self.units.convert_units(350, "m"),
            "ymin": self.units.convert_units(-350, "m"),
            "ymax": self.units.convert_units(350, "m"),
            "zmin": self.units.convert_units(-1450, "m"),
            "zmax": self.units.convert_units(-750, "m"),
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

    def tunnel(self, tm) -> np.ndarray:
        tunnel_orientation = 43 / 180 * np.pi  # N 43 deg W
        return np.array([0, 0, -1000]) + (tm - 2050) * np.array(
            [np.cos(-tunnel_orientation), np.sin(-tunnel_orientation), 0]
        )

    def cb1(self, md) -> np.ndarray:
        """TM: tunnel depth in m. (x,y)=(0,0) corresponds to the lab."""
        tm = 2050  # m
        azimuth_orientation = 133 / 180 * np.pi  # N 133 deg W
        inclination = 45 / 180 * np.pi  # 45 deg
        # Set cb1 to the center of the tunnel, and tunnel at -1000 m depth
        xyz = self.tunnel(tm) - md * np.array(
            [
                np.cos(-azimuth_orientation) * np.sin(-inclination),
                np.sin(-azimuth_orientation) * np.sin(-inclination),
                np.cos(-inclination),
            ]
        )
        return self.units.convert_units(xyz, "m")

    def cb2(self, md) -> np.ndarray:
        tm = 2043  # m
        azimuth_orientation = 133 / 180 * np.pi  # N 133 deg W
        inclination = 40 / 180 * np.pi  # 45 deg
        # Set cb1 to the center of the tunnel, and tunnel at -1000 m depth
        xyz = self.tunnel(tm) - md * np.array(
            [
                np.cos(-azimuth_orientation) * np.sin(-inclination),
                np.sin(-azimuth_orientation) * np.sin(-inclination),
                np.cos(-inclination),
            ]
        )
        return self.units.convert_units(xyz, "m")

    def set_fractures(self) -> None:

        interval = {}
        center = {}
        transmissivity = {}

        # Interval 14
        interval[14] = [
            pp.create_elliptic_fracture(
                center=self.cb1(90),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=237 * np.pi / 180,  # E-W
                dip_angle=63 * np.pi / 180,
                num_points=16,
                # index=0,
            )
        ]
        center[14] = [self.cb1(90)]
        transmissivity[14] = [2.3e-9]

        # Interval 13
        interval[13] = [
            pp.create_elliptic_fracture(
                center=self.cb1(110),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=240 * np.pi / 180,  # E-W
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        center[13] = [self.cb1(110)]
        transmissivity[13] = [8.4e-7]

        # Interval 12
        interval[12] = [
            # NOTE: The interesting one!
            pp.create_elliptic_fracture(
                center=self.cb1(131),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=231 * np.pi / 180,  # E-W
                dip_angle=50 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(133),
            #    major_axis=self.units.convert_units(disk_radius, "m"),
            #    minor_axis=self.units.convert_units(disk_radius, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle= 241 * np.pi / 180,  # E-W
            #    dip_angle=65 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(135),
            #    major_axis=self.units.convert_units(disk_radius, "m"),
            #    minor_axis=self.units.convert_units(disk_radius, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle= 251 * np.pi / 180,  # E-W
            #    dip_angle=80 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
        ]
        center[12] = [
            self.cb1(131),
            #            self.cb1(133),
            #            self.cb1(135),
        ]
        transmissivity[12] = [1.9e-8]

        # Interval 11
        interval[11] = [
            pp.create_elliptic_fracture(
                center=self.cb1(140),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=233 * np.pi / 180,  # E-W
                dip_angle=58 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        center[11] = [self.cb1(140)]
        transmissivity[11] = [5.6e-8]

        # Interval 10
        interval[10] = [
            pp.create_elliptic_fracture(
                center=self.cb1(165),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,
                strike_angle=231 * np.pi / 180,  # E-W
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        center[10] = [self.cb1(165)]
        transmissivity[10] = [1.8e-8]

        # Interval 9
        # TODO many more?
        interval[9] = [
            pp.create_elliptic_fracture(
                center=self.cb1(185),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,
                strike_angle=227 * np.pi / 180,  # E-W
                dip_angle=59 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        center[9] = [self.cb1(185)]
        transmissivity[9] = [4.1e-8]

        # Interval 8
        interval[8] = [
            pp.create_elliptic_fracture(
                center=self.cb1(208),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=231 * np.pi / 180,  # N-S
                dip_angle=69 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
        ]
        center[8] = [self.cb1(208)]
        transmissivity[8] = [3.3e-8]

        # Interval 7
        interval[7] = [
            pp.create_elliptic_fracture(
                center=self.cb1(219),
                major_axis=self.units.convert_units(disk_radius, "m"),
                minor_axis=self.units.convert_units(disk_radius, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=240 * np.pi / 180,  # N-S
                dip_angle=60 * np.pi / 180,
                num_points=16,
                # index=0,
            ),
            # pp.create_elliptic_fracture(
            #    center=self.cb1(221),
            #    major_axis=self.units.convert_units(disk_radius, "m"),
            #    minor_axis=self.units.convert_units(disk_radius, "m"),
            #    major_axis_angle=0,  # TODO?
            #    strike_angle=240 * np.pi / 180,  # N-S
            #    dip_angle=60 * np.pi / 180,
            #    num_points=16,
            #    # index=0,
            # ),
        ]
        center[7] = [self.cb1(219)]
        transmissivity[7] = [3e-7]

        active_intervals = {
            14: False,
            13: True,
            12: True,
            11: True,
            10: True,
            9: True,
            8: True,
            7: True,
        }
        pressurized_intervals = {
            14: False,
            13: False,# True
            12: False,
            11: False,
            10: False,
            9: False,
            8: False, # True
            7: False,
        }
        self._fractures = []
        mask_pressurized = []
        self.fracture_centers = []
        self.fracture_transmissivity = []
        for i, is_active in active_intervals.items():
            if not is_active:
                continue
            self._fractures += interval[i]
            self.fracture_transmissivity += transmissivity[i]
            if pressurized_intervals[i]:
                self.fracture_centers += center[i]
            mask_pressurized += len(interval[i]) * [pressurized_intervals[i]]

        # Convert to numpy arrays for later use
        self.fracture_centers = np.array(self.fracture_centers)

        # Local numbering of pressurized fractures
        self.pressurized_fractures = np.where(
            mask_pressurized
        )[0]

    # This lower part is only used in order to construct vtu files containing the tunnel and the wells
    #    self._fractures = []

    # def set_well_network(self) -> None:
    #    """Assign well network class."""
    #    well_coords = [
    #        #np.vstack((
    #        #    self.cb1(0),
    #        #    self.cb1(303)
    #        #)).transpose(),
    #        #np.vstack((
    #        #    self.cb2(0),
    #        #    self.cb2(220)
    #        #)).transpose(),
    #        np.vstack((
    #            self.tunnel(2000),
    #            self.tunnel(2100),
    #        )).transpose()
    #    ]
    #    wells = [pp.Well(wc) for wc in well_coords]
    #    self.well_network = pp.WellNetwork3d(domain=self._domain, wells=wells, parameters={"mesh_size": self.params["cell_size"]})
