"""Collection of different injection profiles."""

import porepy as pp
import numpy as np
from icecream import ic


class PressureConstraintWell:
    """Pressurize specific fractures in their center."""

    def update_time_dependent_ad_arrays(self) -> None:
        """Set current injection pressure."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        pressure_schedule = self.schedule
        current_injection_overpressure = np.interp(
            self.time_manager.time,
            [t for t, _ in pressure_schedule],
            [p for _, p in pressure_schedule],
            left=0.0,
        )
        ic(self.time_manager.time, current_injection_overpressure)

        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_overpressure",
                values=np.array(
                    [self.units.convert_units(current_injection_overpressure, "Pa")]
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Fetch standard mass balance equation
        std_eq = super().mass_balance_equation(subdomains)

        # Pick fractures
        fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]
        if len(fracture_sds) == 0:
            return std_eq

        # Identify pressurized fractures
        sd_pressurized = self.mdg.subdomains(dim=self.nd - 1)[self.injection_local_fracture_index]

        # Define indicator for injection cell
        sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]
        for i, sd in enumerate(subdomains):
            if sd == sd_pressurized:
                injection_cell = sd.closest_cell(self.injection_coordinate.reshape((-1,1)))
                sd_indicator[i][injection_cell] = 1.
        indicator = np.concatenate(sd_indicator)
        reverse_indicator = 1.0 - indicator

        # Fetch pressure values and fix the pressure constraint
        current_injection_overpressure = pp.ad.TimeDependentDenseArray(
            "current_injection_overpressure", [self.mdg.subdomains()[0]]
        )
        hydrostatic_pressure = pp.ad.TimeDependentDenseArray(
            "hydrostatic_pressure", subdomains
        )
        constrained_eq = (
            self.pressure(subdomains)
            - current_injection_overpressure
            - hydrostatic_pressure
        )

        # Create the equation with the pressure constraint through convex combination
        eq_with_pressure_constraint = (
            pp.ad.DenseArray(reverse_indicator) * std_eq
            + pp.ad.DenseArray(indicator) * constrained_eq
        )
        eq_with_pressure_constraint.set_name(std_eq.name)

        return eq_with_pressure_constraint


class FlowConstraintWell:
    """Flow constraint for the injection well."""

    def update_time_dependent_ad_arrays(self) -> None:
        """Set current flow rate."""
        super().update_time_dependent_ad_arrays()

        # Update injection pressure
        rate_schedule = self.schedule
        current_injection_rate = np.interp(
            self.time_manager.time,
            [t for t, _ in rate_schedule],
            [r for _, r in rate_schedule],
            left=0.0,
        )
        ic(self.time_manager.time, current_injection_rate)

        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="current_injection_rate",
                values=np.array(
                    [self.units.convert_units(current_injection_rate, "m^3/s")]
                ),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )

    def fluid_source(self, subdomains):
        std_fluid_source = super().fluid_source(subdomains)

        # Pick fractures
        fracture_sds = [sd for sd in subdomains if sd.dim == self.nd - 1]
        if len(fracture_sds) == 0:
            return std_fluid_source

        # Identify pressurized fractures
        sd_pressurized = self.mdg.subdomains(dim=self.nd - 1)[self.injection_local_fracture_index]

        # Define indicator for injection cell
        sd_indicator = [np.zeros(sd.num_cells) for sd in subdomains]
        for i, sd in enumerate(subdomains):
            if sd == sd_pressurized:
                injection_cell = sd.closest_cell(self.injection_coordinate.reshape((-1,1)))
                sd_indicator[i][injection_cell] = 1
        indicator = np.concatenate(sd_indicator)

        # Fetch flow rate values and fix the flow constraint
        current_injection_rate = pp.ad.TimeDependentDenseArray(
            "current_injection_rate", [self.mdg.subdomains()[0]]
        )

        fluid_source = (
            std_fluid_source + pp.ad.DenseArray(indicator) * current_injection_rate
        )
        fluid_source.set_name(std_fluid_source.name)

        return fluid_source


class InjectionInterval8(PressureConstraintWell):
    """Extracted from Broeker et al, 2024, Hydromechanical characterization of a
    fractured crystalline rock volume during multi-stage hydraulic stimulations
    at the BedrettoLab. Fig 4a.

    """

    @property
    def injection_local_fracture_index(self):
        return self.interval_to_local_fracture_index[8]

    @property
    def injection_coordinate(self):
        """Defined in geometry.py"""
        return self.fracture_center[8][0]

    @property
    def is_pressure_controlled_injection(self):
        """Check if the injection is pressure controlled."""
        return False

    @property
    def is_rate_controlled_injection(self):
        """Check if the injection is rate controlled."""
        return True

    @property
    def rate_schedule(self):
        """Return the empty rate schedule."""
        return None

    @property
    def _initialization_schedule(self):
        return [(0 * pp.HOUR, 0), (1 * pp.HOUR, 0)]

    @property
    def _pressure_schedule(self):
        return [
            (0 * pp.HOUR, 0),
            (0.02 * pp.HOUR, 2 * pp.MEGA),
            (0.2 * pp.HOUR, 2 * pp.MEGA),
            (0.27 * pp.HOUR, 7.2 * pp.MEGA),
            (0.32 * pp.HOUR, 6.8 * pp.MEGA),
            (0.41 * pp.HOUR, 5.8 * pp.MEGA),
            (0.75 * pp.HOUR, 5.5 * pp.MEGA),
            (0.77 * pp.HOUR, 12 * pp.MEGA),
            (0.85 * pp.HOUR, 11 * pp.MEGA),
            (0.90 * pp.HOUR, 10 * pp.MEGA),
            (1.16 * pp.HOUR, 10 * pp.MEGA),
            (1.23 * pp.HOUR, 8.6 * pp.MEGA),
            (1.24 * pp.HOUR, -2.6 * pp.MEGA),
            (1.74 * pp.HOUR, -2.6 * pp.MEGA),
            (1.77 * pp.HOUR, -1.4 * pp.MEGA),
            (1.87 * pp.HOUR, 0 * pp.MEGA),
            (2.18 * pp.HOUR, 0.5 * pp.MEGA),
            (2.19 * pp.HOUR, 2.2 * pp.MEGA),
            (2.29 * pp.HOUR, 2.1 * pp.MEGA),
            (2.30 * pp.HOUR, 3.8 * pp.MEGA),
            (2.40 * pp.HOUR, 4 * pp.MEGA),
            (2.43 * pp.HOUR, 5 * pp.MEGA),
            (2.51 * pp.HOUR, 4.6 * pp.MEGA),
            (2.55 * pp.HOUR, 6 * pp.MEGA),
            (2.68 * pp.HOUR, 5.4 * pp.MEGA),
            (2.73 * pp.HOUR, 6.5 * pp.MEGA),
            (2.85 * pp.HOUR, 6.7 * pp.MEGA),
            (2.87 * pp.HOUR, 8 * pp.MEGA),
            (3.02 * pp.HOUR, 8 * pp.MEGA),
            (3.03 * pp.HOUR, 8.6 * pp.MEGA),
            (3.08 * pp.HOUR, 8.2 * pp.MEGA),
            (3.23 * pp.HOUR, 8.1 * pp.MEGA),
            (3.25 * pp.HOUR, 9.9 * pp.MEGA),
            (3.29 * pp.HOUR, 8.8 * pp.MEGA),
            (3.48 * pp.HOUR, 8.8 * pp.MEGA),
            (3.50 * pp.HOUR, 10.2 * pp.MEGA),
            (3.55 * pp.HOUR, 9.9 * pp.MEGA),
            (3.79 * pp.HOUR, 10.8 * pp.MEGA),
            (4.07 * pp.HOUR, 10.5 * pp.MEGA),
            (4.30 * pp.HOUR, 7.1 * pp.MEGA),
            (4.57 * pp.HOUR, 5 * pp.MEGA),
        ]

    @property
    def schedule(self):
        # Fetch the initialization and pressure schedules
        _initialization_schedule = self._initialization_schedule
        _pressure_schedule = self._pressure_schedule

        # Merge the initialization schedule with the pressure schedule
        offset = _initialization_schedule[-1][0]
        pressure_schedule = _initialization_schedule + [
            (t + offset, v) for t, v in _pressure_schedule
        ]

        # Remove duplicate times
        pressure_schedule = sorted(set(pressure_schedule), key=lambda x: x[0])
        return pressure_schedule


class InjectionInterval9(PressureConstraintWell):
    """Adapted from Vaezi et al - not true data?"""

    @property
    def injection_local_fracture_index(self):
        print(self.interval_to_local_fracture_index[9])
        return self.interval_to_local_fracture_index[9]

    @property
    def injection_coordinate(self):
        """Defined in geometry.py"""
        print(self.fracture_center[9][0])
        return self.fracture_center[9][0]

    @property
    def is_pressure_controlled_injection(self):
        """Check if the injection is pressure controlled."""
        return False

    @property
    def is_rate_controlled_injection(self):
        """Check if the injection is rate controlled."""
        return True

    @property
    def _initialization_schedule(self):
        return [(0 * pp.HOUR, 0), (1 * pp.HOUR, 0)]

    @property
    def _pressure_schedule(self):
        return [
            (0 * pp.HOUR, 0),
            (0.05 * pp.HOUR, 2 * pp.MEGA),
            (0.2 * pp.HOUR, 2 * pp.MEGA),
            (0.25 * pp.HOUR, 4 * pp.MEGA),
            (0.4 * pp.HOUR, 4 * pp.MEGA),
            (0.45 * pp.HOUR, 6 * pp.MEGA),
            (0.6 * pp.HOUR, 6 * pp.MEGA),
            (0.65 * pp.HOUR, 8 * pp.MEGA),
            (0.8 * pp.HOUR, 8 * pp.MEGA),
            (0.85 * pp.HOUR, 10 * pp.MEGA),
            (1.0 * pp.HOUR, 10 * pp.MEGA),
            (1.05 * pp.HOUR, 12 * pp.MEGA),
            (1.2 * pp.HOUR, 12 * pp.MEGA),
            (1.25 * pp.HOUR, 14 * pp.MEGA),
            (1.4 * pp.HOUR, 14 * pp.MEGA),
            (1.45 * pp.HOUR, 12 * pp.MEGA),
            (1.6 * pp.HOUR, 12 * pp.MEGA),
            (1.65 * pp.HOUR, 14 * pp.MEGA),
            (1.8 * pp.HOUR, 14 * pp.MEGA),
            (1.85 * pp.HOUR, 16 * pp.MEGA),
            (2.0 * pp.HOUR, 16 * pp.MEGA),
            (2.05 * pp.HOUR, 20 * pp.MEGA),
            (2.2 * pp.HOUR, 20 * pp.MEGA),
            (2.25 * pp.HOUR, 16 * pp.MEGA),
            (2.4 * pp.HOUR, 8 * pp.MEGA),
            (2.45 * pp.HOUR, 20 * pp.MEGA),
            (2.6 * pp.HOUR, 20 * pp.MEGA),
            (2.65 * pp.HOUR, 16 * pp.MEGA),
            (2.8 * pp.HOUR, 10 * pp.MEGA),
            (2.85 * pp.HOUR, 20 * pp.MEGA),
            (3.0 * pp.HOUR, 20 * pp.MEGA),
            (3.05 * pp.HOUR, 16 * pp.MEGA),
            (3.2 * pp.HOUR, 11 * pp.MEGA),
            (3.25 * pp.HOUR, 20 * pp.MEGA),
            (3.4 * pp.HOUR, 20 * pp.MEGA),
            (3.45 * pp.HOUR, 16 * pp.MEGA),
            (3.6 * pp.HOUR, 12 * pp.MEGA),
            (3.65 * pp.HOUR, 20 * pp.MEGA),
            (3.8 * pp.HOUR, 20 * pp.MEGA),
            (3.85 * pp.HOUR, 16 * pp.MEGA),
            (4.0 * pp.HOUR, 12 * pp.MEGA),
            (4.05 * pp.HOUR, 20 * pp.MEGA),
            (4.2 * pp.HOUR, 20 * pp.MEGA),
            (4.25 * pp.HOUR, 16 * pp.MEGA),
            (4.4 * pp.HOUR, 12 * pp.MEGA),
            (4.45 * pp.HOUR, 20 * pp.MEGA),
            (4.6 * pp.HOUR, 20 * pp.MEGA),
            (4.65 * pp.HOUR, 16 * pp.MEGA),
            (4.8 * pp.HOUR, 12 * pp.MEGA),
            (4.85 * pp.HOUR, 20 * pp.MEGA),
            (5.0 * pp.HOUR, 20 * pp.MEGA),
            (5.05 * pp.HOUR, 16 * pp.MEGA),
            (5.2 * pp.HOUR, 12 * pp.MEGA),
            (5.25 * pp.HOUR, 20 * pp.MEGA),
            (5.4 * pp.HOUR, 20 * pp.MEGA),
            (5.45 * pp.HOUR, 16 * pp.MEGA),
            (5.6 * pp.HOUR, 12 * pp.MEGA),
            (5.65 * pp.HOUR, 20 * pp.MEGA),
            (5.8 * pp.HOUR, 20 * pp.MEGA),
            (5.85 * pp.HOUR, 16 * pp.MEGA),
            (6.0 * pp.HOUR, 12 * pp.MEGA),
            (6.05 * pp.HOUR, 20 * pp.MEGA),
            (6.2 * pp.HOUR, 20 * pp.MEGA),
            (6.25 * pp.HOUR, 16 * pp.MEGA),
            (6.4 * pp.HOUR, 12 * pp.MEGA),
            (6.45 * pp.HOUR, 20 * pp.MEGA),
            (6.6 * pp.HOUR, 20 * pp.MEGA),
            (6.65 * pp.HOUR, 16 * pp.MEGA),
            (6.8 * pp.HOUR, 12 * pp.MEGA),
            (6.85 * pp.HOUR, 20 * pp.MEGA),
            (7.0 * pp.HOUR, 20 * pp.MEGA),
            (7.05 * pp.HOUR, 16 * pp.MEGA),
            (7.2 * pp.HOUR, 12 * pp.MEGA),
            (7.25 * pp.HOUR, 20 * pp.MEGA),
            (7.4 * pp.HOUR, 20 * pp.MEGA),
            (7.45 * pp.HOUR, 16 * pp.MEGA),
            (7.6 * pp.HOUR, 12 * pp.MEGA),
            (7.65 * pp.HOUR, 20 * pp.MEGA),
            (7.8 * pp.HOUR, 20 * pp.MEGA),
            (7.85 * pp.HOUR, 16 * pp.MEGA),
            (8.0 * pp.HOUR, 12 * pp.MEGA),
        ]

    @property
    def schedule(self):
        # Fetch the initialization and pressure schedules
        _initialization_schedule = self._initialization_schedule
        _pressure_schedule = self._pressure_schedule

        # Merge the initialization schedule with the pressure schedule
        offset = _initialization_schedule[-1][0]
        pressure_schedule = _initialization_schedule + [
            (t + offset, v) for t, v in _pressure_schedule
        ]

        # Remove duplicate times
        pressure_schedule = sorted(set(pressure_schedule), key=lambda x: x[0])
        return pressure_schedule


class InjectionInterval13(FlowConstraintWell):
    """Adapted from Repolles et al, 2024, Modeling coupled hydro-mechanical
    processes during hydraulic stimulation at the Bedretto Underground
    Laboratory, see Fig. 1 in the paper.

    See also Broeker et al, 2024, Hydromechanical characterization of a
    fractured crystalline rock volume during multi-stage hydraulic
    stimulations at the BedrettoLab. Fig 4b.

    """

    @property
    def injection_local_fracture_index(self):
        return self.interval_to_local_fracture_index[13]

    @property
    def injection_coordinate(self):
        """Defined in geometry.py"""
        return self.fracture_center[13][0]

    @property
    def is_pressure_controlled_injection(self):
        """Check if the injection is pressure controlled."""
        return False

    @property
    def is_rate_controlled_injection(self):
        """Check if the injection is rate controlled."""
        return True

    @property
    def _initialization_schedule(self):
        return [(0 * pp.HOUR, 0), (1 * pp.HOUR, 0)]

    @property
    def _rate_schedule(self):
        # Conversion factor from l/min to m^3/s
        conversion_factor_volume = 1e-3 * 60
        # Injection schedule in l/min
        schedule = [
            (0 * pp.HOUR, 0.0),
            (0.8 * pp.HOUR, 0.0),
            (0.86 * pp.HOUR, 23),
            (0.96 * pp.HOUR, 23),
            (1.05 * pp.HOUR, 33),
            (1.2 * pp.HOUR, 33),
            (1.23 * pp.HOUR, 50),
            (1.4 * pp.HOUR, 50),
            (1.46 * pp.HOUR, 70),
            (1.57 * pp.HOUR, 70),
            (1.64 * pp.HOUR, 75),
            (1.68 * pp.HOUR, 95),
            (1.88 * pp.HOUR, 95),
            (1.93 * pp.HOUR, 110),
            (2.10 * pp.HOUR, 110),
            (2.15 * pp.HOUR, 130),
            (2.25 * pp.HOUR, 130),
            (2.26 * pp.HOUR, 0),
            (2.92 * pp.HOUR, 0),
            (2.95 * pp.HOUR - 167,),
            (3.01 * pp.HOUR, -172),
            (3.03 * pp.HOUR, -124),
            (3.09 * pp.HOUR, -97),
            (3.20 * pp.HOUR, -69),
            (3.36 * pp.HOUR, -52),
            (3.62 * pp.HOUR, -43),
            (4.02 * pp.HOUR, -40),
            (4.07 * pp.HOUR, 0),
            (4.88 * pp.HOUR, 0),
            (4.93 * pp.HOUR, 28),
            (5.08 * pp.HOUR, 32),
            (5.14 * pp.HOUR, 48),
            (5.27 * pp.HOUR, 52),
            (5.30 * pp.HOUR, 69),
            (5.45 * pp.HOUR, 71),
            (5.49 * pp.HOUR, 87),
            (5.61 * pp.HOUR, 87),
            (5.71 * pp.HOUR, 112),
            (5.80 * pp.HOUR, 115),
            (5.87 * pp.HOUR, 131),
            (6.03 * pp.HOUR, 136),
            (6.10 * pp.HOUR, 151),
            (6.17 * pp.HOUR, 151),
            (6.19 * pp.HOUR, 0),
            (10.0 * pp.HOUR, 0),
        ]
        # Convert to m^3/s
        schedule = [(t, v * conversion_factor_volume) for t, v in schedule]
        return schedule

    @property
    def schedule(self):
        """Return the rate schedule."""

        # Fetch the initialization and rate schedules
        _initialization_schedule = self._initialization_schedule
        _rate_schedule = self._rate_schedule

        # Merge the initialization schedule with the rate schedule
        offset = _initialization_schedule[-1][0]
        rate_schedule = _initialization_schedule + [
            (t + offset, v) for t, v in _rate_schedule
        ]

        # Remove duplicate times
        rate_schedule = sorted(set(rate_schedule), key=lambda x: x[0])

        return rate_schedule
