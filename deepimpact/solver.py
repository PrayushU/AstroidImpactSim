"""
This module contains the atmospheric entry solver class
for the Deep Impact project
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

__all__ = ["Planet"]


class Planet:
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(
            self,
            atmos_func="exponential",
            atmos_filename=os.sep.join(
                (os.path.dirname(__file__), "..", "resources",
                 "AltitudeDensityTable.csv")
            ),
            Cd=1.0,
            Ch=0.1,
            Q=1e7,
            Cl=1e-3,
            alpha=0.3,
            Rp=6371e3,
            g=9.81,
            H=8000.0,
            rho0=1.2,
    ):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'

        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option

        Cd : float, optional
            The drag coefficient

        Ch : float, optional
            The heat transfer coefficient

        Q : float, optional
            The heat of ablation (J/kg)

        Cl : float, optional
            Lift coefficient

        alpha : float, optional
            Dispersion coefficient

        Rp : float, optional
            Planet radius (m)

        rho0 : float, optional
            Air density at zero altitude (kg/m^3)

        g : float, optional
            Surface gravity (m/s^2)

        H : float, optional
            Atmospheric scale height (m)

        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        self.burst_alt = None
        self.break_up_alt = None
        self.airburst_energy = None
        self.break_up_stop_alt = None
        self.time_series = None
        self.impact_event = None
        self.start_alt = None

        try:
            # set function to define atmoshperic density
            if atmos_func == "exponential":
                self.rhoa = lambda z: self.rho0 * np.exp((-z) / self.H)
            elif atmos_func == "tabular":
                try:
                    df = pd.read_csv(atmos_filename,
                                     sep=r'\s+',
                                     skiprows=1,
                                     names=['Altitude', 'Atmospheric_density'])
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Unable to find file: {atmos_filename}")

                self.rhoa = interp1d(df['Altitude'],
                                     df['Atmospheric_density'],
                                     kind='cubic',
                                     bounds_error=False,
                                     fill_value="extrapolate")
            elif atmos_func == "constant":
                self.rhoa = lambda x: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda x: rho0

    def solve_atmospheric_entry(
            self,
            radius,
            velocity,
            density,
            strength,
            angle,
            init_altitude=100e3,
            dt=0.05,
            radians=False,
    ):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters

        velocity : float
            The entery speed of the asteroid in meters/second

        density : float
            The density of the asteroid in kg/m^3

        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2

        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians

        init_altitude : float, optional
            Initial altitude in m

        dt : float, optional
            The output timestep, in s

        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input

        Returns
        -------
        result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """
        theta = None
        if not radians:
            theta = self._degrees_to_radian(angle)
        elif radians:
            theta = angle

        values = {
            "radius": radius,
            "velocity": velocity,
            "rhom": density,
            "Y": strength,
            "angle": theta,
            "init_altitude": init_altitude,
            "dt": dt,
        }

        volume = (4 / 3) * np.pi * (values["radius"] ** 3)
        mass = volume * density
        initial_condition = [velocity, mass, theta, init_altitude, 0, radius]
        self.start_alt = init_altitude

        if dt <= 0.05:
            time_step = dt
        elif 0.05 < dt <= 10:
            time_step = 0.005
        elif dt > 10:
            time_step = dt / 1000

        self.time_series = self._rk4(
            initial_condition, time_step, dt, self._equations, values
        )
        return self.time_series

    def _degrees_to_radian(self, angle):
        """
        Convert an angle from degrees to radians.

        Parameters
        ----------
        angle : numeric
            The angle in degrees to be converted to radians.

        Returns
        -------
        radian_angle : numeric
            The angle converted to radians. The result is in the range [0, 2*pi).

        Examples
        --------
        >>> instance = Planet()  # Replace YourClassName with the actual class name
        >>> angle_in_degrees = 45
        >>> converted_angle = instance._degrees_to_radian(angle_in_degrees)
        >>> print(converted_angle)
        0.7853981633974483

        """
        return (np.pi / 180) * (angle % 360)

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        ---------
        result : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude

        """

        if (len(result) == 1):
            result["dedz"] = 0
            return result

        kinetic_energy = 0.5 * result["mass"] * (result["velocity"] ** 2)

        kinetic_energy_conv = kinetic_energy * 2.39006e-13

        de = np.diff(kinetic_energy_conv, prepend=0)

        dz = np.diff(result["altitude"], prepend=0) / 1000

        np.seterr(divide='ignore')
        result["dedz"] = de / dz

        result["dedz"][0] = result["dedz"][1]

        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats.

        Parameters
        -------------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time

        Returns
        ---------
        result : Dict
            dictionary with details of the impact event, which should contain the key:
            ``outcome`` (which should contain one of the
            following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
            ``burst_peak_dedz``, ``burst_altitude``,
            ``burst_distance``, ``burst_energy``

        """

        burst_peak_dedz = result["dedz"].max()
        burst_idx = result["dedz"].idxmax()  # Index of the burst moment

        # Determine if it's an airburst or cratering

        if result.at[burst_idx, "altitude"] > 0:
            outcome_type = "Airburst"
            burst_altitude = result.at[burst_idx, "altitude"]
            burst_distance = result.at[burst_idx, "distance"]
            burst_energy = abs(0.5 * result.at[burst_idx, "mass"] *
                               (result.at[burst_idx, "velocity"] ** 2) -
                               0.5 * result.at[0, "mass"] *
                               (result.at[0, "velocity"] ** 2))
        else:
            outcome_type = "Cratering"
            burst_altitude = 0
            burst_distance = result["distance"].iloc[-1]  # Distance at impact

            burst_en_first_inst = 0.5 * result.at[burst_idx, "mass"] * (
                result.at[burst_idx, "velocity"] ** 2)

            burst_en_sec_inst = abs(0.5 * result.at[burst_idx, "mass"] *
                                    (result.at[burst_idx, "velocity"] ** 2) -
                                    0.5 * result.at[0, "mass"] *
                                    (result.at[0, "velocity"] ** 2))

            burst_energy = max(burst_en_first_inst, burst_en_sec_inst)

        # Converting burst energy to kilotons of TNT equivalent
        burst_energy_kilotons = burst_energy * 2.39006e-13

        outcome = {
            "outcome": outcome_type,
            "burst_peak_dedz": burst_peak_dedz,
            "burst_altitude": burst_altitude,
            "burst_distance": burst_distance,
            "burst_energy": burst_energy_kilotons
        }
        return outcome

    def _rk4(self, initial_conditions, h, dt, equations, values_):
        """
        Runge-Kutta 4th order method for solving Ordinary Differential Equations (ODEs).

        Parameters
        ----------
        initial_conditions : array-like
            Initial values for the system of ODEs.

        h : float
            Step size for the RK4 method.

        dt : float
            Output timestep for recording results.

        equations : callable
            A function representing the system of ODEs to be solved.
            Should have the signature equations(t, values, constants),
            where t is the current time, values is an array of current
            state variables, and constants are any additional parameters.

        values_ : any
            Additional constant values required by the ODEs.

        Returns
        -------
        result : DataFrame
            A pandas DataFrame containing the solution to the system of ODEs.
            Columns include 'velocity', 'mass', 'angle', 'altitude', 'distance',
            'radius', and 'time'.

        """
        n = int((10000) / h) + 1  # Number of time steps
        t = np.linspace(0, 10000, n)
        new_values = np.zeros((n, len(initial_conditions)))
        new_values[0] = initial_conditions
        len_n = len(new_values)
        result_rows = []
        i = 0
        max_iter = 0
        while True:

            if max_iter == 1000000:
                break
            if i == len_n:
                new_values = np.concatenate(
                    (new_values, np.zeros((10000,
                                           len(initial_conditions))))
                )
                len_n = len(new_values)
                t = np.concatenate((t,
                                    np.linspace(i, 20000, int(10000 / dt))))
            if i == 0:
                new_values[i] = initial_conditions
            else:
                k1 = h * np.array(
                    equations(t[i - 1], new_values[i - 1], values_))
                k2 = h * np.array(
                    equations(t[i - 1] + h / 2,
                              new_values[i - 1] + k1 / 2, values_)
                )
                k3 = h * np.array(
                    equations(t[i - 1] + h / 2,
                              new_values[i - 1] + k2 / 2, values_)
                )
                k4 = h * np.array(
                    equations(t[i - 1] + h, new_values[i - 1] + k3, values_)
                )
                new_values[i] = new_values[i - 1] + (
                    k1 + 2 * k2 + 2 * k3 + k4) / 6

            if i % int(dt / h) == 0 or i == 0:
                result_row = {
                    "velocity": new_values[i, 0],
                    "mass": new_values[i, 1],
                    "angle": np.degrees(new_values[i, 2]),
                    "altitude": new_values[i, 3],
                    "distance": new_values[i, 4],
                    "radius": new_values[i, 5],
                    "time": t[i],
                }

                result_rows.append(result_row)

                if self._end_cases(new_values[i]):
                    break

            max_iter += 1
            i += 1
        return pd.DataFrame(result_rows)

    def _end_cases(self, current_values):
        if current_values[1] <= 0:
            self.is_negative = "mass"
        elif current_values[3] <= 0:
            self.is_negative = "altitude"
        elif current_values[3] > self.start_alt:
            self.is_negative = 'altitude'
        elif current_values[5] <= 0:
            self.is_negative = "radius"
        else:
            self.is_negative = "None"

        if self.is_negative != "None":
            return True
        else:
            return False

    def _equations(self, t, initial_condition, values_):
        """
        Define the system of Ordinary Differential Equations (ODEs) for the physical system.

        Parameters
        ----------
        t : float
            Current time.

        initial_condition : array-like
            Initial conditions for the system of ODEs, including velocity (v),
            mass (m), angle (theta), altitude (z_), horizontal position (x),
            and radius (r).

        values_ : dict
            Additional constant values required by the ODEs.

        Returns
        -------
        list
            List containing the derivatives of the state variables with respect to time (t).
            The order of derivatives corresponds to the order of variables in the 'initial_condition' array.

        """
    # Function implementation goes here

        # volume = 4/3 * np.pi * radius**3
        # mass = volume * rhom

        v, m, theta, z_, x, r = initial_condition
        A = np.pi * r ** 2

        rhoa_ = self.rhoa(z_)

        dvdt = ((-self.Cd * rhoa_ * A * v ** 2) / (2 * m)) + (
            self.g * np.sin(theta))

        dmdt = (-self.Ch * rhoa_ * A * v ** 3) / (2 * self.Q)

        dthetadt = (
            (self.g * np.cos(theta) / v)
            - (self.Cl * rhoa_ * A * v / (2 * m))
            - (v * np.cos(theta) / (self.Rp + z_))
        )

        dzdt = -v * np.sin(theta)

        dxdt = (v * np.cos(theta)) / (1 + z_ / self.Rp)

        if rhoa_ * v ** 2 > values_["Y"]:
            drdt = ((7 / 2 * self.alpha * rhoa_ / values_["rhom"]) ** 0.5) * v
        else:
            drdt = 0

        return [
            dvdt,
            dmdt,
            dthetadt,
            dzdt,
            dxdt,
            drdt,
        ]
