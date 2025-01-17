import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepimpact
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def create_cubic_interpolation_function(x_observed, y_observed):
    """
    Create a cubic interpolation function based on observed data.

    Parameters
    ----------
    x_observed : array-like
        The x-coordinates of the observed data points.
    y_observed : array-like
        The y-coordinates of the observed data points.

    Returns
    -------
    function
        A function that takes x values and returns interpolated y values.
    """
    # Check if x and y arrays have the same length
    if len(x_observed) != len(y_observed):
        raise ValueError("x_observed and y_observed must have the same length.")

    # Create the cubic interpolation function
    interp_func = interp1d(x_observed[::3], y_observed[::3], kind='quadratic', fill_value='extrapolate')

    return interp_func


class Chelyabinsk(object):
    def __init__(self, data_path='./resources/ChelyabinskEnergyAltitude.csv'):
        """
        Initialize the Chelyabinsk class instance.

        Parameters
        ----------
        data_path : str
            The path to the CSV file containing Chelyabinsk meteor data.
        """
        # Initialize the class with data from a CSV file
        data = pd.read_csv(data_path)
        data.columns = ['Height_km', 'Energy_kt_km']
        self.data = data
        # Create an interpolation function based on the data
        self.f = create_cubic_interpolation_function(data.Height_km, data.Energy_kt_km)
        # Define minimum and maximum altitudes from the data
        self.min = data.Height_km.min()
        self.max = data.Height_km.max()

    def solve_ode(self, radius=55, strength=5e7, dt=0.01, planet=deepimpact.Planet()):
        """
        Solve the ordinary differential equations (ODE) for the Chelyabinsk meteor.

        Parameters
        ----------
        radius : float
            Radius of the meteoroid.
        strength : float
            Strength of the meteoroid.
        dt : float
            Time step for the ODE solver.
        planet : deepimpact.Planet
            The Planet object for the ODE solver.

        Returns
        -------
        tuple
            A tuple containing arrays of altitude (km) and energy (kt/km).
        """
        # Solve the ODE given the parameters and return the altitude and energy
        velocity = 19.2e3  # Velocity in meters per second
        density = 3300     # Density in kg/m^3
        angle = 18.3       # Angle in degrees
        # Run the ODE solver with the specified parameters
        result = planet.solve_atmospheric_entry(
            radius=radius,
            velocity=velocity,
            density=density,
            strength=strength,
            angle=angle,
            dt=dt
        )
        # Calculate the energy at different altitudes
        result_with_energy = planet.calculate_energy(result)
        # Convert altitude to km
        result_with_energy.altitude = result_with_energy.altitude * 1e-3
        return result_with_energy.altitude, result_with_energy.dedz

    def rmse_loss(self, params):
        """
        Calculate the Root Mean Square Error (RMSE) loss between model output and interpolation function.

        Parameters
        ----------
        params: list or array-like
            Parameters to be optimized, e.g., [radius, strength].

        Returns
        -------
        float
            The calculated RMSE between the model output and the interpolation function.
        """
        # Extract radius and strength from params
        radiu, strength = params
        # Solve ODE with given parameters
        height_km, energy_kt_km = self.solve_ode(radiu, strength)
        # Filter output based on altitude range
        valid_idx = (height_km >= self.min) & (height_km <= self.max)
        x_model = height_km[valid_idx]
        y_model = energy_kt_km[valid_idx]
        # Compute interpolated values for comparison
        y_f = self.f(x_model)
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_model - y_f) ** 2))
        return rmse

    def norm2_loss(self, params):
        """
        Calculate the 2-norm loss between the highest energy point from the model output and the observed data.

        Parameters
        ----------
        params: list or array-like
            Parameters to be optimized, e.g., [radius, strength].

        Returns
        -------
        float
            The calculated 2-norm loss.
        """
        # Calculate model predictions
        radius, strength = params
        height_km, energy_kt_km = self.solve_ode(radius, strength)
        # Find the highest energy point in the model predictions
        max_energy_idx = np.argmax(energy_kt_km)
        max_energy_model = (height_km[max_energy_idx], energy_kt_km[max_energy_idx])
        # Find the highest energy point in the observed data
        max_energy_observed = self.data.loc[self.data.Energy_kt_km.idxmax(), ['Height_km', 'Energy_kt_km']].values
        # Calculate the 2-norm (Euclidean distance) between the two points
        loss = np.linalg.norm(np.array(max_energy_model) - np.array(max_energy_observed))
        return loss

    def optimize(self, func='norm2'):
        """
        Optimize the parameters of the meteoroid model to minimize the chosen loss function.

        Parameters
        ----------
        func : str
            The loss function to be used in optimization, 'norm2' or 'rmse'.
        """
        # Optimization function to minimize the loss
        if func == 'norm2':
            error_function = self.norm2_loss
        elif func == 'rmse':
            error_function = self.rmse_loss
        else:
            print('Please use norm2 or rmse')
            pass

        initial_guess = (8., 5e6)
        self.result = minimize(error_function, initial_guess, method='BFGS')
        return self.result

    def test(self):
        """
        Test the optimized parameters and plot the model's prediction against the observed data.
        """
        # Test the optimized parameters and plot the results
        x, y = self.solve_ode(self.result.x[0], self.result.x[1])
        fig, axs = plt.subplots(1, 1)

        # Plot model predictions
        axs.plot(x, y, label='Model Prediction', color='blue')

        # Plot observed data
        axs.plot(self.data.Height_km, self.data.Energy_kt_km, label='Observed Data', color='red', linestyle='--')

        # Add title and axis labels
        plt.title(f'Model Output vs Observed Data (RMSE Loss: {self.rmse_loss(self.result.x):.2f})')
        axs.set_xlabel('Height (km)')
        axs.set_ylabel('Energy (kt/km)')

        # Display the legend
        axs.legend()

        # Add grid lines for better readability
        axs.grid(True)

        # Calculate and print RMSE
        mse = self.rmse_loss(self.result.x)
        print("RMSE:", mse)
