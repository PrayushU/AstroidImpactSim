"""Module to calculate the damage and impact risk for given scenarios"""
import os
import folium

import numpy as np
import pandas as pd

from .locator import GeospatialLocator
from .mapping import plot_circle
from .solver import Planet

from folium.plugins import HeatMap

__all__ = ['damage_zones', 'impact_risk']


def _calculate_surface_zero(lat, lon, distance, bearing):
    """
    Calculate the surface zero latitude and longitude given the initial point,
    distance, and bearing.
    """
    R = 6371000  # Earth's radius in meters
    bearing = np.radians(bearing)  # Convert bearing to radians
    lat = np.radians(lat)  # Convert latitude to radians
    lon = np.radians(lon)  # Convert longitude to radians

    lat2 = np.arcsin(np.sin(lat) * np.cos(distance / R) +
                     np.cos(lat) * np.sin(distance / R) * np.cos(bearing))
    lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(distance / R) * np.cos(lat),
                            np.cos(distance / R) - np.sin(lat) * np.sin(lat2))

    lat2 = np.degrees(lat2)  # Convert latitude back to degrees
    lon2 = np.degrees(lon2)  # Convert longitude back to degrees

    lat2 = float(lat2)
    lon2 = float(lon2)
    return lat2, lon2


def _calculate_airblast_pressure(E_k, z_b, r):
    """
    Calculate the airblast pressure for given energy, burst altitude, and range.

    Parameters
    ----------
    E_k: float
        Explosion energy in kilotons of TNT equivalent.
    z_b: float
        Burst altitude in meters.
    r: float
        Horizontal range in meters.

    Returns
    -------
    float
        Airblast pressure in Pascals.
    """
    return (3 * 10 ** 11) * ((r ** 2 + z_b ** 2) / E_k ** (2 / 3)) ** -1.3 + \
        (2 * 10 ** 7) * ((r ** 2 + z_b ** 2) / E_k ** (2 / 3)) ** -0.57


def _calculate_damage_radii(E_k, z_b, pressures):
    """
    Calculate the damage radii for different pressure thresholds using binary search.

    Parameters
    ----------
    E_k: float
        Burst energy in kilotons of TNT equivalent.
    z_b: float
        Burst altitude in meters.
    pressures: list of float
        Threshold pressures in kilopascals (kPa).

    Returns
    -------
    list of float
        Radii for each pressure threshold.
    """
    radii = []
    # Half the circumference of the Earth in meters
    half_earth_circumference = 20037500
    for p in pressures:
        low, high = 0, half_earth_circumference  # Maximum possible radius
        tolerance = 0.1  # Tolerance in meters

        # Initial check to see if a solution may exist within the range
        if _calculate_airblast_pressure(E_k, z_b, low) > p:
            while high - low > tolerance:
                mid = (low + high) / 2
                current_pressure = _calculate_airblast_pressure(E_k, z_b, mid)

                if current_pressure < p:
                    high = mid
                else:
                    low = mid

            radii.append(mid)
        else:
            # No solution found within the range, append a default value or handle appropriately
            radii.append(None)

    return radii


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> import deepimpact
    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3, 'burst_distance': 90e3,
         'burst_peak_dedz': 1e3, 'outcome': 'Airburst'}
    >>> result = deepimpact.damage_zones(outcome, 52.79, -2.95,
        135, pressures=[1e3, 3.5e3, 27e3, 43e3])
    >>> print(result)
    (52.21396905216966, -2.015908861677074, [117474.63145293295,
    43146.37049101293, 9518.122905865312, 5729.415686801076])
    """

    # Extract the horizontal path length from the outcome dictionary
    if 'burst_distance' in outcome:
        path_length = outcome['burst_distance']
    else:
        raise KeyError(
            "The key 'burst_distance' is not present in the outcome dictionary.")

    # Calculate the surface zero location
    blat, blon = _calculate_surface_zero(lat, lon, path_length, bearing)

    damrad = _calculate_damage_radii(
        outcome['burst_energy'], outcome['burst_altitude'], pressures)

    return blat, blon, damrad


def process_scenario(row, planet, pressure):
    """
    Process a single impact scenario.

    Parameters:
    -----------
    row : pd.Series
        A row from the impact scenarios DataFrame, containing impact parameters.
    planet : Planet
        An instance of the Planet class to solve the atmospheric entry.
    pressure : float
        Pressure at which to calculate the damage zone.

    Returns:
    --------
    tuple
        A tuple containing the latitude and longitude of the surface zero location
        and the list of damage radii.
    """
    result = planet.solve_atmospheric_entry(
        radius=row['radius'],
        velocity=row['velocity'],
        density=row['density'],
        strength=row['strength'],
        angle=row['angle']
    )
    result_with_energy = planet.calculate_energy(result)
    outcome = planet.analyse_outcome(result_with_energy)

    blat, blon, damage_radii = damage_zones(
        outcome, row['entry latitude'], row['entry longitude'], row['bearing'], pressures=[pressure])

    return blat, blon, damage_radii


def impact_risk(planet=Planet(),
                impact_file=os.sep.join((os.path.dirname(__file__),
                                         '..', 'resources',
                                         'impact_parameter_list.csv')),
                pressure=30.e3, nsamples=None):
    """
    Perform an uncertainty analysis to calculate the probability for
    each affected UK postcode and the total population affected.

    Parameters
    ----------
    planet: deepimpact.Planet instance
        The Planet instance from which to solve the atmospheric entry
    impact_file: str
        Filename of a .csv file containing the impact parameter list
        with columns for 'radius', 'angle', 'velocity', 'strength',
        'density', 'entry latitude', 'entry longitude', 'bearing'
    pressure: float
        A single pressure at which to calculate the damage zone for each impact
    nsamples: int or None
        The number of iterations to perform in the uncertainty analysis.
        If None, the full set of impact parameters provided in impact_file
        is used.

    Returns
    -------
    probability: DataFrame
        A pandas DataFrame with columns for postcode and the
        probability the postcode was inside the blast radius.
    population: dict
        A dictionary containing the mean and standard deviation of the
        population affected by the impact, with keys 'mean' and 'stdev'.
        Values are floats.
    """
    # Initialize a GeospatialLocator
    locator = GeospatialLocator()

    # Read impact scenarios from a CSV file
    impact_scenarios = pd.read_csv(impact_file)

    # Sample scenarios if the number of samples is specified
    if nsamples:
        impact_scenarios = impact_scenarios.sample(n=nsamples)

    # Initialize dictionaries and lists
    postcode_probabilities = {}
    total_populations = []

    # Process impact scenarios and create a Folium map
    processed_data = impact_scenarios.apply(
        lambda row: process_scenario(row, planet, pressure), axis=1)
    fmap = None

    # Initialize a list to collect all latitudes and longitudes
    latlon_points = []

    # count = 0
    # Loop through processed data
    for blat, blon, damage_radii in processed_data:
        # count += 1
        # print(f'Processing impact scenario {count}/{len(impact_scenarios)}')
        fmap = plot_circle(blat, blon, damage_radii, pressure, fmap, zoom_start=10)
        for radius in damage_radii:
            affected_postcodes = locator.get_postcodes_by_radius((blat, blon), [radius])[
                0]
            affected_population = locator.get_population_by_radius((blat, blon), [radius])[
                0]

            for postcode in affected_postcodes:
                postcode_probabilities[postcode] = postcode_probabilities.get(
                    postcode, 0) + 1
            total_populations.append(affected_population)

        # Add the latitude and longitude of the circle center to the list
        latlon_points.append([blat, blon])

    # Save the Folium map as an HTML file
    fmap.save('map/risk_map.html')

    # Create a heat map using the collected latitudes and longitudes
    heatmap = folium.Map(location=[blat, blon], zoom_start=10)
    HeatMap(latlon_points).add_to(heatmap)

    # Save the heat map as an HTML file
    heatmap.save('map/heatmap.html')

    # Normalize postcode probabilities
    for postcode in postcode_probabilities:
        postcode_probabilities[postcode] /= len(impact_scenarios)

    # Create a DataFrame and calculate population statistics
    probability_df = pd.DataFrame.from_dict(
        postcode_probabilities, orient='index', columns=['probability'])
    population_stats = {'mean': float(
        np.mean(total_populations)), 'stdev': float(np.std(total_populations))}

    # Return probability DataFrame and population statistics
    return probability_df, population_stats
