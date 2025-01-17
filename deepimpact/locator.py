"""Module dealing with postcode information."""

import math
import os

import numpy as np
import pandas as pd

__all__ = ['GeospatialLocator', 'great_circle_distance',
           '_calculate_longitude_difference']


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [[55, 1.0]]))
    [[128580.53670808]
     [ 63778.24657475]]

    """
    # Convert to radians
    latlon1 = np.radians(latlon1)
    latlon2 = np.radians(latlon2)

    # Radius of the Earth in kilometres
    R = 6371.0
    # Differences in latitudes and longitudes
    dlat = latlon2[:, 0] - latlon1[:, 0, np.newaxis]
    dlon = latlon2[:, 1] - latlon1[:, 1, np.newaxis]
    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(latlon1[:, 0, np.newaxis]) * \
        np.cos(latlon2[:, 0]) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance = R * c * 1000.0
    return distance


def _calculate_longitude_difference(lat, distance_meters):
    """
    Calculate the change in longitude (in degrees) corresponding
    to a given distance (in meters) along the given latitude.

    Parameters
    ----------

    lat : float
        Latitude of the location in degrees.
    distance_meters : float
        Distance along the given latitude in meters.

    Returns
    -------

    float
        Change in longitude in degrees.

    Examples
    --------

    >>> result = _calculate_longitude_difference(0, 100000) # doctest: +SKIP

    """
    R = 6371000.0
    lat_radians = math.radians(lat)
    lon_difference_radians = distance_meters / (R * math.cos(lat_radians))
    lon_difference = math.degrees(lon_difference_radians)
    return lon_difference


class GeospatialLocator(object):
    """
    Class to interact with a postcode database file and a population grid file.
    """

    def __init__(self, postcode_file=os.sep.join((os.path.dirname(__file__), "..", "resources", "full_postcodes.csv")),
                 census_file=os.sep.join((os.path.dirname(__file__), "..", "resources",
                                          "UK_residential_population_2011_latlon.asc")),
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .asc file containing census data on a
            latitude-longitude grid.

        norm : function
            Python function defining the distance between points in
            latitude-longitude space.

        """
        self.postcode_df = pd.read_csv(postcode_file)
        self.census_file = census_file
        self.norm = norm

    def get_postcodes_by_radius(self, X, radii):
        """
        Return postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements
            of radii to the location X.

        Examples
        --------
        >>> locator = GeospatialLocator()
        >>> result = locator.get_postcodes_by_radius((51.4981, -0.1773), [1.5e3])
        >>> print(result[0][1])
        SW100AE
        >>> result = locator.get_postcodes_by_radius((51.4981, -0.1773), [1.5e3, 4.0e3])
        >>> print(result[0][1])
        SW100AE
        """
        center_latlon = np.array([X])

        distances = self.norm(center_latlon, self.postcode_df[[
            'Latitude', 'Longitude']].values)
        result = []

        for radius in radii:
            if radius is None or radius == 0:
                result.append([])
                continue
            bool_list = (distances <= radius)[0]
            true_positions = [index for index,
                              value in enumerate(bool_list) if value]
            selected_postcodes = self.postcode_df.loc[true_positions, 'Postcode'].tolist(
            )
            result.append(selected_postcodes)

        return result

    def get_population_by_radius(self, X, radii, method='basic'):
        """
        Return the population within specific distances of input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X

        Returns
        -------
        list of integers
            Contains the population closer than the elements of radii to the location X.
            Output should be the same shape as the radii array.

        Examples
        --------
        >>> loc = GeospatialLocator()
        >>> loc.get_population_by_radius((51.4981, -0.1773), [1e2, 5e2, 1e3])
        [0, 7412, 27794]
        """
        # Load the census data, read the acsii file
        with open(self.census_file, 'r') as f:
            header = {line.split()[0]: line.split()[1]
                      for line in f for _ in range(3)}
        # Extract the no_value, nrows, ncols
        nrows = int(header['nrows'])  # 1211
        no_value = int(header['NODATA_value'])  # -9999
        data = np.loadtxt(self.census_file, skiprows=6)
        data = pd.DataFrame(data)
        data[data == no_value] = 0  # Replace -9999 with 0
        # Get the seperate lat, lon, population array
        data = pd.DataFrame({
            'Latitude': data.iloc[:nrows, :].values.flatten(),
            'Longitude': data.iloc[nrows:nrows * 2, :].values.flatten(),
            'Population': data.iloc[nrows * 2:, :].values.flatten()
        })
        population_radii = []

        # The default method
        if method == 'basic':
            for radius in radii:
                if radius is None or radius == 0:
                    population_radii.append(0)
                    continue
                distances = self.norm(
                    np.array([X]), data[['Latitude', 'Longitude']].values)
                bool_list = (distances <= radius)[0]
                true_positions = [index for index,
                                  value in enumerate(bool_list) if value]
                selected_population = data.loc[true_positions, 'Population'].tolist(
                )
                population_radii.append(int(sum(selected_population)))

        # To better calculate population with small radius
        elif method == 'scaling':
            for radius in radii:
                if radius is None or radius == 0:
                    population_radii.append(0)
                    continue
                distances = self.norm(
                    np.array([X]), data[['Latitude', 'Longitude']].values)
                if radius < 500:
                    radius_plus = 500
                    bool_list = (distances <= radius_plus)[0]
                    true_positions = [index for index,
                                      value in enumerate(bool_list) if value]
                    selected_population = data.loc[true_positions, 'Population'].tolist(
                    )
                    population_radii.append(
                        int(sum(selected_population) * radius / radius_plus))
                else:
                    distances = self.norm(
                        np.array([X]), data[['Latitude', 'Longitude']].values)
                    bool_list = (distances <= radius)[0]
                    true_positions = [index for index,
                                      value in enumerate(bool_list) if value]
                    selected_population = data.loc[true_positions, 'Population'].tolist(
                    )
                    population_radii.append(int(sum(selected_population)))

        # Divide the grids into smaller size
        elif method == 'subdivide':
            subdivided_grids = []
            n_subdivisions = 10

            for _, row in data.iterrows():
                center_lat, center_lon, population = row['Latitude'], row['Longitude'], row['Population']

                size = 1000.0
                subdivision_size = size / n_subdivisions
                R = 6371000.0
                distance_radians = (size / 2 - subdivision_size / 2) / R

                for i in range(n_subdivisions):
                    for j in range(n_subdivisions):
                        # Calculate latlon of new gird center
                        subgrid_center_lat = center_lat - math.degrees(distance_radians) + math.degrees((
                            subdivision_size * i) / R)
                        subgrid_center_lon = center_lon - _calculate_longitude_difference(
                            subgrid_center_lat, size / 2 - subdivision_size / 2)
                        +_calculate_longitude_difference(subgrid_center_lat, subdivision_size * j)
                        # Average population
                        subgrid_population = population / (n_subdivisions ** 2)
                        # Create a new dataframe
                        subdivided_grids.append(
                            (subgrid_center_lat, subgrid_center_lon, subgrid_population))

            subdivided_df = pd.DataFrame(subdivided_grids, columns=[
                'Latitude', 'Longitude', 'Population'])

            for radius in radii:
                if radius is None or radius == 0:
                    population_radii.append(0)
                    continue
                distances = self.norm(
                    np.array([X]), subdivided_df[['Latitude', 'Longitude']].values)
                bool_list = (distances <= radius)[0]
                true_positions = [index for index,
                                  value in enumerate(bool_list) if value]
                selected_population = subdivided_df.loc[true_positions, 'Population'].tolist(
                )
                population_radii.append(int(sum(selected_population)))

        return population_radii
