from collections import OrderedDict
import pandas as pd
import numpy as np
import os

from pytest import fixture
# geopy is to test great circle distance calculation
from geopy.distance import geodesic

from test_solver_scipy import solve_atmospheric_entry_test
from test_solver_scipy import calculate_energy_test

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly


@fixture(scope='module')
def deepimpact():
    import deepimpact
    return deepimpact


@fixture(scope='module')
def planet(deepimpact):
    return deepimpact.Planet()


@fixture(scope='module')
def loc(deepimpact):
    return deepimpact.GeospatialLocator()


@fixture(scope='module')
def result(planet):
    input = {'radius': 1.,
             'velocity': 2.0e4,
             'density': 3000.,
             'strength': 1e5,
             'angle': 30.0,
             'init_altitude': 0.0,
             }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope='module')
def outcome(planet, result):
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_import(deepimpact):
    assert deepimpact


def test_planet_init_default(planet):
    assert planet.Cd == 1.0
    assert planet.Ch == 0.1
    assert planet.Q == 1e7
    assert planet.Cl == 1e-3
    assert planet.alpha == 0.3
    assert planet.Rp == 6371e3
    assert planet.g == 9.81
    assert planet.H == 8000.0
    assert planet.rho0 == 1.2
    assert planet.rhoa(0) == 1.2


def test_planet_init_exponential(deepimpact):
    planet = deepimpact.Planet(atmos_func="exponential")
    assert planet.rhoa(0) == 1.2
    assert np.isclose(planet.rhoa(8000), 1.2 * np.exp(-8000/planet.H))


def test_planet_init_constant(deepimpact):
    planet = deepimpact.Planet(atmos_func="constant")
    assert planet.rhoa(0) == 1.2
    assert planet.rhoa(8000) == 1.2


def test_planet_init_tabular(deepimpact):
    planet = deepimpact.Planet(atmos_func="tabular")
    print("333", planet.rhoa(8000))
    print("444", 1.2 * np.exp(-8000/planet.H))
    assert np.isclose(planet.rhoa(8000), 0.525168)


def test_planet_signature(deepimpact):
    inputs = OrderedDict(atmos_func='exponential',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    _ = deepimpact.Planet(**inputs)

    # call by position
    _ = deepimpact.Planet(*inputs.values())


def test_attributes(planet):
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)


def test_atmos_filename(planet):

    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    assert type(energy) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns


def test_analyse_outcome(outcome):

    assert type(outcome) is dict

    for key in ('outcome', 'burst_peak_dedz', 'burst_altitude',
                'burst_distance', 'burst_energy'):
        assert key in outcome.keys()


def test_scenario(planet):

    inputs = {'radius': 35.,
              'angle': 45.,
              'strength': 1e7,
              'density': 3000.,
              'velocity': 19e3}

    result = planet.solve_atmospheric_entry(**inputs)

    # For now, we just check the result is a DataFrame
    # and the columns are as expected.

    # You should add more tests here to check the values
    # are correct and match the expected output
    # given in the tests/scenario.npz file

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_solve_atmospheric_entry_chao(planet):

    atol = 1e-3

    test_cases = [
        {
            'radius': 35., 'velocity': 19e3, 'density': 3000., 'strength': 1e7,
            'angle': 45., 'init_altitude': 100e3, 'dt': 0.25, 'radians': False
        },
        {
            'radius': 15., 'velocity': 19e4, 'density': 300., 'strength': 1e5,
            'angle': 30., 'init_altitude': 100e3, 'dt': 0.25, 'radians': False
        },
        # {
        #     'radius': 75., 'velocity': 19e5, 'density': 30000., 'strength': 1e9,
        #     'angle': 75., 'init_altitude': 100e3, 'dt': 0.05, 'radians': False
        # },
    ]

    common_columns = ['velocity', 'mass',
                      'angle', 'altitude', 'distance', 'radius']

    for params in test_cases:
        # Call the first function
        result_t = solve_atmospheric_entry_test(**params)
        # Call the second function
        result = planet.solve_atmospheric_entry(**params)

        merged_data = pd.merge(result_t[common_columns].add_suffix(
            '_t'), result[common_columns].add_suffix('_planet'), left_index=True, right_index=True)
        correlation_matrix = merged_data.corr()

        correlation_coefficient_velocity = correlation_matrix['velocity_t']['velocity_planet']
        correlation_coefficient_mass = correlation_matrix['mass_t']['mass_planet']
        correlation_coefficient_angle = correlation_matrix['angle_t']['angle_planet']
        correlation_coefficient_altitude = correlation_matrix['altitude_t']['altitude_planet']
        correlation_coefficient_distance = correlation_matrix['distance_t']['distance_planet']
        correlation_coefficient_radius = correlation_matrix['radius_t']['radius_planet']

        assert np.isclose(correlation_coefficient_velocity, 1, atol=atol)
        assert np.isclose(correlation_coefficient_mass, 1, atol=atol)
        assert np.isclose(correlation_coefficient_angle, 1, atol=atol)
        assert np.isclose(correlation_coefficient_altitude, 1, atol=atol)
        assert np.isclose(correlation_coefficient_distance, 1, atol=atol)
        assert np.isclose(correlation_coefficient_radius, 1, atol=atol)


def test_calculate_energy_chao(planet, result):
    energy = planet.calculate_energy(result=result)

    assert type(energy) is pd.DataFrame

    for key in (
        "velocity",
        "mass",
        "angle",
        "altitude",
        "distance",
        "radius",
        "time",
        "dedz",
    ):
        assert key in energy.columns

    atol = 1e-7
    atol_dedz = 1e-2

    test_cases = [
        {
            'radius': 35., 'velocity': 19e3, 'density': 3000., 'strength': 1e7,
            'angle': 45., 'init_altitude': 100e3, 'dt': 0.25, 'radians': False
        },
        # {
        #     'radius': 15., 'velocity': 19e4, 'density': 300., 'strength': 1e5,
        #     'angle': 30., 'init_altitude': 100e3, 'dt': 0.25, 'radians': False
        # },
        # {
        #     'radius': 75., 'velocity': 19e5, 'density': 30000., 'strength': 1e9,
        #     'angle': 75., 'init_altitude': 100e3, 'dt': 0.05, 'radians': False
        # },
    ]

    common_columns = ['velocity', 'mass', 'angle',
                      'altitude', 'distance', 'radius', 'dedz']

    for params in test_cases:
        # Call the first function
        result_t = solve_atmospheric_entry_test(**params)
        # Call the second function
        result_e = planet.solve_atmospheric_entry(**params)

        result_t = calculate_energy_test(result_t)
        # Call the second function
        result = planet.calculate_energy(result_e)

        merged_data = pd.merge(result_t[common_columns].add_suffix(
            '_t'), result[common_columns].add_suffix('_planet'), left_index=True, right_index=True)
        correlation_matrix = merged_data.corr()

        correlation_coefficient_velocity = correlation_matrix['velocity_t']['velocity_planet']
        correlation_coefficient_mass = correlation_matrix['mass_t']['mass_planet']
        correlation_coefficient_angle = correlation_matrix['angle_t']['angle_planet']
        correlation_coefficient_altitude = correlation_matrix['altitude_t']['altitude_planet']
        correlation_coefficient_distance = correlation_matrix['distance_t']['distance_planet']
        correlation_coefficient_radius = correlation_matrix['radius_t']['radius_planet']
        correlation_coefficient_dedz = correlation_matrix['dedz_t']['dedz_planet']

        assert np.isclose(correlation_coefficient_velocity, 1, atol=atol)
        assert np.isclose(correlation_coefficient_mass, 1, atol=atol)
        assert np.isclose(correlation_coefficient_angle, 1, atol=atol)
        assert np.isclose(correlation_coefficient_altitude, 1, atol=atol)
        assert np.isclose(correlation_coefficient_distance, 1, atol=atol)
        assert np.isclose(correlation_coefficient_radius, 1, atol=atol)
        assert np.isclose(correlation_coefficient_dedz, 1, atol=atol_dedz)


def test_damage_zones(deepimpact):

    outcome = {'burst_peak_dedz': 1000.,
               'burst_altitude': 9000.,
               'burst_distance': 90000.,
               'burst_energy': 6000.,
               'outcome': 'Airburst'}

    blat, blon, damrad = deepimpact.damage_zones(outcome, 55.0, 0.,
                                                 135., [27e3, 43e3])

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 2


def test_great_circle_distance(deepimpact):

    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array([[1.28580537e+05, 2.59579735e+05, 2.25409117e+02],
                    [0.00000000e+00, 2.24656571e+05, 1.28581437e+05],
                    [2.72529953e+05, 2.08175028e+05, 1.96640630e+05]])

    dist = deepimpact.great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)

    pnts3 = np.array([[51.516444, -0.12913], [52.18347, -0.45143]])
    pnts4 = np.array([[51.47053, -0.45143], [51.516444, -0.12913]])

    # Use geopy to calculate the distance between two points
    expected = []
    for coord1 in pnts3:
        row = []
        for coord2 in pnts4:
            distance = geodesic(coord1, coord2).kilometers * 1000
            row.append(distance)
        expected.append(row)

    dist_test = deepimpact.great_circle_distance(pnts3, pnts4)
    assert np.allclose(expected, dist_test, rtol=1e3)


def test_locator_postcodes(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [0.2e3, 0.1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list

    # Test cases
    latlon2 = (51.4981, -0.1773)  # South Kensington
    result_test = loc.get_postcodes_by_radius(latlon2, [10, None, 50, 150])
    result_expect = [[], [], ['SW7 2AZ'], ['SW7 2AZ', 'SW7 2BS', 'SW7 2BT', 'SW7 2BU',
                                           'SW7 2DD', 'SW7 5HE', 'SW7 5HF', 'SW7 5HG', 'SW7 5HQ']]
    assert result_test == result_expect

    latlon3 = (51.50326, -0.11999)  # London Eye
    result_test = loc.get_postcodes_by_radius(
        latlon3, [1e2, None, 1.5e2, 3.5e2])
    result_expect = [['SE1 7PB'], [], ['SE1 7JA', 'SE1 7PB'], ['SE1 7AF', 'SE1 7BQ', 'SE1 7DH', 'SE1 7GA', 'SE1 7GB',
                                                               'SE1 7GD', 'SE1 7GE', 'SE1 7GF', 'SE1 7GH', 'SE1 7GL',
                                                               'SE1 7GN', 'SE1 7GP', 'SE1 7GQ', 'SE1 7GT', 'SE1 7GU',
                                                               'SE1 7GY', 'SE1 7JA', 'SE1 7NA', 'SE1 7ND', 'SE1 7NJ',
                                                               'SE1 7NN', 'SE1 7NQ', 'SE1 7NZ', 'SE1 7PB', 'SE1 7PD',
                                                               'SE1 7PE', 'SE1 7PJ', 'SE1 7PN', 'SE1 7PY', 'SE1 8XU',
                                                               'SW1A2HR', 'SW1A2JF', 'SW1A2JH', 'SW1A2JL', 'SW1A2TT']]
    assert result_test == result_expect


def test_population_by_radius(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_population_by_radius(latlon, [5e2, 1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is int


# Test the method='basic' option, which is the default
def test_population_by_radius_basic(loc):
    latlon = (51.4981, -0.1773)
    result_test = loc.get_population_by_radius(latlon, [1e2, None, 5e2, 1e3])
    result_expect = [0, 0, 7412, 27794]
    assert result_test == result_expect
    assert type(result_test) is list
    if len(result_test) > 0:
        for element in result_test:
            assert type(element) is int

    latlon2 = (51.50326, -0.11999)  # London Eye
    result_test2 = loc.get_population_by_radius(latlon2, [1e2, None, 5e2, 1e3])
    result_expect2 = [0, 0, 2626, 4749]
    assert result_test2 == result_expect2


# # Test the method='scaling' option
# def test_population_by_radius_scaling(loc):
#     latlon = (51.4981, -0.1773)  # South Kensington
#     result_test = loc.get_population_by_radius(
#         latlon, [1e2, None, 5e2, 1e3], method='scaling')
#     result_expect = [1482, 0, 7412, 27794]
#     assert result_test == result_expect
#     assert type(result_test) is list
#     if len(result_test) > 0:
#         for element in result_test:
#             assert type(element) is int

#     latlon2 = (51.50326, -0.11999)  # London Eye
#     result_test2 = loc.get_population_by_radius(
#         latlon2, [1e2, None, 5e2, 1e3], method='scaling')
#     result_expect2 = [525, 0, 2626, 4749]
#     assert result_test2 == result_expect2


# # Test the method='subdivide' option, it's accurate but slow
# def test_population_by_radius_subdivide(loc):
#     latlon = (51.4981, -0.1773)  # South Kensington
#     result_test = loc.get_population_by_radius(
#         latlon, [1e2, None, 5e2, 1e3], method='subdivide')
#     result_expect = [0, 0, 3706, 31360]
#     assert result_test == result_expect
#     assert type(result_test) is list
#     if len(result_test) > 0:
#         for element in result_test:
#             assert type(element) is int

#     latlon2 = (51.50326, -0.11999)  # London Eye
#     result_test2 = loc.get_population_by_radius(
#         latlon2, [1e2, None, 5e2, 1e3], method='subdivide')
#     result_expect2 = [0, 0, 3693, 17333]
#     assert result_test2 == result_expect2


def test_calculate_longitude_difference(deepimpact):
    lat = 0
    distance_meters = 100000
    result_test = deepimpact._calculate_longitude_difference(
        lat, distance_meters)
    assert type(result_test) is float

    # Use another method to calculate the longitude difference
    result_expect = distance_meters / \
        111194.92664455873 / np.cos(lat * np.pi / 180)
    assert np.isclose(result_test, result_expect)

    lat = 50
    distance_meters = 500
    result_test = deepimpact._calculate_longitude_difference(
        lat, distance_meters)
    assert type(result_test) is float
    result_expect = distance_meters / \
        111194.92664455873 / np.cos(lat * np.pi / 180)
    assert np.isclose(result_test, result_expect)


def test_impact_risk(deepimpact, planet):

    probability, population = deepimpact.impact_risk(planet)

    assert type(probability) is pd.DataFrame
    assert 'probability' in probability.columns
    assert type(population) is dict
    assert 'mean' in population.keys()
    assert 'stdev' in population.keys()
    assert type(population['mean']) is float
    assert type(population['stdev']) is float
