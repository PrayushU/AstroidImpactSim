import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import math


def breakup_ode_system(t, y, Cd, Ch, Q, Cl, alpha, Rp, g, H, rho0, density, strength):
    # Unpack the variables
    velocity, mass, angle, altitude, distance, radius = y

    # Equations
    A = math.pi * radius**2
    rhoa = rho0 * np.exp(-altitude / H)

    d_velocity_dt = (-Cd * rhoa * A * velocity**2) / \
        (2 * mass) + g * np.sin(angle)
    d_mass_dt = (-Ch * rhoa * A * velocity**3) / (2 * Q)
    d_angle_dt = (g * np.cos(angle)) / velocity - (Cl * rhoa * A * velocity) / \
        (2 * mass) - (velocity * np.cos(angle)) / (Rp + altitude)
    d_altitude_dt = -velocity * np.sin(angle)
    d_distance_dt = velocity * np.cos(angle) / (1 + altitude / Rp)
    if rhoa * velocity**2 > strength:
        d_radius_dt = math.sqrt(3.5 * alpha * rhoa / density) * velocity
    else:
        d_radius_dt = 0

    return [d_velocity_dt, d_mass_dt, d_angle_dt, d_altitude_dt, d_distance_dt, d_radius_dt]


def solve_atmospheric_entry_test(radius, velocity, density, strength, angle,
                                 init_altitude=100e3, dt=0.05, radians=False):
    # Constants
    Cd = 1.
    Ch = 0.1
    Q = 1e7
    Cl = 1e-3
    alpha = 0.3
    Rp = 6371e3
    g = 9.81
    H = 8000.
    rho0 = 1.2

    # Convert angle to radians if necessary
    # Convert angle to radians if necessary
    if not radians:
        angle = np.radians(angle)
    elif radians:
        angle = angle % (2 * np.pi)

    # Initial conditions for pre-breakup
    initial_mass = (4 / 3) * math.pi * radius**3 * density
    initial_conditions_pre = [velocity, initial_mass,
                              angle, init_altitude, 0, radius]
    # [velocity, mass, angle, altitude, distance, radius]

    # Time span
    t_span = (0, 100000)  # Example time span, adjust as needed
    t_eval = np.arange(0, 10000, dt)

    # Ground impact event function

    def ground_impact(t, y):
        return y[3]  # Altitude
    ground_impact.terminal = True
    ground_impact.direction = -1

    # Breakup condition function
    def breakup_condition(t, y):
        rhoa = rho0 * np.exp(-y[3] / H)
        return rhoa * y[0]**2 - strength
    breakup_condition.terminal = True

    # Filter out data where altitude is below 0
    breakup_solution = solve_ivp(
        lambda t, y: breakup_ode_system(
            t, y, Cd, Ch, Q, Cl, alpha, Rp, g, H, rho0, density, strength),
        t_span, initial_conditions_pre, t_eval=t_eval, events=[
            ground_impact], max_step=0.0001
    )
    final_solution_y = breakup_solution.y
    final_solution_t = breakup_solution.t

    valid_indices = final_solution_y[3] >= 0
    filtered_solution_y = final_solution_y[:, valid_indices]
    filtered_solution_t = final_solution_t[valid_indices]

    # Creating DataFrame from the filtered solution
    result = pd.DataFrame({
        'velocity': filtered_solution_y[0],
        'mass': filtered_solution_y[1],
        'angle': np.degrees(filtered_solution_y[2]) if not radians else filtered_solution_y[2],
        'altitude': filtered_solution_y[3],
        'distance': filtered_solution_y[4],
        'radius': filtered_solution_y[5],
        'time': filtered_solution_t
    })

    return result


def calculate_energy_test(result):
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
