"""This module contains some useful mapping functions"""
import folium

__all__ = ['plot_circle']


# Helper function to get color based on pressure value


def _get_color(pressure):
    """
    Determine the color that corresponds to a given airblast pressure level.

    Parameters
    ----------
    pressure : float
        The airblast pressure in kilopascals (kPa).

    Returns
    -------
    str
        The color that represents the damage level based on the pressure.
        The color coding is as follows:
        - 'blue' for minor damage (1-4 kPa).
        - 'green' for moderate damage (4-30 kPa).
        - 'orange' for severe damage (30-50 kPa).
        - 'red' for extreme damage (>50 kPa).

    """
    if 4e3 > pressure >= 1e3:
        return 'blue'
    elif 30e3 > pressure >= 4e3:
        return 'green'
    elif 50e3 > pressure >= 30e3:
        return 'orange'
    else:
        return 'red'


def plot_circle(lat, lon, radius, pressures=None, fmap=None, zoom_start=8, **kwargs):
    """
    Plot a circle or multiple circles on a map, each representing airblast pressure levels,
    at specified latitude and longitude coordinates.

    If an existing map object is not provided, a new Folium map instance is created.
    The color of each circle is determined based on the airblast pressure level,
    indicating different levels of damage.

    Parameters
    ----------
    lat : float
        Latitude of the center of the circle(s) to plot, in degrees.
    lon : float
        Longitude of the center of the circle(s) to plot, in degrees.
    radius : float or list of floats
        Radius (in meters) of the circle(s) to be plotted. Can be a single value or a list of radii.
    pressures : float or list of floats, optional
        Airblast pressure levels (in kilopascals, kPa) corresponding to each radius.
        Determines the color of the circle(s).
    fmap : folium.Map, optional
        An existing Folium map object to plot on. If not provided, a new map is created.
    zoom_start : int, optional
        Initial zoom level for the map. Defaults to 8.
    **kwargs :
        Additional keyword arguments to pass to Folium's Circle method.

    Returns
    -------
    folium.Map
        The Folium map object with the plotted circles.

    Examples
    --------
    >>> import deepimpact
    >>> deepimpact.plot_circle(52.79, -2.95, 1000, pressures=1000) # doctest: +SKIP
    >>> deepimpact.plot_circle(52.79, -2.95, [1000, 2000], pressures=[1000, 3000],
      fmap=my_map, zoom_start=10) # doctest: +SKIP

    Notes
    -----
    - The function dynamically adjusts the color of each circle based on the provided airblast pressure,
      using the 'get_color' function.
    - If multiple radii and pressures are provided, they are paired and sorted in descending order of radius.
    - A legend is added to the map if airblast pressures are provided.
    """
    if fmap is None:
        fmap = folium.Map(location=[lat, lon],
                          zoom_start=zoom_start, control_scale=True)

    # Ensure radius and pressure are lists
    if not isinstance(radius, list):
        radius = [radius]
    if pressures is not None and not isinstance(pressures, list):
        pressures = [pressures]

    # Sort by radius in descending order
    if pressures is not None:
        # Ensure the tuples are in the form (radius, pressure)
        radius_pressures = [(radiu, pressure) for radiu, pressure in zip(
            radius, pressures) if radiu is not None]
        radius_pressures = sorted(
            radius_pressures, reverse=True, key=lambda x: x[0])
    else:
        # Sort radii in descending order and pair each with None
        radius_pressures = sorted(
            [(r, None) for r in radius if r is not None], reverse=True)

    for r, p in radius_pressures:
        # Assign color based on pressure, if provided
        color = _get_color(p) if p is not None else 'black'

        folium.Circle(
            location=[lat, lon],
            radius=r,
            color=color,
            fill=True,
            fillOpacity=0.2,
            **kwargs
        ).add_to(fmap)

    # Add a legend to the map

    if pressures is not None:
        # Add a legend to the map only if pressure is provided
        legend_html = '''
        <div style="position: fixed;
            top: 50px; right: 50px; width: 220px; height: 130px;
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 10px; box-shadow: 3px 3px 3px grey;">
            &nbsp; Damage Levels <br>
            &nbsp; <span style="background:blue; display:inline-block;
                width:10px; height:10px;"></span> 1 kPa - Minor Damage <br>
            &nbsp; <span style="background:green; display:inline-block;
                width:10px; height:10px;"></span> 4 kPa - Moderate Damage <br>
            &nbsp; <span style="background:orange; display:inline-block;
                width:10px; height:10px;"></span> 30 kPa - Severe Damage <br>
            &nbsp; <span style="background:red; display:inline-block;
                width:10px; height:10px;"></span> 50 kPa - Extreme Damage <br>
        </div>
        '''
        fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap
