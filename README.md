#  AstoridImpactSim

`AstroidImpactSim` is a comprehensive Python library for modeling and visualizing asteroid impacts on Earth. It's designed for researchers, educators, and enthusiasts in the field of astronomy and earth sciences. This package offers a unique blend of scientific rigor and user-friendly interfaces, making complex simulations accessible.

## Modules

- **`solver.py`**: Solves equations related to asteroid impact physics. For instance, it can predict the trajectory and energy dissipation of an asteroid entering the Earth's atmosphere.
- **`damage.py`**: Calculates and models the potential damage caused by impacts, useful for risk assessment and educational purposes.
- **`locator.py`**: Determines the impact location and outputs the affected postcode and population estimates, a vital tool for disaster management planning.
- **`mapping.py`**: Creates visualizations (like heat maps and impact circles) to represent the impact zone and damage radius visually.

Included Jupyter notebooks (`DataVisualizer.ipynb`, `Chelyabinsk.ipynb`) demonstrate the package's usage and facilitate case studies like the Chelyabinsk meteor event.

## Installation

**Requirements**: Compatible with Python 3.6 and above.

From the base directory, run:

If you are using conda:

```
conda env create -f environment.yml
```

```
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the package in editable mode for easy development.

### Downloading Postcode Data

This step downloads data necessary for the `locator.py` module to function:

```
python download_data.py
```

## Automated Testing

The `pytest` suite covers core functionalities and edge cases:

```shell
pytest tests/
```

To run doctest:

```shell
pytest --doctest-modules
```

## Documentation

Generate HTML documentation:

```
python -m sphinx docs html
```

Find this in the `docs` directory post-generation.

## Example Usage

See `example.py` in the examples folder for a basic demonstration:

```
python examples/example.py
```

`example.py` showcases a simple impact simulation and its results.

## More information

For more information on the project specification, see the Python notebooks:

 `ProjectDescription.ipynb`, `AirburstSolver.ipynb` and `DamageMapper.ipynb`.

## References
In the development of AstroidImpactSim, we have utilized and referenced various libraries and tools.
For a comprehensive list of the libraries and tools used in our project, please refer to the `reference.txt` file.

## Contributing

Contributions are welcome! 

See `CONTRIBUTING.md` for guidelines on how to contribute (bug reports, feature requests, code contributions).

## License

`DeepImpact` is licensed under `MIT License`. See the `LICENSE` file for more details.
