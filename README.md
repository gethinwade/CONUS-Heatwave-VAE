# CONUS Heatwave Detection with a Variational Autoencoder

This project detects heatwaves over the continental United States (CONUS) using a variational autoencoder (VAE) trained on ERA5 reanalysis data. It adapts the VAE-based heatwave detection method of Paçal et al., and extends it from the European domain to the continental United States.

The research was carried out at [LEAP](https://leap.columbia.edu/) (Learning the Earth with Artificial Intelligence and Physics) at Columbia University.

## Repository Structure

```
conus-heatwave-vae/
├── README.md
├── LICENSE
├── environment.yml
├── src/
│   ├── config.py              # paths, domain, and model settings
│   ├── preprocess.py          # ERA5 download, regridding, deseasonalization
│   ├── vae_train.py           # VAE architecture and training loop
│   └── analysis.py            # latent-space analysis
├── notebooks/
│   ├── preprocessing.ipynb    # ERA5 extraction, relative humidity, land mask
│   ├── build_samples.ipynb    # threshold, onset detection, deseasonalization, sample extraction
│   ├── vae.ipynb              # VAE training
│   └── analysis.ipynb         # latent-space analysis and US heatwave plots
└── data/
    └── events.csv             # detected heatwave events
```

## Pipeline

Run the notebooks in this order:

1. **preprocessing**: extracts daily ERA5 variables from ARCO-ERA5 at native resolution, computes relative humidity from t2m and d2m, and builds a land-sea mask.
2. **build_samples**: smooths and decimates t2m to a 48×128 grid, computes a 90th-percentile day-of-year threshold climatology, and detects heatwave onsets over CONUS land (saved to `events.csv`). It then smooths and decimates the remaining variables, deseasonalizes all seven, and extracts VAE samples.
3. **vae**: trains the VAE on the samples and saves the weights to `models/vae_conus.pth`.
4. **analysis**: encodes the samples with the trained VAE and produces maps and plots of the detected US heatwave events.

## Data

- **Input:** ERA5 reanalysis from the Copernicus Climate Change Service (C3S). The raw data is not included in this repository because of its size. The pipeline was run on LEAP-Pangeo, with variables read from the ARCO-ERA5 store.
- **`data/events.csv`:** heatwave events detected by the pipeline.
- The code was developed on the LEAP-Pangeo JupyterHub. File paths in `config.py` and the notebooks will need to be updated to run it.

## Data Attribution

Generated using modified Copernicus Climate Change Service (C3S) ERA5 data, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.

> Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

## Acknowledgments

This work was supported by LEAP, an NSF Science and Technology Center at Columbia University. Thanks to Celeste Bian for her guidance on this project.

## License

The code is released under the MIT License (see `LICENSE`).
