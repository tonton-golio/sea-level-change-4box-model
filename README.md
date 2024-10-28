# Sea Level Prediction Model

This repository contains a Python implementation of a sea level prediction model using Monte Carlo Markov Chain (MCMC) and curve fitting methods. The model forecasts sea level changes based on climate forcing data and various physical contributions to sea level rise.

## Features

* **BaseSeaLevelModel**: A base class for loading and preprocessing forcing and sea level data.
* **ComplexSeaLevelModel**: Extends `BaseSeaLevelModel` to incorporate various contributions from Greenland, Antarctica, glaciers, and thermal expansion.
* **Data Analysis**: Includes methods for calculating Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) for model evaluation.
* **MCMC Sampling**: Uses emcee for MCMC sampling to estimate posterior distributions.
* **Visualization**: Provides plotting tools to compare modeled sea levels against observed data, including confidence intervals and corner plots for parameter distributions.

## Requirements

Install required packages with:

``` bash
pip install pandas numpy matplotlib scipy emcee corner
```

## Usage

### Setup

1. Data Files: Place your forcing and sea level data files in the data folder.
    * `forcing_all.csv`: Radiative forcing data
    * `processed_sea_level_data.csv`: Sea level observations
1. Constants and Parameters:
    * Constants for Greenland, Antarctica, glaciers, and thermal expansion should be set in the script. Sample values are provided but should be adjusted as needed.

### Running the Model

To run the model, execute:

``` bash
python main.py
```

### Output

1. Model Fitting:
    * Uses curve_fit to fit initial parameters.
    * Plots model predictions and calculates AIC/BIC scores.
1. MCMC Sampling:
    * Runs MCMC sampling to estimate parameter distributions.
    * Generates a corner plot of the posterior distributions.
    * Provides confidence intervals for sea level predictions.

### Visualization

The model offers visual outputs:

* Modeled sea level vs. observed data
* Confidence intervals for sea level predictions
* Corner plot for posterior parameter distributions

### File Structure

* `main.py`: Main script to execute the model.
* `BaseSeaLevelModel`: Base class defining preprocessing, likelihood, and plotting methods.
* `ComplexSeaLevelModel`: Extends `BaseSeaLevelModel` to integrate different contributions to sea level rise.

### Example

``` python
# Initialize the complex sea level model
model = ComplexSeaLevelModel(df_F_path, df_S_path, constants, lambda_temp)
# Fit parameters using curve fitting and plot
popt, pcov = model.curve_fit_model()
model.plot_model(popt)
# Run MCMC for parameter estimation
sampler, samples = model.run_mcmc(popt)
model.plot_model_with_ci(samples)
```