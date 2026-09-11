# StrikeCAST

StrikeCAST is a machine-learning-based system for predicting thunderstorm probabilities over Germany and surrounding regions.

The system combines multiple meteorological parameters with lookup-based predictors to calculate a spatial thunderstorm probability. The resulting probabilities are then processed into a geographic visualization and exported as a WebP image.

## Overview

StrikeCAST uses meteorological input data and additional lookup information as features for a machine learning model.

The model calculates a thunderstorm probability for each grid point within a defined geographical bounding box covering Germany and parts of neighboring countries.

The resulting probability field is visualized as a weather map.

## Input

The prediction is based on a combination of different meteorological parameters and lookup-based predictors.

Depending on the available data and model configuration, these parameters can include atmospheric and meteorological variables relevant to thunderstorm development.

Lookup predictors provide additional information that can be incorporated into the model alongside the meteorological features.

## Prediction

For each grid point within the prediction area, StrikeCAST calculates a probability representing the likelihood of thunderstorm activity.

Conceptually, the prediction can be represented as:

```text
Meteorological Parameters
          +
Lookup Predictors
          │
          ▼
   Machine Learning Model
          │
          ▼
Thunderstorm Probability
```

The resulting probabilities form a spatial probability field over the prediction domain.

## Prediction Area

The primary prediction area covers **Germany and parts of neighboring countries**.

A geographical bounding box is used to define the prediction domain, allowing the model to represent thunderstorm probabilities both inside Germany and in surrounding regions.

## Output

StrikeCAST generates a **WebP weather map** containing the predicted thunderstorm probabilities.

The output represents the spatial distribution of the model's predicted probability across the defined prediction area.

The resulting maps can be used for visualization, analysis, and further processing of the predicted thunderstorm activity.

## Data & Technologies

StrikeCAST uses Python-based scientific and meteorological data processing tools, including:

* **NumPy** – numerical computations
* **Pandas** – tabular data processing
* **Xarray** – multidimensional meteorological data
* **SciPy** – scientific computing
* **Matplotlib** – visualization
* **Cartopy** – geographic visualization and map projections
* **ECMWF Open Data** – meteorological forecast data
* **ecCodes** – meteorological data processing
* **cfgrib** – GRIB data access
* **netCDF4** – NetCDF data processing
* **Pillow** – image processing and WebP generation
* **Boto3** – AWS data access

## Disclaimer

StrikeCAST provides machine-learning-based probability estimates. The generated probabilities represent model predictions and should not be interpreted as a guarantee that a thunderstorm will occur at a specific location.
