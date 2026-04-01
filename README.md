# MSE446 — Ride-Pooling Optimization for Emissions Reduction

This repository explores ride-pooling opportunities in NYC FHV (for-hire vehicle) trip data and estimates potential CO₂ reductions using spatiotemporal clustering. The work is organized as reproducible Jupyter notebooks for data cleaning, exploratory analysis, clustering experiments, and pooled-ride evaluation.

### Fleet Composition (NYC FHV)
- **52%** SUV *(Sport Utility Vehicle 2WD / 4WD)*
- **36%** Sedans *(Compact, Midsize, Large Cars)*
- **12%** Minivans *(Minivan 2WD / 4WD)*
This split was identified based on a report by NYC TLC: https://www.nyc.gov/assets/tlc/downloads/pdf/driver_expense_report.pdf

## Project overview

Goals:
- Identify poolable trips using spatial + temporal proximity constraints
- Quantify potential CO₂ savings from pooling under practical constraints
- Compare clustering approaches (K-Means, Hierarchical, DBSCAN) and evaluation metrics

High-level pipeline:
1. Clean and merge trip data with vehicle emissions (DataCleaning/)
2. Exploratory analysis and feature engineering (EDA/)
3. Train and evaluate clustering methods (Clustering/)
4. Compute pooling outcomes and emissions impacts (Clustering/)

## Quick start

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Open the notebooks with Jupyter Lab or run them headlessly with `nbconvert` (examples below).

## Data

- Data/raw/: raw and cleaned CSVs used by the notebooks.
- Data/generated/: outputs produced by notebooks (cluster labels, summaries, comparisons).

Key files (in this repo):
- Data/raw/merged_trip_emissions_coordinates.csv — merged trip + vehicle emissions with coordinates
- Data/raw/vehicles_clean.csv — processed vehicle emissions grouped by class
- Data/generated/ — CSVs created by model runs and evaluations

Large raw trip files are excluded via `.gitignore`. If you need the original large raw CSVs, request access from the data owner — they are not stored in this repo.

## Notebooks and how to run them

You can run notebooks interactively (Jupyter Lab / Notebook) or execute them from the command line. Examples below assume your current working directory is the repository root and that `venv` is activated.

Run a notebook (in-place execution):

```bash
# Data cleaning
jupyter nbconvert --to notebook --execute DataCleaning/For_Hire_dataclean.ipynb --inplace
jupyter nbconvert --to notebook --execute DataCleaning/Fleet_List.ipynb --inplace

# Clustering experiments
jupyter nbconvert --to notebook --execute Clustering/dbscan.ipynb --inplace
jupyter nbconvert --to notebook --execute Clustering/kmeans_comparison.ipynb --inplace
jupyter nbconvert --to notebook --execute Clustering/hierarchical_clustering.ipynb --inplace

# Model comparison / pooling evaluation
jupyter nbconvert --to notebook --execute Clustering/model_comparison.ipynb --inplace
jupyter nbconvert --to notebook --execute Clustering/poolability_scaleup.ipynb --inplace
```

Notes:
- Running the full pipeline requires the merged/cleaned CSVs in `Data/raw/` (see Key files above).
- Notebooks contain comments where paths or large-file references need to be adjusted if you host raw files outside this repo.

## Project structure

```
./
├─ Data/
│  ├─ raw/                # raw & cleaned CSVs used by notebooks
│  └─ generated/          # notebook outputs (summary CSVs, comparisons)
├─ DataCleaning/          # notebooks: data cleaning & vehicle mapping
├─ EDA/                   # notebooks: exploratory data analysis
├─ Clustering/            # notebooks: clustering experiments + evaluation
│  └─ figures/            # generated visualizations
├─ requirements.txt       # Python dependencies
└─ README.md
```

## Dependencies

All Python dependencies are listed in `requirements.txt`. Typical packages used across the notebooks include `pandas`, `numpy`, `scikit-learn`, `geopandas` (for zone shapefiles), and plotting libraries like `matplotlib` / `seaborn`.

Install them with:

```bash
pip install -r requirements.txt
```

## Notes and tips

- If you don't have the large raw trip CSVs, use the cleaned files under `Data/raw/` (the notebooks expect those filenames).
- Inspect the top cells of each notebook for any hard-coded file paths and update them to match your local setup.
- For reproducible runs, activate the `venv` and run notebooks in the order shown in the Quick start section.
