# MSE446 — Ride-Pooling Optimization for Emissions Reduction

A machine learning project that reduces urban carbon emissions by optimizing ride-pooling efficiency within the ride-sharing industry. Using unsupervised learning (spatiotemporal clustering) on public NYC FHV trip data, the system identifies opportunities to combine overlapping solo rides.

---

### Data
| File | Description |
|---|---|
| `data/FHVTrip_VehicleEmissions_Merged_2023.csv` | Main dataset — 2023 NYC High Volume FHV trips merged with EPA vehicle emissions |
| `data/FLEET_LIST_20260222.csv` | Raw NYC TLC fleet list |
| `data/vehicles.csv` | Raw EPA vehicle data |
| `data/vehicles_clean.csv` | Cleaned & grouped EPA vehicle emissions by class |

### Fleet Composition (NYC FHV)
- **52%** SUV *(Sport Utility Vehicle 2WD / 4WD)*
- **36%** Sedans *(Compact, Midsize, Large Cars)*
- **12%** Minivans *(Minivan 2WD / 4WD)*

---

## Notebooks

### `DataCleaning/Fleet_List.ipynb`
- Loads and cleans the EPA vehicle dataset
- Maps vehicle classes to three fleet categories (SUV, Sedans, Minivans)
- Computes average CO₂ (g/mile), city MPG, and fuel cost per class
- Randomly assigns vehicle classes to each trip using the fleet distribution as sampling probabilities
- Outputs `data/FHVTrip_VehicleEmissions_Merged_2023.csv`

### `DataCleaning/For_Hire_dataclean.ipynb`
- Loads and cleans the 2023 NYC High Volume FHV trip CSV
- Filters to relevant columns: datetime, location IDs, trip miles/time, shared match flag
- Drops rows with invalid/missing values
- Loads and cleans the EPA vehicles dataset in parallel

### `EDA/trip_distribution_eda.ipynb`
- **Trip distance distribution** — histogram, KDE by vehicle class, CDF with pooling threshold percentiles
- **Top pickup & dropoff zones** — bar charts + origin-destination heatmap for high-frequency zones
- **CO₂ emissions by vehicle class** — boxplot and average g/mile comparison
- **Shared vs. solo trip analysis** — pie chart + KDE comparing pooled vs. solo trip distances
- **Temporal patterns** — hourly trip volume to identify peak demand windows for clustering

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn
```

> The raw trip CSV (`2023_High_Volume_FHV_Trip_Data_20260227.csv`) is excluded from version control via `.gitignore` due to file size. Request access from the team.
