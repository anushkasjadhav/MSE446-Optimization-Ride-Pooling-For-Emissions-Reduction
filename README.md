# MSE446 — Ride-Pooling Optimization for Emissions Reduction

A machine learning project that reduces urban carbon emissions by optimizing ride-pooling efficiency within the ride-sharing industry. Using unsupervised learning (spatiotemporal clustering) on public NYC FHV trip data, the system identifies opportunities to combine overlapping solo rides.

## Project Overview
Ride-pooling has the potential to significantly reduce urban carbon emissions when implemented effectively. However, practical constraints—such as passenger convenience and wait times—must be considered to ensure feasibility. In this project, pooling eligibility is constrained by spatiotemporal limits (ex. time between trips), balancing environmental benefits with user experience.

This project identifies and evaluates ride-pooling opportunities using real-world trip data. Specifically, it:

1. Processes and cleans raw NYC FHV trip data and EPA vehicle emissions data  
2. Maps vehicle classes to trips and estimates emissions per trip  
3. Applies unsupervised machine learning to identify poolable trips based on proximity  
4. Quantifies potential carbon emission reductions from pooled trips  
5. Visualizes clustering results and pooling effectiveness to support analysis and decision-making  

---

### Data
| File | Description |
|---|---|
| `Data/raw/FHVTrip_VehicleEmissions_Merged_2023.csv` | 2023 NYC High Volume FHV trips merged with EPA vehicle emissions using taxi zones |
| `Data/raw/merged_trip_emissions_coordinates.csv` | 2023 NYC High Volume FHV trips merged with EPA vehicle emissions using taxi zones |
| `Data/raw/FLEET_LIST_20260222.csv` | Raw NYC TLC fleet list |
| `Data/raw/vehicles.csv` | Raw EPA vehicle data |
| `Data/raw/vehicles_clean.csv` | Cleaned & grouped EPA vehicle emissions by class |
| `Data/raw/taxi_zones` | Taxi zone to coordinate mapping as a shape file |


### Fleet Composition (NYC FHV)
- **52%** SUV *(Sport Utility Vehicle 2WD / 4WD)*
- **36%** Sedans *(Compact, Midsize, Large Cars)*
- **12%** Minivans *(Minivan 2WD / 4WD)*
This split was identified based on a report by NYC TLC: https://www.nyc.gov/assets/tlc/downloads/pdf/driver_expense_report.pdf

### Methodology
- Feature engineering: spatial (lat/lon) + temporal (hour encoding)
- Clustering approaches:
  - K-Means (partition-based)
  - Hierarchical clustering (distance-based)
  - DBSCAN (density-based)
- Evaluation metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Cluster size distribution

### Key Results
- Identified significant ride-pooling opportunities using clustering techniques  
- Achieved 1.7% reduction in CO₂ emissions under pooling constraints  
- Best-performing model: K-Means based on silhouette and DBI metrics, DBSCAN based on maximum percent reduction in CO₂ emissions
- Demonstrated trade-off between **emissions reduction and passenger convenience constraints**

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/anushkasjadhav/MSE446-Optimization-Ride-Pooling-For-Emissions-Reduction.git
cd MSE446-Optimization-Ride-Pooling-For-Emissions-Reduction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure 
```
MSE446-Optimization-Ride-Pooling-For-Emissions-Reduction/
├── Data/                      # Data directory
│   ├── raw/                   # Raw and preprocessed datasets
│   ├── generated/             # Output of data trained on models
├── DataCleaning/              # Data preprocessing scripts
├── EDA/                       # Exploratory data analysis
├── Clustering/                # Trained models and evaluation
│   ├── figures/               # Visualizations of results and analysis
├── gitignore                  # Ignore files to prevent pushing
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

### Accessing the Data

#### Option 1: Download from Source
1. Navigate to https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. Apply filters [ask Kanika and Anushka]
3. Download folder
4. Upload to local GitHub repo 

#### Option 2: Use Pre-processed Data (Recommended)
1. The preprocessed data files can be found here [add google drive link]
2. Download the zip file and unzip it locally
3. Upload the folder locally to GitHub repo 

The `Data/raw/vehicles_clean.csv` and `Data/raw/merged_trip_emissions_coordinates` are the key files that must be present in the repo for the model and other evaluations to run. 


## Running the Complete Project Pipeline

### Data Processing and Cleaning
If you went with Option 1 and downloaded the raw data from the source, you will need run preprocessing scripts to generate the cleaned datasets. 

Uncomment line 7 and ensure that the path to the raw datafile is correct.  
Line 7: df = pd.read_csv('/workspaces/MSE446-Optimization-Ride-Pooling-For-Emissions-Reduction/.gitignore/2023_High_Volume_FHV_Trip_Data_20260227.csv'). 
```bash
# Process and clean the raw trip data 
jupyter nbconvert --to notebook --execute DataCleaning/For_Hire_dataclean.ipynb --inplace

# Merge vehicle dataset with trip dataset and extract features from processed data
# Ensure that the path to the raw datafile is correct
# Line 6: df = pd.read_csv("Data/vehicles.csv")
jupyter nbconvert --to notebook --execute DataCleaning/Fleet_List.ipynb --inplace
```
Note: An alternative to running the commands from the terminal is clicking 'Run All' option from each notebook. 

### Model Training 
Available models: dbscan, kmeans, hierarchical clustering

#### DBSCAN (Density Based Spatial Clustering of Applications with Noise)
```bash
jupyter nbconvert --to notebook --execute Clustering/dbscan.ipynb --inplace
```

#### K Means Clustering
```bash
jupyter nbconvert --to notebook --execute Clustering/kmeans_comparison.ipynb --inplace
```

#### Hierarchical Clustering 
```bash
jupyter nbconvert --to notebook --execute Clustering/hierarchical_clustering.ipynb --inplace
```
Note: Within each model training notebook, hyperparameter tuning and validation are conducted. 

### Model Comparison 
```bash
jupyter nbconvert --to notebook --execute Clustering/model_comparison.ipynb --inplace
```
Comparison of the three models by number of clusters created, noise, silhouette score, davies-bouldin index. 

### Pooling Rides, Evaluation and Visualization
```bash
jupyter nbconvert --to notebook --execute Clustering/poolability_scaleup.ipynb --inplace
```


### Notebooks

### `DataCleaning/For_Hire_dataclean.ipynb`
- Loads and cleans the 2023 NYC High Volume FHV trip CSV
- Filters to relevant columns: datetime, location IDs, trip miles/time, shared match flag
- Drops rows with invalid/missing values
- Loads and cleans the EPA vehicles dataset in parallel

### `DataCleaning/Fleet_List.ipynb`
- Loads and cleans the EPA vehicle dataset
- Maps vehicle classes to three fleet categories (SUV, Sedans, Minivans)
- Computes average CO₂ (g/mile), city MPG, and fuel cost per class
- Randomly assigns vehicle classes to each trip using the fleet distribution as sampling probabilities
- Outputs `Data/raw/FHVTrip_VehicleEmissions_Merged_2023.csv`

### `EDA/trip_distribution_eda.ipynb`
- **Trip distance distribution** — histogram, KDE by vehicle class, CDF with pooling threshold percentiles
- **Top pickup & dropoff zones** — bar charts + origin-destination heatmap for high-frequency zones
- **CO₂ emissions by vehicle class** — boxplot and average g/mile comparison
- **Shared vs. solo trip analysis** — pie chart + KDE comparing pooled vs. solo trip distances
- **Temporal patterns** — hourly trip volume to identify peak demand windows for clustering

---

> The raw trip CSV (`2023_High_Volume_FHV_Trip_Data_20260227.csv`) is excluded from version control via `.gitignore` due to file size. Request access from the team.
