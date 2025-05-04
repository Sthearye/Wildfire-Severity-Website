# Wildfire Severity Prediction

## Summary

This is the repository for the CS163 final project, California Forest Fire Prediction, created by Sokuntheary Em and Devin Chau. We aim to analyze the key drivers of wildfires in California based on their county in the hopes of reducing them in the future. Through a website, we were able to display our findings and visualizations of what we found!

Link to the website: https://sanrio-fire-patrols-453720.wm.r.appspot.com/

## Setup

Requirements: 
* Python Version 3.12.6
* Google Cloud SDK
Steps to take: 
```
git clone https://github.com/Sthearye/Wildfire-Severity-Website.git
cd Wildfire-Severity-Website
pip install -r requirements.txt
python app4.py
```

## Project Pipeline
```
Data Collection → Cleaning and Preprocessing Data → Feature Engineering → Modeling → Visualizations → Web Deployment
```

* Data Collection: Obtained datasets from CIMIS Weather Station, Cal Fire Incidents, and the US Census Bureau.
  * [CIMIS Weather](https://cimis.water.ca.gov/)
  * [Cal Fire Incidents](https://www.fire.ca.gov/incidents)
  * [US Census](https://www.census.gov/)

     
* Preprocessing Data: Filled in null values, dropped columns that were not used.
   * Removed columns with too many nulls as it was not dependable
   * Saw some null values in a few columns ( < 10)
       *Filled in with the median


* Feature Engineering: Added lagged features, fire season, and broke down different dates for extra columns. 
  * Used lagged features 7 days back to understand if the week affects fires
  * Determined fire seasons
  * Added Months, Days, and Years derived from Date

  
* Modeling: Trained a Random Forest Classifier and a Stacking Regressor (combining Linear Regression, XGBoost, and Random Forest).
  * Random Forest Classifier used to determine low, medium, or high severity
  * Stacking Regressor used to predict the fire severity value based on the dataset
 
    
* Visualizations: Generated visualizations to understand the datasets
  * Chloropleth Heatmap
  * Correlation Matrix
  * Time Series with ACF and PACF
  * Histogram of the distribution of fire frequency

* Web Deployment: Deployed to Google Cloud Platform.


```
Wildfire-Severity-Website/
│
├── app4.py                          # Main Python script to run the web app
├── app.yaml                         # Google Cloud App Engine deployment config
├── requirements.txt                 # Python dependencies
├── flame_loader_cleaned.html       # Fire-themed loading screen
├── tempCodeRunnerFile.py           # Temporary test file (can be deleted)
├── .gcloudignore                   # Ignore rules for Google Cloud deployment
├── .gitignore                      # Ignore rules for Git version control
├── .DS_Store                       # macOS system file (can be deleted)
│
├── .vscode/                        # VS Code project-specific settings
│   └── settings.json
│
├── assets/                         # Static frontend files like CSS and images
│   ├── style.css
│   ├── loading.css
│   ├── 38A06FAD-...source.png
│   └── merlin_...superJumbo.png
│
├── datasets/                       # Processed and raw wildfire-related data
│   ├── California_County_Boundaries.geojson
│   ├── cimis_merged.csv
│   ├── new_merged_df.csv
│   └── desktop.ini
│
├── models/                         # Machine learning model scripts
│   ├── decomposition.py
│   ├── linear_model.py
│   ├── logistic_model.py
│   ├── random_forest_classifier_model.py
│   ├── random_forest_model.py
│   ├── stacking_regressor_model.py
│   ├── statistical_analysis.py
│   ├── xgboost_model.py
│   └── __pycache__/
│
├── pages/                          # Content definitions for app structure or routing
│   ├── analytics.py
│   ├── findings.py
│   ├── home.py
│   ├── objectives.py
│   └── __pycache__/
│
├── visualizations/                 # Scripts for generating visualizations
│   ├── acres_by_county.py
│   ├── ca_choropleth.py
│   ├── california_wildfire_dist.py
│   ├── climate_parallel.py
│   ├── correlation_heatmap.py
│   ├── drought_fire_scatter.py
│   ├── eto_scatter.py
│   ├── monthly_patterns.py
│   ├── wildfire_trends.py
│   └── __pycache__/
```

## Authors

Sokuntheary Em - [LinkedIn](https://www.linkedin.com/in/elaine-em/) - *insert email here*

Devin Chau - [LinkedIn](https://www.linkedin.com/in/devin-chau-66b5b2208/)- chau.devin031602@gmail.com
