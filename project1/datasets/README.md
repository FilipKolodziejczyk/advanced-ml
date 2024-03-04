# Datasets for binary classification


## Small Datasets

### 1. [National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset)
- **Features:** 7
- **Instances:** 6287
- **Objective:** Predict age group (senior/non-senior) based on medical features.

#### Preprocessing applied:
- drop SEQN serial number
- drop collinear variables: 
  - LBXIN (0.55 correlation with BMXBMI)
  - LBXGLU (0.69 correlation with LBXGLT)
- transform age group to binary target (senior/non-senior):
  - 0: Adult
  - 1: Senior

### 2. [Ajwa or Medjool](https://archive.ics.uci.edu/dataset/879/ajwa+or+medjool)
- **Features:** 6
- **Instances:** 200
- **Objective:** Predict date fruit species (Ajwa/Medjool) based on physical dimensions, weight, and calories.

### 3. [Fertility Dataset](https://archive.ics.uci.edu/dataset/244/fertility)
- **Features:** 9
- **Instances:** 100
- **Objective:** Predict fertility based on behavioral factors (discrete numerical values based on subject's response).

## Large Datasets

### 1. [Mice Protein Expression](https://archive.ics.uci.edu/dataset/342/mice+protein+expression)
- **Features:** ~80
- **Instances:** 1080 (has missing values)
- **Objective:** Predict whether mice have Down syndrome based on some proteins in the brain.

### 2. [Secondary Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- **Features:** 20
- **Instances:** 61068 (simulated data)
- **Objective:** Simulated data based on the original mushroom dataset. Consider using the original dataset for popularity.
- **Citations:** 1
- **Views:** 11685

### 3. [Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)
- **Features:** 18
- **Instances:** 10129
- **Objective:** Estimate room occupancy (0, 1, 2, 3 people) based on sensor data.

### 4. [Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
- **Features:** 95
- **Instances:** 6819
- **Objective:** Predict whether a company will go bankrupt based on business-related data.

### 5. [Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **Features:** 16
- **Instances:** 2111 (preprocessing required)
- **Objective:** Predict obesity level based on patient's survey. Data conversion required for binary meaning (e.g., normal+overweight vs. obese).

### 6. [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Features:** 11
- **Instances:** 4898
- **Objective:** Predict wine quality based on parameters. Option to predict wine color instead.
