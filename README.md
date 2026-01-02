# User Skills Data Analysis:

## Project Overview
This project builds an **end-to-end data pipeline** for user skills data, covering data cleaning, feature engineering, exploratory analysis, and visualization.  
The final output is a **high-quality, ML-ready dataset** with insightful visualizations to support **AI/ML models and business decision-making**.

The workflow is divided into **four structured sprints**, following industry-standard data analytics practices.

---

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- WordCloud
- OS, AST

---

## Sprint 1: Data Ingestion & Baseline Cleaning
**Objective:** Ensure data integrity and create a clean baseline dataset.

### Steps
- Load CSV data and verify ingestion
- Display first 10 records
- Count total records
- Check missing values in critical columns:
  - `skill_code`
  - `title`
  - `proficiency_level`
- Remove duplicate skills based on `skill_code`
- Drop columns with more than **90% missing data**

### Deliverable
 `cleaned_baseline_dataset.csv`

---

## Sprint 2: Feature Engineering & ML Preparation
**Objective:** Convert raw data into structured, ML-compatible features.

### Steps
- Standardize text columns (`category`, `sub_category`)
- Encode proficiency levels into numeric format
- Parse related skills into list structures
- Encode skill importance:
  - Low → 1
  - Medium → 2
  - High → 3
- Remove invalid or incomplete records

### Deliverable
`transformed_feature_set.pkl`

---

##  Sprint 3: Exploratory Data Analysis (EDA)
**Objective:** Discover patterns, correlations, and data gaps.

### Analysis Performed
- Skill density across departments
- Relationship between proficiency and importance
- Identification of missing or anomalous skill data

### Deliverable
 EDA insights through visual analysis

 ---
 

 ### Sprint 4: Visualization & Insights
**Objective:** Generate interpretable visual insights for business and ML use cases.

#### 1. Top 10 Skills by Average Proficiency
**Purpose:** Identify the strongest skills across the dataset.  
**Business Value:** Helps prioritize specialization and talent benchmarking.

#### 2. Proficiency Level Distribution
**Purpose:** Shows how skill levels are spread from beginner to expert.  
**Business Value:** Reveals workforce maturity and training needs.

#### 3. Skill Importance Distribution
**Purpose:** Displays the count of low, medium, and high importance skills.  
**Business Value:** Highlights organizational priorities.

#### 4. Proficiency vs Skill Importance (Scatter Plot)
**Purpose:** Analyzes alignment between importance and proficiency.  
**Business Value:** Identifies critical skill gaps.

#### 5. Correlation Heatmap
**Purpose:** Shows relationships between numeric features.  
**Business Value:** Supports feature validation for ML models.

#### 6. Department-wise Proficiency (Top 10 Departments Box Plot)
**Purpose:** Compares proficiency variation across departments.  
**Business Value:** Supports department-level training decisions.

#### 7. Skill Title Word Cloud
**Purpose:** Visualizes most frequent skill keywords.  
**Business Value:** Identifies trending and in-demand skills.

---

## Final Deliverables
- Cleaned Baseline Dataset  
- ML-Ready Transformed Dataset  
- Insightful Visualizations  
- Automated Data Export  

---

## ▶ How to Run the Project
```bash
- pip install pandas numpy matplotlib seaborn wordcloud
- python user_skills_pipeline.py
