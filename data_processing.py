import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Task 1: Data Loading & Validation

df = pd.read_csv('s_users_skills.csv')

print("First 10 rows:")
print(df.head(10))

print("\nTotal records:", len(df))

critical_columns = ['skill_code', 'title', 'proficiency_level']
for col in critical_columns:
    print(f"Null values in {col}: {df[col].isnull().sum()}")

df_cleaned = df.drop_duplicates(subset=['skill_code'])

missing_pct = df_cleaned.isnull().mean() * 100
df_cleaned = df_cleaned.drop(columns=missing_pct[missing_pct > 90].index)
 
# Task 2.1: Standardization 

df_cleaned['category'] = df_cleaned['category'].str.lower().str.strip()
df_cleaned['sub_category'] = df_cleaned['sub_category'].str.lower().str.strip()

# Task 2.2: Proficiency Encoding

def map_proficiency(level):
    if isinstance(level, str) and level.lower().startswith('level'):
        return int(level.split()[-1])
    return pd.to_numeric(level, errors='coerce')

df_cleaned['proficiency_level_encoded'] = df_cleaned['proficiency_level'].apply(map_proficiency)

# Task 2.3: Related Skills Parsing
df_cleaned['related_skills_parsed'] = df_cleaned['related_skills'].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)


# Encode Skill Importance

importance_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_cleaned['skill_importance_encoded'] = df_cleaned['skill_importance'].map(importance_mapping)

# Drop rows with missing encodings
df_cleaned = df_cleaned.dropna(
    subset=['proficiency_level_encoded', 'skill_importance_encoded']
)

# Save Transformed Data

df_cleaned.to_pickle('transformed_feature_set.pkl')
df_cleaned.to_csv('cleaned_baseline_dataset.csv', index=False)

# Create Visualization Folder

os.makedirs('visualizations', exist_ok=True)

# 1. Top 10 Skills by Proficiency
top_skills = (
    df_cleaned.groupby('skill_code')['proficiency_level_encoded']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_skills.plot(kind='bar')
plt.title('Top 10 Skills by Average Proficiency Level')
plt.ylabel('Average Proficiency')
plt.tight_layout()
plt.savefig('visualizations/top_10_skills.png')
plt.close()

# 2. Proficiency Distribution

plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['proficiency_level_encoded'], bins=5, edgecolor='black')
plt.title('Distribution of Proficiency Levels')
plt.xlabel('Proficiency Level')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('visualizations/proficiency_distribution.png')
plt.close()

# 3. Skill Importance Distribution

plt.figure(figsize=(8, 6))
plt.hist(df_cleaned['skill_importance_encoded'], bins=3, edgecolor='black')
plt.title('Distribution of Skill Importance')
plt.xticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.tight_layout()
plt.savefig('visualizations/importance_distribution.png')
plt.close()

# 4. Scatter: Proficiency vs Importance

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_cleaned,
    x='proficiency_level_encoded',
    y='skill_importance_encoded',
    alpha=0.6
)
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.title('Proficiency Level vs Skill Importance')
plt.tight_layout()
plt.savefig('visualizations/proficiency_vs_importance.png')
plt.close()

# 5. Correlation Heatmap

corr = df_cleaned[
    ['proficiency_level_encoded', 'skill_importance_encoded']
].corr()

plt.figure(figsize=(8, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# 6. Boxplot: TOP 10 Departments ONLY

top10_departments = (
    df_cleaned.groupby('department')['proficiency_level_encoded']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .index
)

df_top10 = df_cleaned[df_cleaned['department'].isin(top10_departments)]

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df_top10,
    x='department',
    y='proficiency_level_encoded'
)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Departments by Proficiency Level')
plt.tight_layout()
plt.savefig('visualizations/top10_departments_boxplot.png')
plt.close()

# 7. Word Cloud (Skill Titles)

from wordcloud import WordCloud

text = ' '.join(df_cleaned['title'].dropna())
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Skill Titles')
plt.tight_layout()
plt.savefig('visualizations/skill_titles_wordcloud.png')
plt.close()

print("All cleaned data & visualizations generated successfully.")
