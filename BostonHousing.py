
import pandas as pd           
import numpy as np            
import matplotlib.pyplot as plt  
import seaborn as sns        


plt.rcParams['figure.figsize'] = (10, 6)

#import os
#os.chdir("/Users/rohaan/Downloads")
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#initially used this
#df = pd.read_csv("/Users/rohaan/Downloads/HousingData.csv.xls")
df = pd.read_csv("HousingData.csv.xls")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Number of rows (houses): {df.shape[0]}")
print(f"Number of columns (features): {df.shape[1]}")
print()

print("First 5 rows of the dataset:")
print(df.head())
print()


column_info = {
    "CRIM"   : "Crime rate per town",
    "ZN"     : "% of residential land zoned for large lots",
    "INDUS"  : "% of non-retail business acres per town",
    "CHAS"   : "1 if next to Charles River, 0 otherwise",
    "NOX"    : "Nitrogen oxide concentration (pollution)",
    "RM"     : "Average number of rooms per house",
    "AGE"    : "% of homes built before 1940",
    "DIS"    : "Distance to employment centres",
    "RAD"    : "Accessibility to highways",
    "TAX"    : "Property tax rate",
    "PTRATIO": "Pupil-to-teacher ratio",
    "B"      : "Proportion of Black residents (historical measure)",
    "LSTAT"  : "% of lower-status population",
    "MEDV"   : "Median home value in $1000s  <-- This is what we want to predict"
}
print("What each column means:")
for col, desc in column_info.items():
    print(f"  {col:10} : {desc}")
print()

print("$" * 60)
print("                PART 1: MISSING VALUE TREATMENT")
print("$" * 60)

missing_count = df.isnull().sum()   # Count missing values per column
missing_pct   = (missing_count / len(df)) * 100  # As a percentage

print("Missing values per column:")
missing_df = pd.DataFrame({
    "Missing Count"  : missing_count,
    "Missing Percent": missing_pct.round(2)
})
print(missing_df[missing_df["Missing Count"] > 0])
print()


plt.figure()
missing_count[missing_count > 0].plot(kind="bar", color="tomato", edgecolor="black")
plt.title("Number of Missing Values per Column")
plt.xlabel("Column Name")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("1_missing_values.png")
plt.close()
print("Chart saved: 1_missing_values.png")


df_clean = df.copy()   

for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)
        print(f"  Filled missing values in '{col}' with median = {median_value:.2f}")

print()
print("Missing values after treatment:", df_clean.isnull().sum().sum())
print("(0 means no missing values remain)")
print()


print(":)" * 60)
print("                          PART 2: OUTLIER TREATMENT")
print(":(" * 60)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "MEDV"]  # Exclude the target

n_plots = len(numeric_cols)
n_grid_rows = (n_plots + 3) // 4
plt.figure(figsize=(18, n_grid_rows * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_grid_rows, 4, i)
    plt.boxplot(df_clean[col], patch_artist=True,
                boxprops=dict(facecolor="lightblue"))
    plt.title(col, fontsize=10)
    plt.ylabel("Value")
plt.suptitle("Boxplots BEFORE Outlier Treatment\n(Dots above/below the whiskers are outliers)",
             fontsize=13)
plt.tight_layout()
plt.savefig("2_boxplots_before.png")
plt.close()
print("Chart saved: 2_boxplots_before.png")

df_treated = df_clean.copy()

outlier_summary = []
for col in numeric_cols:
    Q1  = df_treated[col].quantile(0.25)
    Q3  = df_treated[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Count how many outliers exist
    outliers_count = ((df_treated[col] < lower_fence) |
                      (df_treated[col] > upper_fence)).sum()

    # Cap the outliers
    df_treated[col] = df_treated[col].clip(lower=lower_fence, upper=upper_fence)

    outlier_summary.append({
        "Column"        : col,
        "Lower Fence"   : round(lower_fence, 2),
        "Upper Fence"   : round(upper_fence, 2),
        "Outliers Found": outliers_count
    })

outlier_df = pd.DataFrame(outlier_summary)
print("Outlier treatment summary:")
print(outlier_df.to_string(index=False))
print()

# --- 2d. Show boxplots AFTER treatment ---
plt.figure(figsize=(18, n_grid_rows * 3))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_grid_rows, 4, i)
    plt.boxplot(df_treated[col], patch_artist=True,
                boxprops=dict(facecolor="lightgreen"))
    plt.title(col, fontsize=10)
    plt.ylabel("Value")
plt.suptitle("Boxplots AFTER Outlier Treatment\n(Outliers have been capped)",
             fontsize=13)
plt.tight_layout()
plt.savefig("3_boxplots_after.png")
plt.close()
print("Chart saved: 3_boxplots_after.png")


print("*" * 60)
print("PART 3: UNDERSTANDING WHAT DRIVES HOME PRICES (MEDV)")
print("*" * 60)

correlation = df_treated.corr()["MEDV"].drop("MEDV").sort_values()

print("Correlation of each feature with Home Price (MEDV):")
print(correlation.round(3).to_string())
print()
print("Interpretation:")
print("  Positive number → higher value = higher price")
print("  Negative number → higher value = lower price")
print()

colors = ["tomato" if c < 0 else "steelblue" for c in correlation]

plt.figure()
correlation.plot(kind="barh", color=colors, edgecolor="black")
plt.title("Correlation of Features with Home Price (MEDV)\n"
          "Blue = Positive  |  Red = Negative", fontsize=12)
plt.xlabel("Correlation Coefficient")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("4_correlation_bar.png")
plt.close()
print("Chart saved: 4_correlation_bar.png")

plt.figure(figsize=(12, 9))
sns.heatmap(df_treated.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5,
            annot_kws={"size": 8})
plt.title("Correlation Heatmap\n(Darker red = stronger positive, Darker blue = stronger negative)",
          fontsize=12)
plt.tight_layout()
plt.savefig("5_heatmap.png")
plt.close()
print("Chart saved: 5_heatmap.png")

top_positive = correlation.nlargest(2).index.tolist()
top_negative = correlation.nsmallest(2).index.tolist()
top_features = top_negative + top_positive   # 4 features total

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    axes[i].scatter(df_treated[col], df_treated["MEDV"],
                    alpha=0.5, color="steelblue", edgecolors="white", s=30)
  
    z = np.polyfit(df_treated[col], df_treated["MEDV"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_treated[col].min(), df_treated[col].max(), 100)
    axes[i].plot(x_line, p(x_line), color="tomato", linewidth=2, label="Trend")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("MEDV (Home Price $1000s)")
    axes[i].set_title(f"{col} vs Home Price (Correlation: {correlation[col]:.2f})")
    axes[i].legend()

plt.suptitle("Top 4 Features vs Home Price", fontsize=14)
plt.tight_layout()
plt.savefig("6_scatter_plots.png")
plt.close()
print("Chart saved: 6_scatter_plots.png")


print()
print("=" * 60)
print("FINAL SUMMARY: KEY DRIVERS OF HOME PRICE")
print("=" * 60)
print()
print("TOP FEATURES THAT INCREASE HOME PRICE:")
for col in correlation.nlargest(3).index:
    print(f"  + {col:10}: {correlation[col]:+.3f}  → {column_info[col]}")

print()
print("TOP FEATURES THAT DECREASE HOME PRICE:")
for col in correlation.nsmallest(3).index:
    print(f"  - {col:10}: {correlation[col]:+.3f}  → {column_info[col]}")

print()
print("HULULU" * 10)
print("ALL DONE! Check the saved .png files for all charts.")
print("HULULU" * 10)
