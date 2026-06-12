import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))


file_path = os.path.join(script_dir, "HousingData.csv.xls")
housing   = pd.read_csv(file_path)

feature_descriptions = {
    "CRIM"   : "Per capita crime rate by town",
    "ZN"     : "Proportion of residential land zoned for large plots",
    "INDUS"  : "Proportion of non-retail business acres",
    "CHAS"   : "Charles River dummy variable (1 = tract bounds river)",
    "NOX"    : "Nitric oxide concentration (air pollution)",
    "RM"     : "Average number of rooms per dwelling",
    "AGE"    : "Proportion of owner-occupied units built before 1940",
    "DIS"    : "Weighted distance to employment centres",
    "RAD"    : "Index of accessibility to radial highways",
    "TAX"    : "Full-value property tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B"      : "Proportion of Black residents (historical measure)",
    "LSTAT"  : "Percentage of lower-status population",
    "MEDV"   : "Median value of homes in $1000s  <-- TARGET variable"
}

print("-" * 60)
print("DATASET OVERVIEW")
print("-" * 60)
print(f"Rows    : {housing.shape[0]}")
print(f"Columns : {housing.shape[1]}")
print()
print("Preview (first 5 rows):")
print(housing.head())
print()
print("Column descriptions:")
for feature, desc in feature_descriptions.items():
    print(f"  {feature:<10} {desc}")
print()



print("-" * 60)
print("PART 1 - MISSING VALUE TREATMENT")
print("-" * 60)

null_counts = housing.isna().sum()
null_pct    = (null_counts / len(housing) * 100).round(2)
missing_report = pd.DataFrame({"Count": null_counts, "Percentage": null_pct})
cols_with_nulls = missing_report[missing_report["Count"] > 0]

print("Columns that have missing data:")
print(cols_with_nulls)
print()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(cols_with_nulls.index, cols_with_nulls["Count"], color="salmon", edgecolor="black")
ax.set_title("Missing Value Count per Column")
ax.set_xlabel("Feature")
ax.set_ylabel("Number of Missing Entries")
plt.xticks(rotation=40)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "1_missing_values.png"))
plt.close()
print("Chart saved: 1_missing_values.png")

data = housing.copy()
for feature in data.columns:
    n_missing = data[feature].isna().sum()
    if n_missing > 0:
        fill_val = data[feature].median()
        data[feature] = data[feature].fillna(fill_val)
        print(f"  '{feature}': filled {n_missing} gaps with median ({fill_val:.2f})")

print()
remaining = data.isna().sum().sum()
print(f"Total missing values remaining: {remaining}  (should be 0)")
print()

print("-" * 30)
print("PART 2 - OUTLIER TREATMENT")
print("-" * 30)


predictors = [col for col in data.columns if col != "MEDV"]

rows_needed = (len(predictors) + 3) // 4
fig, axes = plt.subplots(rows_needed, 4, figsize=(18, rows_needed * 3))
axes = axes.flatten()
for idx, feature in enumerate(predictors):
    axes[idx].boxplot(data[feature], patch_artist=True,
                      boxprops=dict(facecolor="lightyellow"))
    axes[idx].set_title(feature, fontsize=9)
for j in range(len(predictors), len(axes)):   # hide unused subplots
    axes[j].set_visible(False)
fig.suptitle("Boxplots BEFORE Outlier Treatment  (dots = outliers)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "2_boxplots_before.png"))
plt.close()
print("Chart saved: 2_boxplots_before.png")

clean_data = data.copy()
records = []
for feature in predictors:
    q1 = clean_data[feature].quantile(0.25)
    q3 = clean_data[feature].quantile(0.75)
    iqr = q3 - q1
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    flagged = int(((clean_data[feature] < low) | (clean_data[feature] > high)).sum())
    clean_data[feature] = clean_data[feature].clip(lower=low, upper=high)
    records.append({"Feature": feature, "Lower Cap": round(low, 2),
                    "Upper Cap": round(high, 2), "Outliers Capped": flagged})

summary_table = pd.DataFrame(records)
print("Outlier capping summary:")
print(summary_table.to_string(index=False))
print()

fig, axes = plt.subplots(rows_needed, 4, figsize=(18, rows_needed * 3))
axes = axes.flatten()
for idx, feature in enumerate(predictors):
    axes[idx].boxplot(clean_data[feature], patch_artist=True,
                      boxprops=dict(facecolor="lightcyan"))
    axes[idx].set_title(feature, fontsize=9)
for j in range(len(predictors), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Boxplots AFTER Outlier Treatment  (extremes have been capped)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "3_boxplots_after.png"))
plt.close()
print("Chart saved: 3_boxplots_after.png")




print("-" * 30)
print("PART 3 - WHAT DRIVES HOME PRICES (MEDV)?")
print("-" * 30)


corr_with_price = clean_data.corr()["MEDV"].drop("MEDV").sort_values()

print("Correlation of every feature with MEDV (home price):")
print(corr_with_price.round(3).to_string())
print()
print("  A positive value means the feature tends to RAISE price.")
print("  A negative value means the feature tends to LOWER price.")
print()

bar_colors = ["coral" if v < 0 else "cornflowerblue" for v in corr_with_price]
fig, ax = plt.subplots(figsize=(10, 6))
corr_with_price.plot(kind="barh", color=bar_colors, edgecolor="black", ax=ax)
ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
ax.set_title("Feature Correlation with Home Price (MEDV)\nBlue = raises price  |  Red = lowers price",
             fontsize=12)
ax.set_xlabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "4_correlation_bar.png"))
plt.close()
print("Chart saved: 4_correlation_bar.png")


fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(clean_data.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, annot_kws={"size": 8}, ax=ax)
ax.set_title("Full Correlation Heatmap\nRed = positive relationship  |  Blue = negative relationship",
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "5_heatmap.png"))
plt.close()
print("Chart saved: 5_heatmap.png")

# Step 3c: Scatter plots for the 4 strongest drivers
strongest_positive = corr_with_price.nlargest(2).index.tolist()
strongest_negative = corr_with_price.nsmallest(2).index.tolist()
key_features = strongest_negative + strongest_positive

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, feature in enumerate(key_features):
    x_vals = clean_data[feature]
    y_vals = clean_data["MEDV"]
    axes[idx].scatter(x_vals, y_vals, alpha=0.45, s=25,
                      color="cornflowerblue", edgecolors="white")
    # Trend line
    coeffs  = np.polyfit(x_vals, y_vals, 1)
    trend   = np.poly1d(coeffs)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
    axes[idx].plot(x_range, trend(x_range), color="crimson",
                   linewidth=2, label="Trend line")
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Home Price (MEDV $1000s)")
    axes[idx].set_title(f"{feature}  vs  MEDV  |  r = {corr_with_price[feature]:.2f}")
    axes[idx].legend()
fig.suptitle("4 Strongest Drivers of Home Price", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "6_scatter_plots.png"))
plt.close()
print("Chart saved: 6_scatter_plots.png")


print()
print("-" * 30)
print("FINAL SUMMARY")
print("-" * 30)
print()
print("Features that most INCREASE home price:")
for f in corr_with_price.nlargest(3).index:
    print(f"  [+] {f:<10}  r = {corr_with_price[f]:+.3f}  →  {feature_descriptions[f]}")
print()
print("Features that most DECREASE home price:")
for f in corr_with_price.nsmallest(3).index:
    print(f"  [-] {f:<10}  r = {corr_with_price[f]:+.3f}  →  {feature_descriptions[f]}")
print()
print("-" * 30)
print("ANALYSIS COMPLETE — check folder for saved chart images")
print("-" * 30)