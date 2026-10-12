# Lab Sheet 02 - Harsh Saini

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run this file or notebook from the practicals folder.
DATA = Path("datasets")
OUT = Path("outputs") / "lab_02"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 78)
pd.set_option("display.max_columns", 8)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
np.set_printoptions(precision=3, suppress=True)
sns.set_theme(style="whitegrid", font_scale=0.9)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.png", dpi=160)
    plt.show()


# Program 01: Load a dataset and identify missing values in each column.

try:
    original = pd.read_csv(DATA / "employees.csv")
except (FileNotFoundError, pd.errors.ParserError) as error:
    raise RuntimeError("Check datasets/employees.csv") from error
print("Shape:", original.shape)
print(original.isna().sum())


# Program 02: Display the percentage of missing values in every feature.

print((original.isna().mean() * 100).round(1))


# Program 03: Remove rows containing missing values from the dataset.

complete_rows = original.dropna()
print("Before:", len(original))
print("After:", len(complete_rows))
print(complete_rows.head())


# Program 04: Remove columns having more than 50% missing values.

kept = original.loc[:, original.isna().mean() <= 0.50].copy()
print("Kept:", kept.columns.tolist())
print("Dropped:", original.columns.difference(kept.columns).tolist())


# Program 05: Replace missing numerical values using the mean.

mean_filled = original.copy()
columns = mean_filled.select_dtypes(include="number").columns
mean_filled[columns] = mean_filled[columns].fillna(
    mean_filled[columns].mean())
print(mean_filled[columns].head(8))
print("Missing numeric values:",
      mean_filled[columns].isna().sum().sum())


# Program 06: Replace missing numerical values using the median.

median_filled = original.copy()
columns = median_filled.select_dtypes(include="number").columns
median_filled[columns] = median_filled[columns].fillna(
    median_filled[columns].median())
print(median_filled[columns].head(8))


# Program 07: Replace missing categorical values using the mode.

mode_filled = original.copy()
mode = mode_filled["department"].mode()[0]
mode_filled["department"] = mode_filled["department"].fillna(mode)
print("Mode:", mode)
print(mode_filled.loc[5:9, "department"])


# Program 08: Fill missing values using forward fill (Forward Propagation).

forward = original.copy().ffill()
print(forward[["age", "salary", "department"]].iloc[4:9])
print("Missing bonus values:", forward["bonus"].isna().sum())


# Program 09: Fill missing values using backward fill (Backward Propagation).

backward = original.copy().bfill()
print(backward[["age", "salary", "department"]].iloc[4:9])
print("Missing bonus values:", backward["bonus"].isna().sum())


# Program 10: Compare the dataset before and after handling missing values.

clean = original.drop(columns="bonus").copy()
numeric = ["age", "salary", "experience", "performance"]
clean[numeric] = clean[numeric].fillna(clean[numeric].median())
clean["department"] = clean["department"].fillna(
    clean["department"].mode()[0])
print(pd.DataFrame({"before": original.isna().sum(),
                    "after": clean.isna().sum().reindex(
                        original.columns)}))
clean.to_csv(OUT / "missing_values_handled.csv", index=False)


# Program 11: Detect outliers using the Interquartile Range (IQR) method.

salary = clean["salary"]
q1, q3 = salary.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = (salary < lower) | (salary > upper)
print("IQR limits:", lower, upper)
print(clean.loc[outlier_mask, ["salary", "experience"]])


# Program 12: Detect outliers using the Z-score method.

z_scores = (salary - salary.mean()) / salary.std(ddof=0)
print(pd.DataFrame({"salary": salary,
                    "z_score": z_scores}).loc[z_scores.abs() > 3])


# Program 13: Visualize outliers using a Box Plot.

plt.figure(figsize=(6, 3))
sns.boxplot(x=clean["salary"])
plt.title("Salary before outlier treatment")
save_plot("13_boxplot")


# Program 14: Visualize outliers using a Scatter Plot.

plt.figure(figsize=(6, 3))
plt.scatter(clean["experience"], clean["salary"])
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
plt.title("Salary and experience")
save_plot("14_scatter")


# Program 15: Remove outliers using the IQR method.

no_outliers = clean.loc[~outlier_mask].copy()
print("Rows before:", len(clean))
print("Rows after:", len(no_outliers))
no_outliers.to_csv(OUT / "outliers_removed.csv", index=False)


# Program 16: Replace outliers with the median value.

replaced = clean.copy()
replacement = clean.loc[~outlier_mask, "salary"].median()
replaced.loc[outlier_mask, "salary"] = replacement
print("Replacement median:", replacement)
print(replaced.loc[outlier_mask, ["salary", "experience"]])


# Program 17: Cap outliers using percentile-based capping.

capped = clean.copy()
low, high = capped["salary"].quantile([0.05, 0.95])
capped["salary"] = capped["salary"].clip(low, high)
print("5th and 95th percentile:", low, high)
print("New salary range:", capped["salary"].min(),
      capped["salary"].max())


# Program 18: Compare the dataset before and after outlier treatment.

print(pd.DataFrame({"original": clean["salary"].describe(),
                    "removed": no_outliers["salary"].describe(),
                    "replaced": replaced["salary"].describe(),
                    "capped": capped["salary"].describe()}))


# Program 19: Apply Min-Max Normalization to numerical features.

from sklearn.preprocessing import MinMaxScaler
features = replaced[["age", "salary", "experience"]].copy()
minmax = pd.DataFrame(MinMaxScaler().fit_transform(features),
                      columns=features.columns)
print(minmax.head())
print("Minimums:", minmax.min().tolist())
print("Maximums:", minmax.max().tolist())


# Program 20: Apply Standardization (Z-score Scaling).

from sklearn.preprocessing import StandardScaler
standard = pd.DataFrame(StandardScaler().fit_transform(features),
                        columns=features.columns)
print(standard.head())
print("Means:", standard.mean().round(3).tolist())
print("Standard deviations:", standard.std(ddof=0).round(3).tolist())


# Program 21: Apply Robust Scaling to handle outliers.

from sklearn.preprocessing import RobustScaler
robust = pd.DataFrame(RobustScaler().fit_transform(
    clean[["age", "salary", "experience"]]), columns=features.columns)
print(robust.head())
print("Outlier salary after scaling:", robust.loc[13, "salary"])


# Program 22: Apply Max Absolute Scaling.

from sklearn.preprocessing import MaxAbsScaler
maxabs = pd.DataFrame(MaxAbsScaler().fit_transform(features),
                      columns=features.columns)
print(maxabs.head())
print("Maximum absolute values:", maxabs.abs().max().tolist())


# Program 23: Compare original and normalized datasets.

print(pd.DataFrame({"original_salary": features["salary"],
                    "minmax_salary": minmax["salary"]}).head(8))


# Program 24: Visualize the effect of normalization using histograms.

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].hist(features["salary"], bins=6, edgecolor="black")
axes[0].set_title("Original salary")
axes[1].hist(minmax["salary"], bins=6, edgecolor="black")
axes[1].set_title("Min-max salary")
save_plot("24_histograms")


# Program 25: Visualize the effect of scaling using box plots.

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
sns.boxplot(data=features, ax=axes[0])
axes[0].set_title("Before scaling")
sns.boxplot(data=standard, ax=axes[1])
axes[1].set_title("After standardization")
save_plot("25_scaling_boxplots")


# Program 26: Compare different scaling techniques on the same dataset.

comparison = pd.DataFrame({
    "original": features["salary"],
    "minmax": minmax["salary"],
    "standard": standard["salary"],
    "robust": robust["salary"],
    "maxabs": maxabs["salary"]})
print(comparison.head(8))


# Program 27: Encode categorical variables using Label Encoding.

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
labels = encoder.fit_transform(clean["department"])
print("Mapping:", dict(zip(encoder.classes_,
                            range(len(encoder.classes_)))))
print("Encoded values:", labels)


# Program 28: Encode categorical variables using One-Hot Encoding.

one_hot = pd.get_dummies(clean["department"],
                         prefix="dept", dtype=int)
print(one_hot.head(8))


# Program 29: Perform Binary Encoding on categorical data.

codes, categories = pd.factorize(clean["department"], sort=True)
# Reserve 0 for unknown categories; known categories start at 1.
codes = codes + 1
width = int(np.ceil(np.log2(len(categories) + 1)))
binary = pd.DataFrame({
    f"dept_bit_{bit}": (codes >> bit) & 1
    for bit in range(width)})
print("Categories:", categories.tolist())
print(binary.head(8))


# Program 30: Create a new feature by combining two existing columns.

engineered = replaced.copy()
engineered["salary_per_year"] = (
    engineered["salary"] / engineered["experience"].clip(lower=1))
print(engineered[["salary", "experience", "salary_per_year"]].head())


# Program 31: Extract year, month, and day from a date column.

dates = pd.to_datetime(original["joined"], errors="raise")
parts = pd.DataFrame({"year": dates.dt.year,
                      "month": dates.dt.month,
                      "day": dates.dt.day})
print(parts.head())


# Program 32: Create a new feature using mathematical transformations.

transformed = replaced.copy()
transformed["experience_squared"] = transformed["experience"] ** 2
transformed["salary_sqrt"] = np.sqrt(transformed["salary"])
print(transformed[["experience", "experience_squared",
                   "salary_sqrt"]].head())


# Program 33: Apply Log Transformation to skewed data.

log_salary = np.log1p(clean["salary"])
print("Original skew:", round(clean["salary"].skew(), 3))
print("Log skew:", round(log_salary.skew(), 3))
plt.figure(figsize=(6, 3))
plt.hist(log_salary, bins=8, edgecolor="black")
plt.xlabel("log(1 + salary)")
plt.title("Log-transformed salary")
save_plot("33_log_salary")


# Program 34: Perform Feature Selection using correlation analysis.

correlations = replaced[numeric].corr()["performance"].drop(
    "performance")
selected = correlations[correlations.abs() >= 0.60].index.tolist()
print("Correlation with performance:")
print(correlations)
print("Selected at |r| >= 0.60:", selected)


# Program 35: Create a final preprocessed dataset ready for Machine Learning.

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Split original rows before fitting any preprocessing step.
X = original[["age", "salary", "experience", "department"]]
y = original["performance"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
numeric_columns = ["age", "salary", "experience"]
num_pipeline = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())])
cat_pipeline = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False))])
processor = ColumnTransformer([
    ("num", num_pipeline, numeric_columns),
    ("cat", cat_pipeline, ["department"])])
train_array = processor.fit_transform(X_train)
test_array = processor.transform(X_test)
names = processor.get_feature_names_out()
train_ready = pd.DataFrame(train_array, columns=names)
test_ready = pd.DataFrame(test_array, columns=names)
train_ready["performance"] = y_train.to_numpy()
test_ready["performance"] = y_test.to_numpy()
train_ready.to_csv(OUT / "train_preprocessed.csv", index=False)
test_ready.to_csv(OUT / "test_preprocessed.csv", index=False)
print("Train:", train_ready.shape, "Test:", test_ready.shape)
print("Remaining missing values:",
      train_ready.isna().sum().sum() + test_ready.isna().sum().sum())
print(train_ready.head(3))
