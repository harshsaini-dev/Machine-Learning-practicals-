# Lab Sheet 01 - Harsh Saini

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run this file or notebook from the practicals folder.
DATA = Path("datasets")
OUT = Path("outputs") / "lab_01"
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


# Program 01: Install Python and verify the installed version using Python commands.

import platform
print("Python:", platform.python_version())
print("Executable:", Path(sys.executable).name)
print("Python >= 3.11:", sys.version_info >= (3, 11))


# Program 02: Install Jupyter Notebook and launch the notebook interface.

import subprocess
result = subprocess.run(
    [sys.executable, "-m", "notebook", "--version"],
    capture_output=True, text=True, check=True)
print("Jupyter Notebook version:", result.stdout.strip())
print("Launch command: python -m notebook")


# Program 03: Install NumPy using pip and verify its installation.

import numpy
print("NumPy version:", numpy.__version__)
print("Array:", numpy.array([10, 20, 30]))


# Program 04: Install Pandas using pip and verify its installation.

import pandas
print("Pandas version:", pandas.__version__)
print(pandas.DataFrame({"name": ["Aman", "Riya"],
                        "marks": [78, 88]}))


# Program 05: Install Matplotlib and create a simple line plot.

import matplotlib
print("Matplotlib version:", matplotlib.__version__)
plt.figure(figsize=(6, 3))
plt.plot([1, 2, 3, 4], [10, 15, 13, 20], marker="o")
plt.xlabel("Day")
plt.ylabel("Value")
plt.title("Simple line plot")
save_plot("05_line_plot")


# Program 06: Install Seaborn and generate a basic statistical plot.

print("Seaborn version:", sns.__version__)
plt.figure(figsize=(6, 3))
sns.barplot(x=["Aman", "Riya", "Kabir"], y=[78, 88, 65])
plt.ylabel("Marks")
plt.title("Marks of three students")
save_plot("06_bar_plot")


# Program 07: Install Scikit-learn and verify its version.

import sklearn
from sklearn.datasets import load_iris
print("Scikit-learn version:", sklearn.__version__)
print("Iris shape:", load_iris().data.shape)


# Program 08: Create and execute your first Python program in Jupyter Notebook.

# Paste these statements in a Jupyter code cell.
print("Hello, Python!")
a = 12
b = 8
print("Sum:", a + b)


# Program 09: Create and execute a Python script using Visual Studio Code or PyCharm.

# Save as first_program.py and run in VS Code or PyCharm.
length = 8
width = 5
area = length * width
print("Length:", length)
print("Width:", width)
print("Area:", area)


# Program 10: Create a virtual environment and install required Python libraries inside it.

import tempfile
import venv
import subprocess

# Reuse installed packages for this lightweight demonstration.
with tempfile.TemporaryDirectory() as temp:
    folder = Path(temp) / "demo_env"
    venv.EnvBuilder(system_site_packages=True).create(folder)
    executable = folder / (
        "Scripts/python.exe" if sys.platform == "win32"
        else "bin/python")
    result = subprocess.run(
        [str(executable), "-c",
         "import numpy, pandas; print('Libraries verified')"],
        capture_output=True, text=True, check=True)
    print("Virtual environment created:", folder.exists())
    print(result.stdout.strip())
print("For a fresh local environment:")
print("python -m venv .venv")
print(".venv\\Scripts\\activate   (Windows)")
print("python -m pip install -r requirements.txt")


# Program 11: Load a CSV dataset using the Pandas library.

try:
    students = pd.read_csv(DATA / "students.csv")
    print("Dataset loaded:", students.shape)
    print(students.head(3))
except (FileNotFoundError, pd.errors.ParserError) as error:
    raise RuntimeError("Check datasets/students.csv") from error


# Program 12: Display the first five records of a dataset using head().

print(students.head())


# Program 13: Display the last five records of a dataset using tail().

print(students.tail())


# Program 14: Find the total number of rows and columns in a dataset.

rows, columns = students.shape
print("Rows:", rows)
print("Columns:", columns)


# Program 15: Display the names of all columns in the dataset.

print("Columns:", students.columns.tolist())


# Program 16: Check the data types of all columns.

print(students.dtypes)


# Program 17: Generate descriptive statistics of numerical features.

print(students.describe())


# Program 18: Display complete information about the dataset using info().

students.info()


# Program 19: Identify missing values in the dataset.

print(students.isna())


# Program 20: Count the total missing values in each column.

print(students.isna().sum())


# Program 21: Display unique values present in a selected column.

print("Courses:", students["course"].unique())


# Program 22: Count the frequency of each unique value in a column.

print(students["course"].value_counts())


# Program 23: Rename one or more columns in a dataset.

renamed = students.rename(columns={"name": "student_name",
                                   "marks": "score"})
print(renamed.head(3))


# Program 24: Select specific rows and columns using loc[].

print(students.loc[0:3, ["name", "marks"]])


# Program 25: Select specific rows and columns using iloc[].

print(students.iloc[0:4, [0, 3]])


# Program 26: Filter records based on a given condition.

print(students.loc[students["marks"] >= 80])


# Program 27: Sort the dataset using one or more columns.

print(students.sort_values(["course", "marks"],
                           ascending=[True, False]))


# Program 28: Add a new column to an existing dataset.

with_column = students.copy()
with_column["passed"] = with_column["marks"].ge(70).where(
    with_column["marks"].notna(), other=pd.NA)
print(with_column.head(6))


# Program 29: Delete an existing column from the dataset.

without_column = with_column.drop(columns="passed")
print("Remaining columns:", without_column.columns.tolist())


# Program 30: Remove duplicate records from a dataset.

deduplicated = students.drop_duplicates()
print("Before:", len(students))
print("After:", len(deduplicated))
print("Duplicates removed:", len(students) - len(deduplicated))


# Program 31: Save the modified dataset as a new CSV file.

deduplicated.to_csv(OUT / "students_modified.csv", index=False)
reloaded = pd.read_csv(OUT / "students_modified.csv")
print("Saved file: students_modified.csv")
print("Reloaded shape:", reloaded.shape)


# Program 32: Load a dataset directly from the Scikit-learn library.

from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
print("Feature shape:", iris.data.shape)
print("Target names:", iris.target_names)
print(iris.data.head(3))


# Program 33: Create a histogram for a numerical feature using Matplotlib.

plt.figure(figsize=(6, 3))
plt.hist(students["marks"].dropna(), bins=6, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.title("Distribution of marks")
save_plot("33_histogram")


# Program 34: Create a scatter plot to visualize the relationship between two variables.

plt.figure(figsize=(6, 3))
plt.scatter(students["age"], students["marks"])
plt.xlabel("Age")
plt.ylabel("Marks")
plt.title("Age and marks")
save_plot("34_scatter")


# Program 35: Generate a correlation matrix and visualize it using a heatmap.

correlation = students[["age", "marks"]].corr()
print(correlation)
plt.figure(figsize=(5, 3))
sns.heatmap(correlation, annot=True, vmin=-1, vmax=1,
            cmap="coolwarm")
plt.title("Correlation matrix")
save_plot("35_correlation")
