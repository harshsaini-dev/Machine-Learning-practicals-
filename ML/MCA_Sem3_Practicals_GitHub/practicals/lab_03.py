# Lab Sheet 03 - Harsh Saini

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Run this file or notebook from the practicals folder.
DATA = Path("datasets")
OUT = Path("outputs") / "lab_03"
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
import joblib

def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted),
            "MSE": mse, "RMSE": np.sqrt(mse),
            "R2": r2_score(actual, predicted)}


# Program 01: Load a regression dataset using the Pandas library.

try:
    data = pd.read_csv(DATA / "diabetes.csv")
except (FileNotFoundError, pd.errors.ParserError) as error:
    raise RuntimeError("Check datasets/diabetes.csv") from error
print("Dataset: scikit-learn diabetes (real clinical data)")
print("Shape:", data.shape)


# Program 02: Display the first and last five records of the dataset.

print("First five records:")
print(data.head())
print("Last five records:")
print(data.tail())


# Program 03: Explore dataset information and descriptive statistics.

data.info()
print("Statistics:")
print(data.describe().round(2).T)
print("Total missing:", data.isna().sum().sum())


# Program 04: Identify input (independent) and output (dependent) variables.

X = data.drop(columns="target")
y = data["target"]
print("Inputs:", X.columns.tolist())
print("Output:", y.name)
print("Simple regression input: bmi")


# Program 05: Split the dataset into training and testing sets.

train_index, test_index = train_test_split(
    np.arange(len(data)), test_size=0.20, random_state=42)
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
bmi_train, bmi_test = X_train[["bmi"]], X_test[["bmi"]]
print("Training:", X_train.shape, "Testing:", X_test.shape)
print("Overlapping indices:",
      len(set(train_index).intersection(test_index)))


# Program 06: Implement a Simple Linear Regression model.

simple = LinearRegression()
print(simple)
print("Input feature:", bmi_train.columns.tolist())


# Program 07: Train the Linear Regression model using the training dataset.

simple.fit(bmi_train, y_train)
print("Model trained on", len(bmi_train), "records")
print("Coefficient:", simple.coef_)
print("Intercept:", round(simple.intercept_, 3))


# Program 08: Predict output values using the trained Linear Regression model.

simple_pred = simple.predict(bmi_test)
print("First predictions:", np.round(simple_pred[:8], 3))
print("Test R2:", round(r2_score(y_test, simple_pred), 3))


# Program 09: Visualize the Linear Regression line using Matplotlib.

line_x = np.linspace(X["bmi"].min(), X["bmi"].max(), 100)
line_frame = pd.DataFrame({"bmi": line_x})
plt.figure(figsize=(6, 3.5))
plt.scatter(bmi_test["bmi"], y_test, label="Test records", s=18)
plt.plot(line_x, simple.predict(line_frame), color="red",
         label="Fitted line")
plt.xlabel("BMI (dataset scale)")
plt.ylabel("Disease progression")
plt.legend()
save_plot("09_regression_line")


# Program 10: Compare actual and predicted values.

actual_vs_pred = pd.DataFrame({"actual": y_test.to_numpy(),
                               "predicted": simple_pred})
actual_vs_pred["error"] = (actual_vs_pred["actual"] -
                            actual_vs_pred["predicted"])
print(actual_vs_pred.head(8))


# Program 11: Display the regression coefficient and intercept.

print("Slope:", round(float(simple.coef_[0]), 3))
print("Intercept:", round(float(simple.intercept_), 3))
print("Equation: target = slope * bmi + intercept")


# Program 12: Predict the output for new user-defined input values.

new_input = pd.DataFrame({"bmi": [0.02, 0.04]})
print("New inputs (dataset scale):")
print(new_input)
print("Predictions:", simple.predict(new_input).round(3))


# Program 13: Implement Multiple Linear Regression using multiple independent variables.

multiple = LinearRegression()
print("Multiple regression feature count:", X_train.shape[1])
print(multiple)


# Program 14: Train the Multiple Linear Regression model.

multiple.fit(X_train, y_train)
print("Trained features:", multiple.feature_names_in_)
print("Training R2:", round(multiple.score(X_train, y_train), 3))


# Program 15: Predict output values using the testing dataset.

multiple_pred = multiple.predict(X_test)
print(pd.DataFrame({"actual": y_test.to_numpy(),
                    "predicted": multiple_pred}).head(8))
print("Test metrics:")
print(pd.Series(metrics(y_test, multiple_pred)))


# Program 16: Compare actual and predicted values graphically.

plt.figure(figsize=(5.5, 3.5))
plt.scatter(y_test, multiple_pred, s=20)
limits = [min(y_test.min(), multiple_pred.min()),
          max(y_test.max(), multiple_pred.max())]
plt.plot(limits, limits, "r--", label="Perfect prediction")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
save_plot("16_actual_predicted")


# Program 17: Analyze the effect of each independent variable on prediction.

coefficients = pd.Series(multiple.coef_, index=X.columns)
print(coefficients.sort_values(key=abs, ascending=False))


# Program 18: Implement Polynomial Regression of Degree 2.

degree2 = make_pipeline(PolynomialFeatures(degree=2,
                                         include_bias=False),
                        LinearRegression())
degree2.fit(bmi_train, y_train)
pred2 = degree2.predict(bmi_test)
print("Degree 2 test metrics:")
print(pd.Series(metrics(y_test, pred2)))


# Program 19: Implement Polynomial Regression of Degree 3.

degree3 = make_pipeline(PolynomialFeatures(degree=3,
                                         include_bias=False),
                        LinearRegression())
degree3.fit(bmi_train, y_train)
pred3 = degree3.predict(bmi_test)
print("Degree 3 test metrics:")
print(pd.Series(metrics(y_test, pred3)))


# Program 20: Compare Linear Regression and Polynomial Regression models.

model_metrics = pd.DataFrame({
    "Linear BMI": metrics(y_test, simple_pred),
    "Polynomial 2": metrics(y_test, pred2),
    "Polynomial 3": metrics(y_test, pred3),
    "Multiple linear": metrics(y_test, multiple_pred)}).T
print(model_metrics.round(3))


# Program 21: Visualize Polynomial Regression curves.

plt.figure(figsize=(6, 3.5))
plt.scatter(bmi_test["bmi"], y_test, s=15, alpha=0.6)
for label, model in [("Linear", simple), ("Degree 2", degree2),
                     ("Degree 3", degree3)]:
    plt.plot(line_x, model.predict(line_frame), label=label)
plt.xlabel("BMI (dataset scale)")
plt.ylabel("Disease progression")
plt.legend()
save_plot("21_polynomial_curves")


# Program 22: Predict output values using the Polynomial Regression model.

print(pd.DataFrame({"bmi": new_input["bmi"],
                    "degree2": degree2.predict(new_input),
                    "degree3": degree3.predict(new_input)}))


# Program 23: Compare prediction accuracy for different polynomial degrees.

accuracy = model_metrics.loc[
    ["Linear BMI", "Polynomial 2", "Polynomial 3"], ["RMSE", "R2"]]
print(accuracy)
print("Smallest test RMSE:", accuracy["RMSE"].idxmin())


# Program 24: Calculate Mean Absolute Error (MAE).

mae = mean_absolute_error(y_test, multiple_pred)
print("MAE:", round(mae, 3))


# Program 25: Calculate Mean Squared Error (MSE).

mse = mean_squared_error(y_test, multiple_pred)
print("MSE:", round(mse, 3))


# Program 26: Calculate Root Mean Squared Error (RMSE).

rmse = np.sqrt(mse)
print("RMSE:", round(rmse, 3))


# Program 27: Calculate the R-squared (R²) Score.

r2 = r2_score(y_test, multiple_pred)
print("R2:", round(r2, 3))


# Program 28: Compare the performance of Linear and Polynomial Regression using evaluation metrics.

print(model_metrics[["MAE", "MSE", "RMSE", "R2"]].round(3))
print("All models evaluated on", len(y_test), "test records")


# Program 29: Interpret the meaning of MSE and R² values.

print("Multiple linear MSE:", round(mse, 3))
print("Multiple linear R2:", round(r2, 3))
print("MSE is average squared error; lower is better.")
print("R2 = 1 is perfect; 0 matches the test-mean baseline.")
print("A negative R2 is worse than that baseline.")


# Program 30: Visualize prediction errors using scatter plots.

errors = y_test.to_numpy() - multiple_pred
plt.figure(figsize=(6, 3))
plt.scatter(multiple_pred, errors, s=20)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted target")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual plot")
save_plot("30_residuals")


# Program 31: Train the regression model using standardized features.

scaled_model = make_pipeline(StandardScaler(), LinearRegression())
scaled_model.fit(X_train, y_train)
scaled_pred = scaled_model.predict(X_test)
print(pd.Series(metrics(y_test, scaled_pred)))


# Program 32: Compare model performance before and after feature scaling.

print(pd.DataFrame({
    "unscaled": metrics(y_test, multiple_pred),
    "scaled": metrics(y_test, scaled_pred)}).T)
print("Maximum prediction difference:",
      np.max(np.abs(multiple_pred - scaled_pred)))


# Program 33: Perform regression using another real-world dataset.

other = pd.read_csv(DATA / "linnerud.csv")
other_X = other[["Chins", "Situps", "Jumps"]]
other_y = other["Weight"]
a, b, c, d = train_test_split(other_X, other_y,
                             test_size=0.25, random_state=42)
other_model = LinearRegression().fit(a, c)
print("Dataset: Linnerud (20 real exercise measurements)")
print("Train/test:", a.shape, b.shape)
print(pd.Series(metrics(d, other_model.predict(b))))


# Program 34: Save the trained regression model using the Joblib library.

joblib.dump(scaled_model, OUT / "regression_model.joblib")
print("Saved: regression_model.joblib")
print("Model includes fitted scaler and linear regression")


# Program 35: Load the saved model and use it to predict new data.

loaded_model = joblib.load(OUT / "regression_model.joblib")
new_data = X_test.iloc[:2].copy()
loaded_predictions = loaded_model.predict(new_data)
print("Inputs (first two held-out records):")
print(new_data)
print("Reloaded predictions:", loaded_predictions.round(3))
print("Matches original:", np.allclose(
    loaded_predictions, scaled_model.predict(new_data)))
