# Used Car Price Prediction

A machine learning project that predicts the price of used cars based on vehicle attributes such as manufacturer, condition, mileage, fuel type, and more. Built with scikit-learn and trained on a large real-world dataset of US used car listings.

---

## Project Structure

```
├── used_cars.ipynb       # Pickle file containing preprocessed train/test splits and the fitted ColumnTransformer
├── car_model.ipynb       # Model training notebook — loads preprocessed data, trains a Linear Regression model
└── README.md
```

> **Note:** `used_cars.ipynb` is a serialised pickle file (despite the `.ipynb` extension) that stores the output of the preprocessing pipeline. It is loaded directly by the model notebook using Python's `pickle` module.

---

## Dataset

- **Total samples:** ~383,068 listings (80/20 train-test split)
  - Training set: 306,454 samples
  - Test set: 76,614 samples
- **Target variable:** `price` (USD)
  - Range: $501 – $99,999
  - Mean: ~$19,191 | Median: ~$15,900
- **Source:** US used car listings (Craigslist-style dataset)

---

## Features

The preprocessing pipeline produces **121 features** from the following raw columns:

| Encoding | Transformer | Columns |
|---|---|---|
| Ordinal | `OrdinalEncoder` | `cylinders`, `title_status`, `condition` |
| One-Hot | `OneHotEncoder` | `manufacturer`, `fuel`, `transmission`, `drive`, `type`, `state` |
| Scaled | `StandardScaler` | `year`, `odometer` |

The fitted `ColumnTransformer` is saved inside `used_cars.ipynb` (the pickle) and reused at model training time to ensure consistent encoding.

---

## Preprocessing Pipeline (`used_cars.ipynb` / pickle)

The preprocessing notebook handles:

1. Cleaning the raw dataset and handling missing values
2. Encoding categorical features using `OrdinalEncoder` and `OneHotEncoder`
3. Scaling numerical features (`year`, `odometer`) with `StandardScaler`
4. Splitting data into `X_train`, `X_test`, `y_train`, `y_test`
5. Serialising everything (arrays + fitted transformer) to a pickle file for reuse

---

## Model Training (`car_model.ipynb`)

The modelling notebook:

1. Loads the preprocessed splits from the pickle file
2. Inspects the encoded feature matrix and target series
3. Trains a **Linear Regression** model via scikit-learn
4. Evaluates using **R²**, **MAE**, and **RMSE**

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```

> The notebook also imports `Ridge` regression — regularised modelling is a natural next step if overfitting is observed.

---

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

---

## Usage

1. Run the preprocessing notebook (or use the existing pickle file) to generate `used_cars.ipynb`.
2. Open and run `car_model.ipynb` to train the model and evaluate predictions.

---

