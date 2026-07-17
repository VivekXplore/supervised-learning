
# Breast Cancer Classification using K-Nearest Neighbors (KNN)
## Project Overview
In this project, we built a **K-Nearest Neighbors (KNN)** classifier to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin dataset.
---
## Steps Performed
### 1. Data Preprocessing
- Removed unnecessary columns:
  - `id`
  - `Unnamed: 32`
- Encoded the target column:
  - `M → 1`
  - `B → 0`
### 2. Train-Test Split
The dataset was divided into:
- **80% Training data**
- **20% Testing data**
### 3. Feature Scaling
Since KNN is a distance-based algorithm, we scaled the features using **StandardScaler** so that all features contribute equally to distance calculations.
```python
from sklearn.preprocessing import StandardScaler

⸻

Model Training

We trained multiple KNN models using different values of k from 1 to 19.

KNeighborsClassifier(n_neighbors=k)

For each model, we evaluated:

* Accuracy
* Recall Score

⸻

Choosing the Best k

After comparing the recall scores for different values of k, the best performance was achieved at:

k = 9

⸻

How KNN Works

For a new data point:

1. KNN calculates the distance between the new point and all training samples.
2. The distances are sorted in ascending order.
3. The 9 nearest neighbors (smallest distances) are selected.
4. A majority vote is performed among these 9 neighbors.
5. The class with the highest vote count becomes the predicted label.

⸻

Example

If the 9 nearest neighbors contain:

Malignant (1): 6
Benign (0): 3

Then the model predicts:

Malignant (1)

⸻

Results

* Best k: 9
* Accuracy: High classification performance
* Recall: Best recall among the tested values

⸻

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* mlxtend (for decision boundary visualization)

⸻

Conclusion

The KNN model successfully classified breast cancer tumors after proper preprocessing and feature scaling. By testing multiple neighbor values, k = 9 was selected as the optimal choice, providing the best recall performance for this dataset.
