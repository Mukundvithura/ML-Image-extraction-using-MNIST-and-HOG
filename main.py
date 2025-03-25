import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.datasets import mnist
import time

# 1️⃣ Load Full MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

print(f"Dataset Loaded: {X_train.shape[0]} training, {X_test.shape[0]} testing samples.")

# 2️⃣ HOG Feature Extraction Function (Improved)
def extract_hog_features(images):
    return np.array([hog(img, pixels_per_cell=(4, 4), cells_per_block=(3, 3), feature_vector=True) for img in images])

# Extract HOG features (May take a few minutes)
start_time = time.time()
X_train_hog, X_test_hog = extract_hog_features(X_train), extract_hog_features(X_test)
print(f"HOG Features Extracted in {time.time() - start_time:.2f} seconds.")
print(f"Feature shape: {X_train_hog.shape[1]} features per image.")

# 3️⃣ Train Improved Classifier (Random Forest with More Trees)
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train_hog, y_train)

# Alternative: Try Gradient Boosting (Slightly Slower but More Accurate)
# clf = GradientBoostingClassifier(n_estimators=300)
# clf.fit(X_train_hog, y_train)

# 4️⃣ Evaluate Model
y_pred = clf.predict(X_test_hog)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
