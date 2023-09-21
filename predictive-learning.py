import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Step 1: Data Collection and Preparation
# Load the historical student data
data = pd.read_csv("student_data.csv")

# Preprocess the data
data.fillna(0, inplace=True)  # Replace missing values with 0

# Encode categorical variables using LabelEncoder
categorical_cols = ["gender", "class"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 2: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data["math_score"], bins=20, kde=True)
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.title("Distribution of Math Scores")

plt.subplot(1, 2, 2)
sns.countplot(data["target_variable"])
plt.xlabel("Pass/Fail")
plt.ylabel("Count")
plt.title("Distribution of Pass/Fail")
plt.show()

# Step 3: Feature Engineering
# Create relevant features
data["attendance_percentage"] = (data["attended_classes"] / data["total_classes"]) * 100
data["frequent_absences"] = (data["attendance_percentage"] < 80).astype(int)

# Step 4: Model Building
# Define features and target variable
X = data.drop("target_variable", axis=1)  # Replace "target_variable" with your target variable
y = data["target_variable"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

# Evaluate and visualize model performance
plt.figure(figsize=(14, 5))

for i, (model_name, model) in enumerate(models.items(), 1):
    plt.subplot(1, len(models), i)

    # Cross-validation for model evaluation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    # Train the model on the full training set
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    accuracy = classification_rep['accuracy']

    # Plot confusion matrix
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Pass', 'Fail'])
    plt.title(f"{model_name}\nAccuracy: {accuracy:.2f}")
    plt.grid(False)

plt.tight_layout()
plt.show()
