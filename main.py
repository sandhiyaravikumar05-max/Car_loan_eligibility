import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("car loan.csv")

print("CSV Loaded Successfully!")
print("Original Shape:", df.shape)

# Remove unwanted spaces from column names
df.columns = df.columns.str.strip()

print("\nColumns in CSV:")
print(df.columns.tolist())


# ==========================================
# 2. DATA CLEANING
# ==========================================

df = df.drop_duplicates()

# Fill missing numeric values
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing text values
text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nData Cleaning Completed!")


# ==========================================
# 3. TARGET COLUMN
# ==========================================

# Your CSV target column
target = "loan_status"

# Check target exists
if target not in df.columns:
    print("\nERROR: loan_status column not found!")
    print("Available columns are:")
    print(df.columns.tolist())
    exit()

print("\nTarget Column:", target)


# ==========================================
# 4. REMOVE LOAN ID
# ==========================================

if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])


# ==========================================
# 5. ENCODE CATEGORICAL COLUMNS
# ==========================================

# Encode education
if "education" in df.columns:
    education_encoder = LabelEncoder()
    df["education"] = education_encoder.fit_transform(
        df["education"].astype(str)
    )

# Encode self_employed
if "self_employed" in df.columns:
    employment_encoder = LabelEncoder()
    df["self_employed"] = employment_encoder.fit_transform(
        df["self_employed"].astype(str)
    )

# Encode loan_status
loan_status_encoder = LabelEncoder()

df["loan_status"] = loan_status_encoder.fit_transform(
    df["loan_status"].astype(str)
)

print("\nEncoding Completed!")


# ==========================================
# 6. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["loan_status"])

y = df["loan_status"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# 8. DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ==========================================
# 10. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 11. ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("DECISION TREE RESULTS")
print("====================================")

print("Accuracy: {:.2f}%".format(accuracy * 100))


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(model, "car_loan_model.pkl")

print("\n====================================")
print("SUCCESS!")
print("====================================")

print("Model saved as:")
print("car_loan_model.pkl")