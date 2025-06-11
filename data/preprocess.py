import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# Load the raw dataset
df = pd.read_csv("data/raw/healthcare_dataset.csv")

# Drop irrelevant columns
df = df.drop(columns=["Name", "Doctor", "Room Number"])

# Preview results
print("Columns after dropping:")
print(df.columns)



# Convert dates to datetime
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

# Compute stay duration
df["Stay Length"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

# Create Readmitted flag: 1 if stay < 4 days, else 0
df["Readmitted"] = (df["Stay Length"] < 4).astype(int)

# Drop the original date columns and stay length
df = df.drop(columns=["Date of Admission", "Discharge Date", "Stay Length"])

# Preview distribution
print("Readmitted label distribution:")
print(df["Readmitted"].value_counts())


# Drop high-cardinality categorical column
df = df.drop(columns=["Hospital"])

# Encode binary gender
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

# One-hot encode remaining categoricals
categorical_cols = [
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results"
]

df = pd.get_dummies(df, columns=categorical_cols)

print("Final columns:")
print(df.columns)
print("Final shape:", df.shape)


# Normalize numeric columns
scaler = MinMaxScaler()
df[["Age", "Billing Amount"]] = scaler.fit_transform(df[["Age", "Billing Amount"]])

print("Normalized 'Age' sample:")
print(df["Age"].head())

print("Normalized 'Billing Amount' sample:")
print(df["Billing Amount"].head())


# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 5 nearly equal parts
num_clients = 5
shard_size = len(df) // num_clients

# Make sure folder exists
client_dir = "data/clients"
os.makedirs(client_dir, exist_ok=True)

# Split and save
for i in range(num_clients):
    start = i * shard_size
    end = (i + 1) * shard_size if i < num_clients - 1 else len(df)
    client_df = df.iloc[start:end]
    client_path = os.path.join(client_dir, f"client_{i+1}.csv")
    client_df.to_csv(client_path, index=False)
    print(f"Saved {client_path} with shape {client_df.shape}")

# Save the full cleaned dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/full_preprocessed.csv", index=False)
print("Saved full preprocessed dataset")
