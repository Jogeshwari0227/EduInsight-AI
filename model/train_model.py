import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train():
    # 1. Load dataset
    try:
        # Looking for the file in the data folder
        df = pd.read_csv('data/xAPI-Edu-Data.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: data/xAPI-Edu-Data.csv not found. Ensure the CSV is in the 'data' folder.")
        return

    # 2. Preprocessing
    # Converting categorical text to numbers for the Random Forest
    df['Class'] = df['Class'].map({'L': 0, 'M': 1, 'H': 2})
    df['StudentAbsenceDays'] = df['StudentAbsenceDays'].map({'Under-7': 0, 'Above-7': 1})
    
    # Selecting core features
    features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudentAbsenceDays']
    X = df[features]
    y = df['Class']

    # 3. Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Create directory and save
    if not os.path.exists('model'):
        os.makedirs('model')
    
    joblib.dump(model, 'model/model.pkl')
    print("Success! model/model.pkl has been created.")

if __name__ == "__main__":
    train()