# diet_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_diet_model(csv_path="/Volumes/ExtraSpace/curovoo/Data/Personalized_Diet_Recommendations1.csv"):
    # Load and prepare data
    patients = pd.read_csv(csv_path)
    for col in ['Gender', 'Chronic_Disease', 'Preferred_Cuisine']:
        patients[col] = LabelEncoder().fit_transform(patients[col].astype(str))
    
    X = patients[['Age', 'Gender', 'BMI', 'Chronic_Disease']]
    y = patients[['Recommended_Calories', 'Recommended_Protein', 
                  'Recommended_Carbs', 'Recommended_Fats']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    return model
