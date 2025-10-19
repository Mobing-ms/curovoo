import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


patients = pd.read_csv("/Volumes/ExtraSpace/curovoo/Data/Personalized_Diet_Recommendations1.csv")

for col in ['Gender', 'Chronic_Disease', 'Preferred_Cuisine']:
    if col in patients.columns:
        patients[col] = LabelEncoder().fit_transform(patients[col].astype(str))


X = patients[['Age', 'Gender', 'BMI', 'Chronic_Disease']]
y = patients[['Recommended_Calories', 'Recommended_Protein', 
              'Recommended_Carbs', 'Recommended_Fats']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


user = pd.DataFrame([{
    'Age': 45,
    'Gender': 1,              # Example encoded value
    'BMI': 28,
    'Chronic_Disease': 2      # e.g. 0=None, 1=Hypertension, 2=Diabetes, etc.
}])


predicted = model.predict(user)
print("Recommended Calorie, Protein, Carb, Fat intake (approx):")
print(predicted)


indb = pd.read_excel("/Volumes/ExtraSpace/curovoo/Data/ANUVAAD_INDB_DiseaseTagged.xlsx")


disease_label = "Diabetes"
safe_foods = indb[~indb["Diseases_to_Avoid"].str.contains(disease_label, case=False)]

