from dietmodel import train_diet_model
import pandas as pd

# Train the model
model = train_diet_model()


age = int(input("Enter your age: "))
gender = int(input("Enter your gender (0: Female, 1: Male): "))
bmi = float(input("Enter your BMI: "))
chronic_disease = int(input("Enter your chronic disease status (0: None, 1: Diabetes, 2: Hypertension, 3: Heart Disease, 4: Obesity): "))

# Make a prediction
user = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Chronic_Disease': chronic_disease
}])
prediction = model.predict(user)
print("Recommended Calorie, Protein, Carb, Fat intake (approx):")
print(prediction)
