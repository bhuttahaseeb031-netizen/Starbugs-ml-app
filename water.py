import joblib
import pandas as pd
import streamlit as st

st.title("🍹 Alcohol Consumption Prediction App")

# 1. Dataset load karein
@st.cache_data
def load_data():
    return pd.read_csv("drinks.csv")


df = load_data()
st.write("### Dataset Preview", df.head())

# 2. Model train ya load karne ka logic
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df[["beer_servings", "spirit_servings", "wine_servings"]]
median_alcohol = df["total_litres_of_pure_alcohol"].median()
y = (df["total_litres_of_pure_alcohol"] > median_alcohol).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 3. User Input for Prediction
st.subheader("Enter Values for Prediction:")
beer = st.number_input("Beer Servings", value=50.0)
spirit = st.number_input("Spirit Servings", value=50.0)
wine = st.number_input("Wine Servings", value=50.0)

if st.button("Predict 🚀"):
    input_data = [[beer, spirit, wine]]
    prediction = clf.predict(input_data)
    if prediction[0] == 1:
        st.success("Result: High Consumption (Above Median)")
    else:
        st.success("Result: Low Consumption (Below Median)")
