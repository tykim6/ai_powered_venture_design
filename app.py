import streamlit as st
from datetime import datetime, date
import pandas as pd
from main import run_model, shap_explain_model, lime_explain_model
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model here
# model = load_your_model()


# Streamlit App UI
def main():
    st.title("Startup Outcome Predictor")

    # Dropdown to select the outcome
    outcome = st.selectbox(
        "Choose the outcome you want to predict:",
        ["acquired", "ipo", "closed", "operating"],
    )

    # Numerical Input
    cb_rank = st.number_input("What is your Crunchbase Ranking?:", value=0, step=1)
    num_rounds = st.number_input(
        "How many funding rounds have you had?:", value=0, step=1
    )
    funding_amt = st.number_input(
        "How much total funding (USD) have you received?:", value=0, step=1
    )

    # Checkboxes
    st.subheader("Check the box if your company has:")
    website = int(st.checkbox("A website?"))
    cs_email = int(st.checkbox("A customer support email?"))
    cs_phone = int(st.checkbox("A customer support phone number?"))
    facebook = int(st.checkbox("A Facebook page?"))
    twitter = int(st.checkbox("A Twitter page?"))
    linkedin = int(st.checkbox("A LinkedIn page?"))
    logo = int(st.checkbox("A logo?"))

    # Date Picker
    found_date = st.date_input(
        "When was your company founded?:",
        max_value=datetime.today(),
    )
    fund_date = st.date_input(
        "What date did you last recieve funding?:",
        max_value=datetime.today(),
    )

    # Dropdown list
    industry = st.selectbox(
        "Which industry is most applicable to your company?",
        ["Energy", "Sustainability"],
    )
    employee_count = st.selectbox(
        "How many employees does your company have?", ["1-10", "11-50", "51-100"]
    )

    # Prediction Button
    if st.button("Predict"):
        # Collect all inputs into a feature array
        days_since_founding = (date.today() - found_date).days
        days_since_last_funding = (date.today() - fund_date).days
        features = {
            "rank": cb_rank,
            "homepage_url": website,
            "num_funding_rounds": num_rounds,
            "total_funding_usd": funding_amt,
            "email": cs_email,
            "phone": cs_phone,
            "facebook_url": facebook,
            "linkedin_url": linkedin,
            "twitter_url": twitter,
            "logo_url": logo,
            "days_since_founding": days_since_founding,
            "days_since_last_funding": days_since_last_funding,
            "category_groups_list_Energy": 0,
            "category_groups_list_Sustainability": 0,
            "employee_count_1-10": 0,
            "employee_count_11-50": 0,
            "employee_count_51-100": 0,
        }
        # convert selections for category and employee count to bool
        features[f"category_groups_list_{industry}"] = 1
        features[f"employee_count_{employee_count}"] = 1

        # convert to dataframe
        features_df = pd.DataFrame(features, index=[0])

        # Run model
        trained_model, X_train, X_test, y_train, y_test = run_model(
            outcome, pd.read_csv("clean_data.csv")
        )

        prediction = trained_model.predict_proba(features_df)
        st.subheader(f"The predicted probability of {outcome} is: {prediction[0][1]}")

        # Explain the model
        st.subheader("LIME Explainer")
        exp = lime_explain_model(trained_model, X_train, X_test, y_train, y_test)
        st.write(exp.as_list())

        st.subheader("SHAP Explainer")
        values = shap_explain_model(trained_model, X_train, X_test, y_train, y_test)
        plt.figure(figsize=(20, 10))
        plt.title("SHAP Feature Importance")
        shap.summary_plot(values, X_test)
        st.pyplot(plt.gcf())


if __name__ == "__main__":
    main()
