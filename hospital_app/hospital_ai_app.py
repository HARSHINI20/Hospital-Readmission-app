import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import json


# ---------------- Must be first Streamlit command ----------------
st.set_page_config(page_title="Hospital AI Portal", layout="wide")



# Get the current folder path (where hospital_ai_app.py is located)
BASE_DIR = os.path.dirname(__file__)

# ---------------- Load model ----------------
model_path = os.path.join(BASE_DIR, "readmission_model.pkl")
xgb_model = joblib.load(model_path)

# ---------------- Load JSON ----------------
json_path = os.path.join(BASE_DIR, "health_meal_plans.json")
with open(json_path, "r") as f:
    meal_plan = json.load(f)

# ---------------- CSV history ----------------
history_file = os.path.join(BASE_DIR, "risk_prediction_history.csv")

# ---------------- Initialize session ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- Sidebar navigation ----------------
with st.sidebar:
    st.title("🏥 Hospital AI Portal")
    nav = st.radio("Navigation", ["Home", "Readmission Prediction", "Risk History", "Dashboard"])
st.session_state.page = nav

# ---------------- Preprocess user input ----------------
def preprocess_user_input(user_input):
    age_map = {'[0-10]': 5, '[10-20]': 15, '[20-30]':25, '[30-40]':35, '[40-50]':45,
               '[50-60]':55, '[60-70]':65, '[70-80]':75, '[80-90]':85, '[90-100]':95}
    age_numeric = age_map[user_input['age']]
    
    X_user = pd.DataFrame({
        'total_visits':[user_input['total_visits']],
        'age_numeric':[age_numeric],
        'n_medications':[user_input['n_medications']],
        'n_procedures':[user_input['n_procedures']],
        'glucose_test':[user_input['glucose_test']],
        'A1Ctest':[user_input['A1Ctest']],
    })
    
    diag_cols = ['diag_1_Circulatory','diag_1_Diabetes','diag_1_Digestive',
                 'diag_1_Injury','diag_1_Missing','diag_1_Musculoskeletal',
                 'diag_1_Other','diag_1_Respiratory']
    for col in diag_cols:
        X_user[col] = 0
    diag_name = f"diag_1_{user_input['diag_1']}"
    if diag_name in diag_cols:
        X_user[diag_name] = 1
    
    return X_user

# ---------------- Home Page ----------------
if st.session_state.page == "Home":
    st.markdown("<h2 style='color:blue;'>Welcome to Hospital AI Portal</h2>", unsafe_allow_html=True)
    st.write("Use the sidebar to navigate between features.")

# ---------------- Readmission Prediction Page ----------------
elif st.session_state.page == "Readmission Prediction":
    st.header("Predict Patient Readmission")
    
    # Inputs
    age = st.selectbox("Age", ['[0-10]','[10-20]','[20-30]','[30-40]','[40-50]','[50-60]','[60-70]','[70-80]','[80-90]','[90-100]'])
    total_visits = st.number_input("Total Visits", min_value=0, step=1)
    n_medications = st.number_input("Number of Medications", min_value=0, step=1)
    n_procedures = st.number_input("Number of Procedures", min_value=0, step=1)
    diag_1 = st.selectbox("Primary Diagnosis", list(meal_plan.keys()))
    diag_2 = st.selectbox("Secondary Diagnosis", list(meal_plan.keys()))
    diag_3 = st.selectbox("Additional Diagnosis", list(meal_plan.keys()))
    
    # Show "High, Medium, No" to user
    glucose_test_ui = st.selectbox("Glucose Test", ["High", "Medium", "No"])
    A1Ctest_ui = st.selectbox("A1C Test", ["High", "Medium", "No"])

    # Map UI values to model-compatible yes/no (1/0)
    def map_to_model(value):
        return 0 if value == "No" else 1   # High/Medium -> Yes, No -> No
    
    glucose_test = map_to_model(glucose_test_ui)
    A1Ctest = map_to_model(A1Ctest_ui)
    
    if st.button("Predict Readmission"):
        user_input = {
            'age': age,
            'total_visits': total_visits,
            'n_medications': n_medications,
            'n_procedures': n_procedures,
            'diag_1': diag_1,
            'glucose_test': glucose_test,
            'A1Ctest': A1Ctest
        }
        X_user = preprocess_user_input(user_input)

        # Predict
        pred = xgb_model.predict(X_user)[0]
        prob = xgb_model.predict_proba(X_user)[0][1]*100

        # Risk category
        if prob < 40:
            risk_cat = "Low"
        elif prob < 70:
            risk_cat = "Medium"
        else:
            risk_cat = "High"

        recommendation = {
            "Low": "Maintain regular check-ups and a healthy lifestyle.",
            "Medium": "Consult your doctor for preventive measures.",
            "High": "Immediate medical attention recommended and monitor closely."
        }

        # Show prediction
        st.success(f"Predicted Readmission: {'Yes' if pred==1 else 'No'}")
        st.info(f"Risk Probability: {prob:.2f}% ({risk_cat})")
        st.warning(f"Recommendation: {recommendation[risk_cat]}")

        # Save history
        history_entry = {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': age,
            'total_visits': total_visits,
            'n_medications': n_medications,
            'n_procedures': n_procedures,
            'diag_1': diag_1,
            'diag_2': diag_2,
            'diag_3': diag_3,
            'glucose_test': glucose_test,
            'A1Ctest': A1Ctest,
            'predicted_readmission': 'Yes' if pred==1 else 'No',
            'risk_probability': prob,
            'risk_category': risk_cat
        }

        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df = pd.concat([df, pd.DataFrame([history_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([history_entry])
        df.to_csv(history_file, index=False)

        # ---------------- Food suggestions ----------------
        st.subheader("🍽️ Recommended Food Plans for Selected Conditions")
        conditions = [diag_1, diag_2, diag_3]
        for cond in conditions:
            st.markdown(f"<div style='background-color:#cce5ff; padding:5px; border-radius:5px;'><b>{cond}</b></div>", unsafe_allow_html=True)
            if cond in meal_plan:
                for meal_type in ["Breakfast","Lunch","Dinner","Snacks"]:
                    st.markdown(f"<div style='color:#1f77b4; font-weight:bold;'>{meal_type}</div>", unsafe_allow_html=True)
                    for item in meal_plan[cond][meal_type]:
                        st.markdown(f"- {item['name']} ({item['calories']} cal, {item['category']})")
                # Tips box
                st.markdown(f"<div style='background-color:#fff3cd; padding:5px; border-radius:5px;'>Tips: {meal_plan[cond].get('tips','')}</div>", unsafe_allow_html=True)
            else:
                st.write("No meal plan available.")

# ---------------- Risk History ----------------
elif st.session_state.page == "Risk History":
    st.header("Patient Risk History")
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        if not df.empty:
            df = df.sort_values(by="risk_probability", ascending=False).reset_index(drop=True)
            
            # Color coding
            def color_risk(row):
                if row['risk_category']=="High":
                    return ['background-color: #ff9999']*len(row)
                elif row['risk_category']=="Medium":
                    return ['background-color: #fff79a']*len(row)
                else:
                    return ['background-color: #b3ffb3']*len(row)
            
            st.dataframe(df.style.apply(color_risk, axis=1), height=600)
        else:
            st.info("No history available.")
    else:
        st.info("No history available.")

# ---------------- Dashboard ----------------
elif st.session_state.page == "Dashboard":
    st.title("Patient Readmission Dashboard")

    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
    else:
        st.warning("No data available for dashboard.")
        df = pd.DataFrame()

    expected_cols = ['id','timestamp','age','total_visits','n_medications','n_procedures',
                     'diag_1','diag_2','diag_3','glucose_test','A1Ctest','predicted_readmission',
                     'risk_probability','risk_category']

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        df = pd.DataFrame(columns=expected_cols)

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Sidebar filters
        risk_filter = st.sidebar.multiselect("Select Risk Category", options=df['risk_category'].unique(), default=df['risk_category'].unique())
        age_filter = st.sidebar.multiselect("Select Age Group", options=df['age'].unique(), default=df['age'].unique())
        filtered_df = df[df['risk_category'].isin(risk_filter) & df['age'].isin(age_filter)]

        # Top 10
        st.subheader("Top 10 Patients by Risk Probability")
        if not filtered_df.empty:
            top10 = filtered_df.sort_values(by='risk_probability', ascending=False).head(10)
            st.dataframe(top10[['id','age','risk_probability','risk_category']])
        else:
            st.info("No data to display.")

        # Pie chart
        st.subheader("Risk Category Distribution")
        if not filtered_df.empty:
            pie_data = filtered_df['risk_category'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(4,2))
            ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            ax1.axis('equal')
            st.pyplot(fig1)
        else:
            st.info("No data to display.")

        # Risk Trend Over Time
        st.subheader("Risk Probability Trend Over Time")
        if not filtered_df.empty and 'timestamp' in filtered_df.columns:
            trend_data = filtered_df.groupby('timestamp')['risk_probability'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(4,2))
            sns.lineplot(data=trend_data, x='timestamp', y='risk_probability', marker='o', ax=ax2, color='#1f77b4')
            ax2.set_ylabel("Average Risk Probability")
            ax2.set_xlabel("Timestamp")
            plt.xticks(rotation=30)
            st.pyplot(fig2)
            
        else:
            st.info("No data to display.")

        # High/Medium/Low Count Bar
        st.subheader("High/Medium/Low Count")
        if not filtered_df.empty:
            bar_data = filtered_df['risk_category'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(4,2))
            sns.barplot(x=bar_data.index, y=bar_data.values, palette="pastel", ax=ax3)
            ax3.set_ylabel("Count")
            ax3.set_xlabel("Risk Category")
            st.pyplot(fig3)
        else:
            st.info("No data to display.")
