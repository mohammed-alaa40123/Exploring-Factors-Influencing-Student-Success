import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from gradient import gradient
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import EDACopy as eda_functions
from hypothesis import *
from regression import *

st.set_page_config(layout="wide")

# Load sample datasets
df = pd.read_csv("cleaned_df.csv")
# iris = load_iris()
def data_description():

    data_description1 = """## Attributes

1. **school**: student's school (binary: 'GP' for Gabriel Pereira or 'MS' for Mousinho da Silveira)
2. **sex**: student's sex (binary: 'F' for female or 'M' for male)
3. **age**: student's age (numeric: from 15 to 22)
4. **address**: student's home address type (binary: 'U' for urban or 'R' for rural)
5. **famsize**: family size (binary: 'LE3' for less than or equal to 3 or 'GT3' for greater than 3)
6. **Pstatus**: parent's cohabitation status (binary: 'T' for living together or 'A' for apart)
7. **Medu**: mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education, or 4 – higher education)
8. **Fedu**: father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education, or 4 – higher education)
9. **Mjob**: mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10. **Fjob**: father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11. **reason**: reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12. **guardian**: student's guardian (nominal: 'mother', 'father' or 'other')
13. **traveltime**: home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14. **studytime**: weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15. **failures**: number of past class failures (numeric: n if 1<=n<3, else 4)
16. **schoolsup**: extra educational support (binary: yes or no)
17. **famsup**: family educational support (binary: yes or no)
18. **paid**: extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19. **activities**: extra-curricular activities (binary: yes or no)
20. **nursery**: attended nursery school (binary: yes or no)
21. **higher**: wants to take higher education (binary: yes or no)
22. **internet**: Internet access at home (binary: yes or no)
23. **romantic**: with a romantic relationship (binary: yes or no)
24. **famrel**: quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25. **freetime**: free time after school (numeric: from 1 - very low to 5 - very high)
26. **goout**: going out with friends (numeric: from 1 - very low to 5 - very high)
27. **Dalc**: workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28. **Walc**: weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29. **health**: current health status (numeric: from 1 - very bad to 5 - very good)
30. **absences**: number of school absences (numeric: from 0 to 93)

### Target Variable

31. **G1**: first period grade (numeric: from 0 to 20)
32. **G2**: second period grade (numeric: from 0 to 20)
33. **G3**: final grade (numeric: from 0 to 20, output target)

### Citation

P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
"""

# Function for data description
    st.header('Data Description')
    st.write(data_description1)

# Function for exploratory data analysis
def eda():
    st.header('Exploratory Data Analysis')
    st.write("After importing the data, we preformed a computation to get Average Grades from Column G1 and G2 before droping them then getting the feature distribution.")
    st.subheader("Feature Distributions")
    st.image('Features_Distributions.png')
    st.write("The histograms reveal that the majority of students are between 15 and 18 years old, typically spend 2 hours or less on study time, and have few absences and no academic failures. Parental education levels are relatively even, with a slight preference for mid-level education. Most students have short travel times and rate their family relationships and health positively. While workday alcohol consumption is low, weekend consumption is moderate for most. Free time and social activities are fairly balanced, with a slight inclination towards higher levels of going out. Grades follow a normal distribution, with most students achieving middle-range grades. This suggests typical student behavior and performance, with balanced lifestyles and moderate academic achievement.")
    st.divider()
    st.subheader("Outliers")
    st.write("After seeing the feature distibution we checked for outliers like absences, failures, farmel, Dalc, travelyime, studytime, avg_grade and age which we then removed.")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.subheader("Before Outliers Removal")
        st.image("StudytimeBefore.png")
        st.image("TraveltimeBefore.png")
        st.image("FreetimeBefore.png")
    with col2:
        st.subheader("After Outliers Removal")
        st.image("StudytimeAfter.png")
        st.image("TraveltimeAfter.png")
        st.image("FreetimeAfter.png")
    st.divider()
    st.subheader("Distributions")
    fig1 = eda_functions.sex_dist()
    st.plotly_chart(fig1)
    st.caption("The pie chart illustrates the distribution of students by gender in the dataset. It shows that 59.4% of the students are female and 40.6% are male.")
    fig2 = eda_functions.grades_dist()
    st.plotly_chart(fig2)
    st.caption("This suggests that going out habits may not necessarily have a negative impact on academic performance, as a substantial portion of students who go out achieve average to high grades.")
    st.divider()
    fig3 = eda_functions.AvgGradeBySchool()
    st.plotly_chart(fig3)
    st.caption("The bar chart illustrates the average grades for students from each school. It shows that students from Gabriel Pereira (GP) have higher average grades than students from Mousinho da Silveira (MS).")
    fig4 = eda_functions.WeeklyAlcohol()
    st.plotly_chart(fig4)
    st.caption("The box plot illustrates the distribution of final grades by weekday alcohol consumption. It shows that students with very low weekday alcohol consumption have a wider range of higher grades, while students with very high weekday alcohol consumption have a varying range of lower grades.")
    st.divider()
    st.image('AvgFinal_Grade_Gender_School.png')
    st.caption("The bar chart illustrates the average final grades with female students from Gabriel Pereira (GP) having the highest average final grades then both males in Gabriel Pereira (GP) and Mousinho da Silveira (MS) combined.")
    fig5 = eda_functions.FinalGrades_AddressType_Desire()
    st.plotly_chart(fig5)
    st.caption("The box plot illustrates the distribution of final grades by address type and desire for higher education. It shows that students living in urban areas who desire higher education have higher final grades than students living in rural areas.")
    st.divider()
    fig6 = eda_functions.FinalGrades_Internet_Romantic()
    st.plotly_chart(fig6)
    st.subheader("Internet Access and Grades:")
    st.write("Students with internet access tend to exhibit a broader spectrum of grades, spanning both higher and lower scores, in comparison to their counterparts without internet access. Interestingly, irrespective of their romantic status, students lacking internet access demonstrate relatively consistent grade distributions.")
    st.write("However, a notable observation emerges among students with internet access: there is a discernible increase in grade variability. This suggests that internet access may play a pivotal role in influencing the diversity of academic performance.")
    st.subheader("Romantic Relationship and Grades:")
    st.write("Being in a romantic relationship does not significantly affect the grade distribution. There is no clear pattern indicating that romantic status impacts final grades.")



# Function for hypothesis results and analysis
def hypothesis_analysis():
    st.header('Hypothesis Results and Analysis')
    st.write("")
    
    st.write("Z-Test for Difference in Means of Two Populations:")

    st.latex(r"""
    \mu = \mu_0 - \mu_1
    """)

    st.latex(r"""
    \sigma = \sqrt{\frac{\sigma_0^2}{n_0} + \frac{\sigma_1^2}{n_1}}
    """)


    st.write("The test point is:")
    st.latex(r"""
     \bar{x} = \bar{x}_0 - \bar{x}_1
    """)
    st.write("Z-Score is:")
    st.latex(r"""
    \text{Z-score = } \frac{\bar{x} - \mu}{\sigma}
    """)

    st.write("Null Hypothesis (H0): The mean final grade does not significantly differ between the two populations categorized by the binary feature.")
    st.write("Alternative Hypothesis (H1): The mean final grade significantly differs between the two populations categorized by the binary feature.")  

    feature = st.selectbox("feature",binary_categorical_columns)
    fig = hypothesis_test_means(feature)
    st.pyplot(fig)

    st.subheader("Heatmap for binary categorical features that were found to be significant:")
    st.image("correlation1.png")

    st.subheader("Heatmap for binary categorical features that were found to be non significant:")
    st.image("correlation2.png")

    
# Function for regression modeling
def regression_modeling():
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb
    st.title("Regression: Build your own model")


    models = [LinearRegression(),Ridge(),xgb.XGBRegressor(),KNeighborsRegressor()]
    st.subheader("Choose the features")
    st.write("Hint: choose Avg_grade")
    columns = st.multiselect("features",X.columns)
    st.subheader("Choose Your Model")
    model = st.selectbox("Select a model",models)
    if isinstance(model,Ridge):
        alpha = st.slider("Choose the alpha",min_value=0.01,max_value=1.0,step=0.01)
        model=Ridge(alpha=alpha)
    if model and columns:
        run_regression(columns,model)

    
    


# Function for classification
def classification():
    st.header('Classification')

    classification = st.selectbox("Select a classification", ["Binary Classification", "Multi-Class Classification"])
    if classification == "Binary Classification":
        st.write("Binary Classification")
        model_binary = st.selectbox("Select a model", ["Logistic Classification", "Naive Bayes Classification", "K Nearest Neighbors"])
        if model_binary == "Logistic Classification":
            binary_Logistic()


        elif model_binary == "Naive Bayes Classification":
            binary_Naive_Bayes()


        elif model_binary == "K Nearest Neighbors":
            binary_knn()



    elif classification == "Multi-Class Classification":
        model_many = st.selectbox("Select a model", ["Logistic Classification", "Naive Bayes Classification", "K Nearest Neighbors"])
        if model_many == "Logistic Classification":
            multi_Logistic()


        elif model_many == "Naive Bayes Classification":
            multi_Naive_Bayes()


        elif model_many == "K Nearest Neighbors":
             multi_knn()


    pass


def multi_Logistic():
            test_accuracy = 0.6881720430107527
            train_accuracy = 0.7580645161290323

            classification_report_data = {
                "Class": ["A", "B", "C", "D", "F"],
                "Precision": [0.88, 0.50, 0.67, 0.57, 0.74],
                "Recall": [0.88, 0.33, 0.84, 0.25, 0.88],
                "F1-Score": [0.88, 0.40, 0.74, 0.35, 0.81],
                "Support": [8, 12, 31, 16, 26]
            }

            overall_stats_data = {
                "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
                "Value": [0.69, 0.67, 0.64, 0.63, 0.67, 0.69, 0.66]
            }

            # Creating DataFrames
            classification_report_df = pd.DataFrame(classification_report_data)
            overall_stats_df = pd.DataFrame(overall_stats_data)

            # Title
            st.title("Model Performance Metrics")

            # Accuracy Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Accuracy")
                st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")
                st.metric(label="Train Accuracy", value=f"{train_accuracy:.2f}")
            with col2:
                st.image("Logistic_Classification_Multi.png")

            # Classification Report
            st.subheader("Classification Report")
            st.dataframe(classification_report_df, hide_index=True, use_container_width=True)

            # Overall Statistics
            st.subheader("Overall Statistics")
            st.dataframe(overall_stats_df, hide_index=True, use_container_width=True)

def multi_Naive_Bayes():
    # Data for model performance
            test_accuracy = 0.3870967741935484
            train_accuracy = 0.41397849462365593

            classification_report_data = {
                "Class": ["A", "B", "C", "D", "F"],
                "Precision": [0.00, 0.21, 0.57, 0.25, 0.79],
                "Recall": [0.00, 1.00, 0.13, 0.06, 0.73],
                "F1-Score": [0.00, 0.34, 0.21, 0.10, 0.76],
                "Support": [8, 12, 31, 16, 26]
            }

            overall_stats_data = {
                "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
                "Value": [0.39, 0.36, 0.38, 0.28, 0.48, 0.39, 0.34]
            }

            # Creating DataFrames
            classification_report_df = pd.DataFrame(classification_report_data)
            overall_stats_df = pd.DataFrame(overall_stats_data)

            # Title
            st.title("Model Performance Metrics")

            # Accuracy Metrics
            
            st.subheader("Accuracy")
            st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")
            st.metric(label="Train Accuracy", value=f"{train_accuracy:.2f}")
        

            # Classification Report
            st.subheader("Classification Report")
            st.dataframe(classification_report_df,hide_index=True, use_container_width=True)

            # Overall Statistics
            st.subheader("Overall Statistics")
            st.dataframe(overall_stats_df,hide_index=True, use_container_width=True)
           

def multi_knn():
            
            test_accuracy = 0.7204301075268817
            train_accuracy = 0.7634408602150538

            classification_report_data = {
                "Class": ["A", "B", "C", "D", "F"],
                "Precision": [0.88, 0.64, 0.72, 0.54, 0.79],
                "Recall": [0.88, 0.58, 0.74, 0.44, 0.88],
                "F1-Score": [0.88, 0.61, 0.73, 0.48, 0.84],
                "Support": [8, 12, 31, 16, 26]
            }

            overall_stats_data = {
                "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
                "Value": [0.72, 0.71, 0.70, 0.71, 0.71, 0.72, 0.71]
            }

            # Creating DataFrames
            classification_report_df = pd.DataFrame(classification_report_data)
            overall_stats_df = pd.DataFrame(overall_stats_data)

            # Title
            st.title("Model Performance Metrics")

            # Accuracy Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Accuracy")
                st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")
                st.metric(label="Train Accuracy", value=f"{train_accuracy:.2f}")
            with col2:
                st.image("k_nearest_neighbors_Multi.png")

            # Classification Report
            st.subheader("Classification Report")
            st.dataframe(classification_report_df, hide_index=True,use_container_width=True)

            # Overall Statistics
            st.subheader("Overall Statistics")
            st.dataframe(overall_stats_df,hide_index=True, use_container_width=True)


def binary_Logistic():
        test_accuracy = 0.9247311827956989
        train_accuracy = 0.9381720430107527

        classification_report_model_data = {
            "Class": ["0", "1"],
            "Precision": [0.83, 0.94],
            "Recall": [0.67, 0.97],
            "F1-Score": [0.74, 0.96],
            "Support": [15, 78]
        }

        overall_stats_model_data = {
            "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", 
                    "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
            "Value": [0.92, 0.89, 0.82, 0.85, 0.92, 0.92, 0.92]
        }

        # Creating DataFrames
        classification_report_df = pd.DataFrame(classification_report_model_data)
        overall_stats_df = pd.DataFrame(overall_stats_model_data)

        # Title
        st.title("Model Performance Metrics")

        # Accuracy Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy")
            st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")
            st.metric(label="Train Accuracy", value=f"{train_accuracy:.2f}")
        with col2:
            st.image("Logistic_Classification_Binary.png")

        # Classification Report
        st.subheader("Classification Report")
        st.dataframe(classification_report_df, hide_index=True, use_container_width=True)

        # Overall Statistics
        st.subheader("Overall Statistics")
        st.dataframe(overall_stats_df, hide_index=True, use_container_width=True)


def binary_Naive_Bayes():
        test_accuracy_model = 0.9032258064516129
        train_accuracy_model = 0.9032258064516129

        classification_report_model_data = {
            "Class": ["0", "1"],
            "Precision": [0.67, 0.96],
            "Recall": [0.80, 0.92],
            "F1-Score": [0.73, 0.94],
            "Support": [15, 78]
        }

        overall_stats_model_data = {
            "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", 
                    "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
            "Value": [0.90, 0.81, 0.86, 0.83, 0.91, 0.90, 0.91]
        }


        # Creating DataFrames
        classification_report_df = pd.DataFrame(classification_report_model_data)
        overall_stats_df = pd.DataFrame(overall_stats_model_data)

        # Title
        st.title("Model Performance Metrics")

        # Accuracy Metrics
        st.subheader("Accuracy")
        st.metric(label="Test Accuracy", value=f"{test_accuracy_model:.2f}")
        st.metric(label="Train Accuracy", value=f"{train_accuracy_model:.2f}")
        # Classification Report
        st.subheader("Classification Report")
        st.dataframe(classification_report_df, hide_index=True, use_container_width=True)

        # Overall Statistics
        st.subheader("Overall Statistics")
        st.dataframe(overall_stats_df, hide_index=True, use_container_width=True)

def binary_knn():
        best_test_accuracy = 0.946236559139785
        best_train_accuracy = 0.9946236559139785

        classification_report_best_data = {
            "Class": ["0", "1"],
            "Precision": [0.92, 0.95],
            "Recall": [0.73, 0.99],
            "F1-Score": [0.81, 0.97],
            "Support": [15, 78]
        }

        overall_stats_best_data = {
            "Metric": ["Accuracy", "Macro Avg Precision", "Macro Avg Recall", "Macro Avg F1-Score", 
                    "Weighted Avg Precision", "Weighted Avg Recall", "Weighted Avg F1-Score"],
            "Value": [0.95, 0.93, 0.86, 0.89, 0.95, 0.95, 0.94]
        }


        # Creating DataFrames
        classification_report_df = pd.DataFrame(classification_report_best_data)
        overall_stats_df = pd.DataFrame(overall_stats_best_data)

        # Title
        st.title("Model Performance Metrics")

        # Accuracy Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy")
            st.metric(label="Test Accuracy", value=f"{best_test_accuracy:.2f}")
            st.metric(label="Train Accuracy", value=f"{best_train_accuracy:.2f}")
        with col2:
            st.image("k_nearest_neighbors_Binary.png")
        # Classification Report
        st.subheader("Classification Report")
        st.dataframe(classification_report_df, hide_index=True, use_container_width=True)

        # Overall Statistics
        st.subheader("Overall Statistics")
        st.dataframe(overall_stats_df, hide_index=True, use_container_width=True)


# # Function for classification

#     st.header('Classification')

#     # Load Iris dataset
#     X, y = load_iris(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train a simple classifier
#     from sklearn.svm import SVC
#     clf = SVC(kernel='linear', C=1.0)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     st.write(f'Accuracy: {accuracy}')

# Create tabs
tabs = {
    "Data Description": data_description,
    "EDA": eda,
    "Hypothesis Analysis": hypothesis_analysis,
    "Gradient Descent for Regression":gradient,
    "Regression Modeling": regression_modeling,
    "Classification": classification
}

# Render tabs
st.sidebar.title('Navigation')

selected_tab = st.sidebar.radio("Go to", list(tabs.keys()))
tabs[selected_tab]()
