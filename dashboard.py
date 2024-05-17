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
    st.caption("The pie chart illustrates the distribution of grades for students who go out. The largest segment represents average grades 33.1%, followed by high grades 23.7%. Additionally, there are moderate 22.4%, very high 6.24%, and low 14.6% grade categories. Overall, it suggests that most students who go out fall within the average to high grade range.")
    fig3 = eda_functions.AvgGradeBySchool()
    st.divider()
    st.plotly_chart(fig3)
    fig4 = eda_functions.WeeklyAlcohol()
    st.plotly_chart(fig4)
    st.image('AvgFinal_Grade_Gender_School.png')
    fig5 = eda_functions.FinalGrades_AddressType_Desire()
    st.plotly_chart(fig5)
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
    st.write("Null Hypothesis (H0): The mean final grade does not significantly differ between the two populations categorized by the binary feature.")
    st.write("Alternative Hypothesis (H1): The mean final grade significantly differs between the two populations categorized by the binary feature.")  

    feature = st.selectbox("feature",binary_categorical_columns)
    fig = hypothesis_test_means(feature)
    st.pyplot(fig)
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
    pass
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
