import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from gradient import gradient
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
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
    st.subheader("Feature Distributions")
    st.image('Features_Distributions.png')
    fig5 = px.box(df, x='Dalc', y='G3', title='Grades (G3) vs Weekday Alcohol Consumption (Dalc)',
              labels={'Dalc': 'Weekday Alcohol Consumption', 'G3': 'Final Grade'})
    # st.pyplot()
    
    st.plotly_chart(fig5)



# Function for hypothesis results and analysis
def hypothesis_analysis():
    st.header('Hypothesis Results and Analysis')
    st.write('Placeholder for hypothesis results and analysis')
    # st.dataframe(pd.DataFrame())
# Function for regression modeling
def regression_modeling():
    # st.header('Regression Modeling')

    # # Load Boston housing data
    # X, y = df[["avg_Grade","age"]],df["G3"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # best_alpha = st.slider(label="alpha",min_value=0.0,max_value=10.0,step=0.01)
    # # Fit linear regression model
    # lr = Ridge(alpha=best_alpha)
    # lr.fit(X_train, y_train)
    # lr_pred = lr.predict(X_test)
    # lr_mse = mean_squared_error(y_test, lr_pred)

    # # Fit random forest regression model
    

    # st.subheader('Linear Regression')
    # st.write(f'Mean Squared Error: {lr_mse}')
    gradient()
    

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
    "Regression Modeling": regression_modeling,
    "Classification": classification
}

# Render tabs
st.sidebar.title('Navigation')
selected_tab = st.sidebar.radio("Go to", list(tabs.keys()))
tabs[selected_tab]()
