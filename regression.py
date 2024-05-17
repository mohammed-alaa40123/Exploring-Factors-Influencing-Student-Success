from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder as ore
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import streamlit as st    
def denormalize(y):#this is a denormalize function
    u=np.mean(df['G3'])
    sigma=np.std(df['G3'])
    for i in y:
        i=i*sigma+u
    return y


def significance_hypothesis_test(X,y_train,y_pred,coeff):
    N=len(y_train)
    numerator=np.sqrt(mean_squared_error(y_train,y_pred)*(N/(N-2)))
    SE=[]
   
    cols=X.columns.copy()
    for column in cols:
        denomenator=np.sqrt(np.std(X[column])*N)
        SE.append(numerator/denomenator)
    Z_score=[]
    p_score=[]
    new_coeff=[]


    for i in range(len(coeff)):
        Z_score.append(coeff[i]/SE[i])
        p_score.append(2*stats.norm.cdf(-abs(Z_score[i])))
        if(p_score[i]<0.05):
            new_coeff.append((coeff[i],X.columns[i]))
        else:
            new_coeff.append((0,X.columns[i]))
    return new_coeff,p_score,Z_score
    


df = pd.read_csv("cleaned_df.csv")


X = df.drop(columns=["G3"])
y = df['G3']

def is_categorical(df, column):
    return True if df[column].dtype.name == 'category' or df[column].dtype.name == 'object'else False
        
    
def run_regression(columns,model,Normalize=False):
    cat = False # checkif thereis any categorical variables
    for c in columns:
        if is_categorical(df,c):
            cat = True
    X = df[columns]
    y = df['G3']

    # Extract numerical columns
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Extract categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing data
    X_test_scaled = scaler.transform(X_test)

    # Define the transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Scale the numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('label_encoder', LabelEncoder())  # Encode categorical features
    ])

    # Combine the transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_columns),
            ('cat', categorical_transformer, cat_columns)
        ])

    # Define the final pipeline with preprocessing and regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())  # Example: Linear regression model
    ])

    # Now you can fit your pipeline to your data
    pipeline.fit(X_train, y_train)

    # And make predictions
    y_pred = pipeline.predict(X_test_scaled)

        

    # Calculate mean and standard deviation of MSE and R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean MSE: {:.2f}".format(mse))
    st.write("Mean R^2: {:.2f}".format(r2))
        
    

