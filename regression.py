from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import shap
import xgboost as xgb 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st    
st.set_option('deprecation.showPyplotGlobalUse', False)


def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", plot_size=(12,12))
    summary_fig, _ = plt.gcf(), plt.gca()
    plt.show()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    st.pyplot(display_shap_summary(shap_values_cat_train, X_train))


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
            new_coeff.append(coeff[i])
        else:
            new_coeff.append("not significant")
    return new_coeff,p_score,Z_score
    


df = pd.read_csv("cleaned_df.csv")
X = df.drop(columns=["G3"])
y = df['G3']
 
    
def run_regression(columns,model):
    X = df[columns]
    y = df['G3']

    # Extract numerical columns
    num_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Extract categorical columns
    cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize preprocessing steps for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Scale the numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('one-hot-encoder', OneHotEncoder())  # Encode categorical features
    ])
    # Create a ColumnTransformer to apply different transformations to numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_columns),
            ('cat', categorical_transformer, cat_columns)
        ])
    

    
    
    # Apply preprocessing steps to training and testing data
        
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    encoded_columns = preprocessor.transformers_[1][1].named_steps['one-hot-encoder'].get_feature_names_out(input_features=cat_columns)
    all_columns = num_columns + list(encoded_columns)
    # Create a DataFrame with preprocessed data and column names
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_columns)
    X_test_preprocessed = preprocessor.transform(X_test)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=all_columns)

    # Apply HyperParameterTuning for KNN Regressor
    if isinstance(model,KNeighborsRegressor):
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
        }
        with st.spinner('Performing hyperparameter tuning...'):
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train_preprocessed_df, y_train)
        model = grid_search.best_estimator_
        st.success('Hyperparameter tuning complete!')
        st.write('Best Parameters:', grid_search.best_params_)
        
        
        
    if isinstance(model,xgb.XGBRegressor):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        }
        # Perform cross-validation with hyperparameter tuning
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        with st.spinner('Performing cross-validation and hyperparameter tuning...'):
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, scoring='neg_mean_squared_error', cv=kfold, verbose=1, random_state=42)
            random_search.fit(X_train_preprocessed_df, y_train)
        model = random_search.best_estimator_
        st.success('Hyperparameter tuning complete!')
        st.write('Best Parameters:', random_search.best_params_)
        
    # Define the final pipeline with preprocessing and regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)  # Example: Linear regression model
    ])
    # Now you can fit your pipeline to your data
    pipeline.fit(X_train, y_train)
    # And make predictions
    y_pred = pipeline.predict(X_test)
    # Calculate mean and standard deviation of MSE and R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("Mean MSE: {:.2f}".format(mse))
    st.write("Mean R^2: {:.2f}".format(r2))
    y_pred_train = pipeline.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    st.write("Train R^2 Score:", train_r2)

    # Evaluate the best model on the test set
    y_pred_test = pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    st.write("Test R^2 Score:", test_r2)

    if not isinstance(pipeline.named_steps['regressor'], xgb.XGBRegressor) and not isinstance(pipeline.named_steps['regressor'], KNeighborsRegressor):
        # Extract coefficients from the linear regression model
        coefficients = pipeline.named_steps['regressor'].coef_

        # Create a DataFrame to display columns and coefficients
        coefficients_df = pd.DataFrame({'Column': all_columns, 'Coefficient': coefficients})
        place = st.empty()
    
        with place:
            st.dataframe(coefficients_df,hide_index=True)   
        hypothesis_button = st.button("Hypothesis Testing on Coeffecints")
        if hypothesis_button:
            significance, p_values,_ = significance_hypothesis_test(X_train_preprocessed_df[all_columns],y_test,y_pred,coefficients)
            coefficients_df = pd.DataFrame({'Column': all_columns, 'Coefficient': coefficients,'P_value':p_values,"significant":significance})
            
            # Displaying the first equation
            st.markdown(r"""
            $$
            \text{Z-test} = \frac{\text{value} - \text{hypothesized value}}{\text{standard error}} = \frac{\hat{a}_m - a_m}{SE_{a_m}} = \frac{\hat{a}_m}{SE_{a_m}}
            $$
            """)

            # Displaying the second equation
            st.markdown(r"""
            $$
            SE_{a_m} = \sqrt{\frac{\sum_{n=1}^{N} (y_n - \hat{y}_n)^2}{N - 2}} \Bigg/ \sqrt{\sum_{n=1}^{N} (x_{n,m} - \bar{x}_m)^2}
            $$
            """)

            with place:
                st.dataframe(coefficients_df,hide_index=True)   
        
    elif not isinstance(pipeline.named_steps['regressor'], KNeighborsRegressor):
        st.subheader("Feature Importance using Shap")
        summary(model, df, X_train_preprocessed_df, X_test_preprocessed_df)
        model.fit(X_train_preprocessed_df,y_train)
        explainer = shap.Explainer(model,X_train_preprocessed_df)
        shap_values= explainer(X_train_preprocessed_df)
        st.pyplot(shap.plots.beeswarm(shap_values))
        