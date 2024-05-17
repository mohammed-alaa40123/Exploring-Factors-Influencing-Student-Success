import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import streamlit as st
df = pd.read_csv("cleaned_df.csv")
numerical_columns = df.select_dtypes(include=['int64','Float64'])
categorical_columns = [c for c  in df.columns if c not in numerical_columns]
binary_categorical_columns=[c for c in categorical_columns if len(df[c].unique()) == 2]

def hypothesis_test_means(column):  
    cat1 = df[column].unique()[0]
    cat2 = df[column].unique()[1]
    
    df1 = df[df[column] == cat1]
    df2 = df[df[column] == cat2]
    
    X0 = np.mean(df1["G3"])
    X1 = np.mean(df2["G3"])
    X_bar = X0 - X1
    
    Sigma1 = np.std(df1["G3"])
    Sigma2 = np.std(df2["G3"])
    
    Sig_1 = (Sigma1**2)/len(df1)
    Sig_2 = (Sigma2**2)/len(df2)
    SE = np.sqrt(Sig_1+Sig_2)
    
    Z_score = X_bar/SE  
    p_value = 2 * stats.norm.cdf(-abs(Z_score))
    alpha = 0.05

    if p_value < alpha:
        st.write(f"Reject the null hypothesis. The means are significantly different (p-value = {p_value:.04f})")
        hypothesis_rejected = True
    else:
        st.write(f"Fail to reject the null hypothesis. The means are not significantly different (p-value = {p_value:.04f})")
        hypothesis_rejected = False
    
    # Plotting the hypothesis test result
    plt.figure(figsize=(8, 6))
    plt.bar([cat1, cat2], [X0, X1], color=['blue', 'orange'])
    plt.xlabel(column)
    plt.ylabel('Mean Grade (G3)')
    plt.title('Mean Grade (G3) by ' + column)
    plt.xticks([cat1, cat2])
    plt.axhline(np.mean(df["G3"]), color='red', linestyle='--', label='Overall Mean')
    plt.legend()
    
    return plt

