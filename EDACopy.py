
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")


# ## import Data

# In[2]:


df = pd.read_csv("Data/student-por.csv",sep=";")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[8]:


df.duplicated().sum()


# We Decided that our target will be the grade for 3rd year

# In[9]:


df["avg_Grade"] = (df["G1"]+df["G2"]) /2


# In[10]:


df.drop(columns=["G1","G2"],inplace=True)


# In[11]:


df.head()


# In[12]:


numerical_columns = df.select_dtypes(include=['int64','Float64']).columns
numerical_columns


# ## Features Distributions

# In[13]:

fig = plt.figure(figsize=(20, 20))

for i in range(len(numerical_columns)):
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.7, wspace=0.4, hspace=0.4)
    plt.subplot(4, 4, i + 1)
    plt.title(f"{numerical_columns[i]} Distribution",
              fontdict={'fontsize': 20, 'fontweight': 'bold'})
    sns.histplot(df[numerical_columns[i]], bins='rice', kde=True)  # Use square root estimator
    plt.tight_layout()

    plt.savefig('Features_Distributions.png')

plt.show()

#plt.show()


# ### Target ralationship with other features 

# In[14]:


cols = list(df.columns)
cols.remove('G3')
#for i in range(3,len(cols),3):
   # sns.pairplot(df,x_vars=cols[i-3:i],y_vars=["G3"])
        


# In[15]:


numerical_columns_list = numerical_columns

fig = plt.figure(figsize=(20, 20))

for i in range(len(numerical_columns_list)):
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.7, wspace=0.4, hspace=0.4)
    plt.subplot(4, 4, i + 1)
    plt.title(f"{numerical_columns_list[i]} Boxplot")
    sns.boxplot(df[numerical_columns_list[i]],orient="v")

#plt.show()


# There exists outliers in absences, failures, farmel, Dalc, travelyime, studytime, avg_grade and age

# ## Outliers Removal

# In[16]:


df["failures"].value_counts()


# ![image.png](attachment:image.png)

# We will keep the outliers in failures otherwise it will all be zeros

# In[17]:


for col in numerical_columns_list:
    if col == "failures":
        continue
    Q1 = df[col].quantile(.25)
    Q3 = df[col].quantile(.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)] #masking


# In[18]:


# numerical_columns_list = df.select_dtypes(include=['int64', 'float64']).columns

fig = plt.figure(figsize=(20, 20))

for i in range(len(numerical_columns_list)):
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.7, wspace=0.4, hspace=0.4)
    plt.subplot(4, 4, i + 1)
    plt.title(f"{numerical_columns_list[i]} Boxplot")
    sns.boxplot(df[numerical_columns_list[i]],orient="v")

#plt.show()


# In[19]:


df.info()


# In[20]:


df.to_csv("cleaned_df.csv",index=False)


# In[21]:


df_vis = pd.read_csv("cleaned_df.csv")


# In[22]:


sex_labels = {'M': 'Male', "F": 'Female'}
df_vis['sex_label'] = df_vis['sex'].map(sex_labels)

class_distribution = df_vis['sex_label'].value_counts()

def sex_dist():
    # Create a pie chart using Plotly Express
    fig = px.pie(names=class_distribution.index, values=class_distribution.values,
                title='Distribution of sex Class')
    return fig

# Display the pie chart
#ig.show()


# In[23]:


import pandas as pd
import plotly.express as px
# Assume df is your DataFrame containing the data
goout_labels = {1: 'Low', 2: 'Moderate', 3: 'Average', 4: 'High', 5: 'Very High'}
df_vis['goout_label'] = df_vis['goout'].map(goout_labels)

# Calculate the distribution of grades for students who go out and those who don't
goout_distribution = df_vis['goout_label'].value_counts()

def grades_dist():
    
    # Create pie charts for the distribution of grades
    fig_goout = px.pie(names=goout_distribution.index, values=goout_distribution.values,
                    title='Distribution of Grades for Students Who Go Out')

    return fig_goout
# Display the pie charts
#fig_goout.show()


# In[24]:


df_vis.columns


# In[25]:

def grades_distOut():
    
        # Create pie charts for the distribution of grades
    fig_goout = px.bar(x=goout_distribution.index, y=goout_distribution.values,
                    title='Distribution of Grades for Students Who Go Out',
                    labels={'x': 'Grades', 'y': 'Count'},color=goout_distribution.index)
    return fig_goout

# In[26]:


#fig_goout.show()


# In[27]:


import pandas as pd
import plotly.express as px

# Assume df is your DataFrame containing the data
def AvgGradeBySchool():
    # Calculate the average grades for students from each school
    average_grades_by_school = df.groupby('school')['G3'].mean().reset_index()

    # Create a bar chart for average grades by school
    fig = px.bar(average_grades_by_school, x='school', y='G3', 
                title='Average Grades by School',
                labels={'school': 'School', 'G3': 'Average Grades'},color="school")
    return fig
# Display the bar chart
#fig.show()


# In[28]:

def WeeklyAlcohol():
    # Create a box plot of final grades by weekday alcohol consumption
    fig5 = px.box(df, x='Dalc', y='G3', title='Grades (G3) vs Weekday Alcohol Consumption (Dalc)',
                labels={'Dalc': 'Weekday Alcohol Consumption', 'G3': 'Final Grade'})
#fig5.show()
    return fig5

# In[29]:


    # Create a bar plot of average final grades
plt.figure(figsize=(10, 6))
sns.barplot(x='school', y='G3', data=df, hue='sex', ci=None)
plt.title('Average Final Grades (G3) by School and Gender')
plt.xlabel('School')
plt.ylabel('Average Final Grade (G3)')
#plt.show()
plt.savefig('AvgFinal_Grade_Gender_School.png')

# In[30]:

def FinalGrades_AddressType_Desire():
    fig2 = px.box(df, x='address', y='G3', color='higher',
              title='Box Plot of Final Grades (G3) by Address Type and Desire for Higher Education',
              labels={'address': 'Address Type', 'G3': 'Final Grade (G3)'})
    return fig2
#fig2.show()


# In[31]:

def FinalGrades_Internet_Romantic():

    fig1 = px.violin(df, x='internet', y='G3', color='romantic', box=True,
                 title='Distribution of Final Grades (G3) by Internet Access and Romantic Relationship',
                 labels={'internet': 'Internet Access', 'G3': 'Final Grade (G3)'})
    return fig1
    #fig1.show()

