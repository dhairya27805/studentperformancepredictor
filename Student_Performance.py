#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import seaborn as sns
import statsmodels.formula.api as smf
import sklearn.linear_model as lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[4]:


df = pd.read_csv("student-mat.csv", sep = ';')
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


X = df.drop('G3', axis=1)
Y = df['G3']


# In[9]:


X = pd.get_dummies(X, drop_first=True)


# In[10]:


alphas = 10**np.linspace(10, -2, 100) * 0.5


# In[13]:


ridgecv=RidgeCV(alphas=alphas)
ridgecv.fit(X,Y)
ridgecv.alpha_


# In[12]:


lassocv=LassoCV(alphas=alphas)
lassocv.fit(X,Y)
lassocv.alpha_


# In[ ]:


fitL=Lasso(alpha=0.10772173450159389)
fitL.fit(X,Y)
print(pd.Series(fitL.coef_,index=X.columns))


# In[14]:


fitR = Ridge(alpha=201.85086292982749)
fitR.fit(X, Y)
print(pd.Series(fitR.coef_,index=X.columns))


# In[ ]:


X_temp = X.astype(float)
X_temp = X_temp.replace([np.inf, -np.inf], np.nan)
X_temp = X_temp.dropna()
X_vif = pd.DataFrame()
X_vif["feature"] = X_temp.columns
X_vif["VIF"] = [variance_inflation_factor(X_temp.values, i) for i in range(X_temp.shape[1])]

print(X_vif.sort_values(by='VIF', ascending=False))


# In[ ]:


drop_cols = [
    'Medu', 'Fedu', 'traveltime', 'health', 'Dalc',
    'address_U', 'famsize_LE3', 'Pstatus_T',
    'Mjob_health', 'Mjob_other', 'Mjob_services',
    'school_MS', 'sex_M', 'internet_yes',
    'higher_yes', 'paid_yes', 'Fjob_other',
    'reason_reputation', 'guardian_other',
    'famsup_yes', 'Fjob_teacher', 'nursery_yes',
    'reason_home', 'activities_yes', 'romantic_yes'
]


# In[ ]:


X = X_temp[['G2', 'G1', 'absences', 'famrel']]


# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# In[ ]:


X_ols = sm.add_constant(X_scaled)
model = sm.OLS(Y, X_ols).fit()
print(model.summary())


# In[ ]:


df1 = X.copy()
df1['G3'] = Y


# In[ ]:


sns.regplot(data = df1, x='G1',y="G3",order = 1)


# In[ ]:


sns.regplot(data = df1, x='G2',y="G3",order = 1)


# In[ ]:


sns.regplot(data = df1, x='absences',y="G3",order = 3)


# In[ ]:


sns.regplot(data = df1, x='famrel',y="G3",order = 1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


# In[ ]:


lr = LinearRegression()
lr.fit(X_train, Y_train)


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


print("MSE:", mean_squared_error(Y_test, y_pred))
print("R^2:", r2_score(Y_test, y_pred))


# In[ ]:


errors = Y_test - y_pred


# In[ ]:


comparison = pd.DataFrame({
    'Actual': Y_test.values,
    'Predicted': y_pred
})

print(comparison.head(10))


# In[ ]:


sns.histplot(errors, kde=True)
plt.title("Distribution of Error")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

