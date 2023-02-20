#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas
df = pandas.read_csv("C:/Users/Lenovo/Documents/CarPrice_Assignment.csv") 
#Let's predict the price of car according to the enginesize and the horsepower.


# In[17]:


df.head()


# In[18]:


df.describe() #Generate descriptive statistics.

#Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.


# In[19]:


import numpy as np


# In[26]:


df_var = np.std(df["price"])


# In[27]:


print(df_var)


# In[ ]:





# In[28]:


import matplotlib.pyplot as plt
# To visualize the data set we can draw a histogram with the data we collected.

# We will use the Python module Matplotlib to draw a histogram.


# In[33]:


plt.hist(df["price"],10)
plt.show()


# In[36]:


plt.scatter(df["price"],df["enginesize"])
plt.show() # A scatter plot is a diagram where each value in the data set is represented by a dot.


# In[37]:


plt.scatter(df["price"],df["horsepower"])
plt.show()


# In[40]:


# The term regression is used when you try to find the relationship between variables.

# In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.

# Linear regression uses the relationship between the data-points to draw a straight line through all them

# Import scipy and draw the line of Linear Regression :

from scipy import stats
y = df["price"]

x = df["enginesize"]

slope, intercept, r, p, std_err = stats.linregress(x,y) # This is for get some key values of Linear Regression 

def myfunc(x):
    return slope * x + intercept #Create a function that uses the slope and intercept values to return a new value
                                #where on the y-axis the corresponding x value. 

newmodel = list(map(myfunc,x)) # Create new array with the new values of privious function.

plt.scatter(x,y)

plt.plot(x,newmodel) # Draw the line of linear regression
plt.show()


# In[43]:


# Let try the same thing with 'horsepower'

x = df["horsepower"]
y = df["price"]


slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope  * x + intercept

mymodel = list(map(myfunc,x))

plt.scatter(x,y)

plt.plot(x,mymodel)
plt.show()


# In[54]:


# Previously i mentioned about the key vlues of linear regression 
# Now let's look at closely
# R Show relationship between two variable
# The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.
from scipy import stats

x = df["enginesize"]

y = df["price"]


slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)


# In[55]:


df.head()


# In[60]:


# Now we can use the information we have gathered to predict future values.

x = df["horsepower"]
y = df["price"]


slope, intercept, r, p, std_err = stats.linregress(x,y)

def myfunc(x):
    return slope  * x + intercept
future_price  = myfunc(100)

print(round(future_price,2))


# In[64]:


# Now we can use both of them
# Multiple regression is like linear regression, but with more than one independent value, 
# meaning that we try to predict a value based on two or more variables.

from sklearn import linear_model

X = df[['horsepower','enginesize']]

Y = df['price']

regr = linear_model.LinearRegression()

regr.fit(X,Y)

predicted_price = regr.predict([[100,100]])

print(predicted_price)


# In[65]:


# Next step learn about the coefficient which decribe the relationship between unknown variables.
print(regr.coef_) # This show us ,if we increase 'horepower' by 1 ,the price will increase 58.
                    # And if we increase 'enginsize' by 1 ,the price will increase 122.


# In[ ]:




