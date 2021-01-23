"""Regression analysis is one of the most important fields in statistics and machine learning.Typically, you need regression to answer whether and how some phenomenon influences the other or how several variables are related.Regression is also useful when you want to forecast a response using a new set of predictors. """



#Import the packages you need.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import seaborn as sns
#Provide data to work with and eventually do appropriate transformations.
df = pd.read_csv('mem.csv')
print(df.head())


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(df[['timestamp', 'storage_used','cpu_allocated','cpu_used','nw_bandwidth','mem_allocated']], df['mem_used'], test_size=0.4, random_state=109)

#Creating the model
logisticRegr = LogisticRegression(C=1)
logisticRegr.fit(X_train, y_train)
y_pred = logisticRegr.predict(X_test)#Saving the Model
pickle_out = open("logisticRegr.pkl", "wb") 
pickle.dump(logisticRegr, pickle_out) 
pickle_out.close()
