#Import the packages you need.
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#get test data from user
dt = int(input("Enter Date:"))
su = int(input("Enter storage_used:"))
ca = int(input("Enter cpu_allocated:"))
cu = int(input("cpu cpu_used:"))
nb = int(input("Enter nw_bandwidth:"))
ma = int(input("memeory allocated:"))

#Applying the model for predictions
pickle_in = open('logisticRegr.pkl', 'rb')
classifier = pickle.load(pickle_in)
prediction = classifier.predict([[dt,su,ca,cu,nb,ma]])
print("Predicted Memory use:",prediction)
