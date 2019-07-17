
# coding: utf-8

# In[4]:

import sys
import pickle
from sklearn.externals import joblib


# In[10]:

clf =  joblib.load('iris.pkl')

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]


# In[11]:

n  = clf.predict([[arg1,arg2,arg3,arg4]])
print(n)


# In[ ]:



