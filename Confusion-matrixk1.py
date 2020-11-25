#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[3]:


cell_df = pd.read_csv("C:/Users/sades/OneDrive/Desktop/crowd-control/detect_count_knn1.csv")
#cell_df.head(10)


# In[4]:


col = cell_df['diff']
cate=np.asarray(col).T
#cate
tp = cell_df['actual']
#np.asarray(tp).T


# In[5]:


#get the uique values
#np.unique(tp)
np.unique(cate)


# In[6]:


print('TN')
tp_zero =tp[tp==0]
len(tp_zero)


# In[7]:


npos = cate[cate > 0]
print('FP')
np.sum(npos)


# In[8]:


nneg = cate[cate < 0]
#print(nneg)
#len(nneg)
print('FN')
np.absolute(np.sum(nneg))


# In[9]:


tp=tp[tp>0]
print('TP value')
np.sum(tp)


# In[10]:


nzer0 = cate[cate == 0]
#print(nzer0)
len(nzer0)


# In[17]:


cm = [[36,4437],[0,1495]]

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Confusion Matrix K =1 ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# In[12]:


TN=36
FP=4437
FN=0
TP=1495


# In[13]:


#Accuracy = TP+TN/TP+FP+FN+TN
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Accuracy


# In[14]:


#Precision = TP/TP+FP
Precision = TP/(TP+FP)
Precision


# In[15]:


#Recall = TP/TP+FN
Recall = TP/(TP+FN)
Recall


# In[16]:


#F1 Score = 2*(Recall * Precision) / (Recall + Precision)
F1score = 2*(Recall * Precision) / (Recall + Precision)
F1score


# In[ ]:





# In[ ]:




