import pandas as pd  
import math        
import numpy as np 
from sklearn.model_selection import train_test_split
from KNNmodel import KNN
import decisionTree
import matrix
import naive
import SVM
import time
#start = time.time()

df = pd.read_csv("C:/Users/20men/vs/glass+identification/glass.data", header=None, delimiter=',')
column_names = ['index', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type_Glass'] 
df.columns = column_names
dfX = df.iloc[0:214,1:10] ##SHOULDN'T HAVE COL 0 AND 10!!
dfY = df['Type_Glass']

X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.2, random_state=38967472)

# ## code to run KNN model
arr_knn = []
knn = KNN()
for i in range(0, len(X_test)): #allows each data in X_test to go through the KNN model to find the estimated glass group
    arr_knn.append(knn.KNN_model(X_train,y_train,X_test.iloc[i,:]))
knn_mod = "KNN model"
print("KNN model accuracy:")
matrix.confusion_matrix(arr_knn,y_test, knn_mod)
#end = time.time()


# ## code to run decision tree model
root = decisionTree.Treenode(None, None, None, None, None)
root = decisionTree.tree(dfX,dfY,0)
arr_dt = []
for i in range(X_test.shape[0]): # calls the function for each data in the testing dataset
    val = decisionTree.search(X_test.iloc[i,:], root)
    arr_dt.append(val[0][0])
dt_mod = "decision tree model"
print("decision trees model accuracy:")
matrix.confusion_matrix(arr_dt,y_test, dt_mod)
# # end = time.time()

# #code to run naive bayes
arr_nb = []
for i in range(X_test.shape[0]):
    arr_nb.append(naive.naive_bayes(X_train, y_train, X_test.iloc[i,:]))
nb_mod = "naive bayes model"
print("naive bayes model accuracy:")
matrix.confusion_matrix(arr_nb,y_test, nb_mod)
# end = time.time()


# #code to run support vector machine
# as the model compares one class to all classes, we need to call the method 7 times 
w_and_b_arr = [] # contains 7 sets of w and b
for i in range(1,8):
    tuple = SVM.fit(X_train, y_train,i)
    w_and_b_arr.append(tuple)

svm_arr= []
for i in range(X_test.shape[0]): # calls the function for each data in the testing dataset
    pred = SVM.prediction(X_test.iloc[i,:],w_and_b_arr)
    svm_arr.append(pred)
svm_mod = "SVM model"
print("SVM model accuracy:")
matrix.confusion_matrix(svm_arr,y_test, svm_mod)
# end = time.time()

## used for taking the time for training and testing
# interval = end - start
# print("interval:",interval,"seconds")
