import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix(arr_pred, y_test, name):
    y_numpy = y_test.to_numpy()
    #creating the confusion matrices as a numpy array 
    arr = np.zeros((len(np.unique(y_test)) + 2, len(np.unique(y_test))+ 2))
    for i in range(len(arr_pred)):
        arr[arr_pred[i]][y_numpy[i]] = arr[arr_pred[i]][y_numpy[i]] + 1
    
    # remove these rows and columns because there is no class/type 0. We only have classes from 1 - 7.
    arr = np.delete(arr, 0, axis = 0)
    arr = np.delete(arr, 0, axis = 1)
    row = ['1', '2', '3', '4', '5', '6','7']
    col = ['1', '2', '3', '4', '5', '6','7']

    #forming the dataframe and plots for the figures for the confusion matrix
    df = pd.DataFrame(arr, index = row, columns= col)
    plt.title("Your Title")
    plt.figure(figsize=(5,4))
    sns.heatmap(df, annot=True, fmt="g", cmap="Blues", linewidths=0.5)
    plt.title(name)
    plt.xlabel("Actual labels")
    plt.ylabel("predicted labels")
    #plt.show()

    #calculates the accuracy based from the confusion matrix (array)
    sum = 0
    for i in range(len(np.unique(y_test)) + 1):
        sum = sum + arr[i][i]
    
    accuracy = (sum/len(y_test))*100
    print("accuracy:",accuracy,"%")
    
