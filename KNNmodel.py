import pandas as pd  
import math        
import numpy as np 
from sklearn.model_selection import train_test_split

class KNN:
    def KNN_model(self,Xtrain, ytrain, Xtest):
        #I find it easier to deal with numpy arrays
        Xtrain_np = Xtrain.to_numpy()
        ytrain_np = ytrain.to_numpy()
        Xtest_np = Xtest.to_numpy() #contains one row of data
        
        #calculating the Euclidian distance for one x test data for all train x data
        # mydict[Euclidian distance] = the corresponding y training value (corresponds to the training x data used)
        my_dict = {}
        for i in range(0,len(Xtrain)):
            euclid = np.sqrt(np.sum((Xtest_np - Xtrain_np[i,:])**2))
            my_dict[euclid] = ytrain_np[i]
            
        sorted_dict = dict(sorted(my_dict.items()))

        #finding an appropriate value for k
        k = round(math.sqrt(len(Xtrain)))
        if k%2 == 0:
            k = k + 1
        
        #finding which glass type the testing data is through a vote system
        # only considers the first k entries of the dictionary as it has the lowest distance from the testing data
        num = {1:0,2:0,3:0,4:0,5:0,6:0,7:0}

        for i, (key, val) in enumerate(sorted_dict.items()):
            if i == k + 1:
                break
            elif i < k + 1:
                num[sorted_dict[key]] = num[sorted_dict[key]] + 1
        #finding the highest vote = the type of glass
        pred = max(num, key = num.get)
        return pred




