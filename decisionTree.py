import pandas as pd  
import math        
import numpy as np 
from sklearn.model_selection import train_test_split

class Treenode:
    def __init__(self, right = None, left = None, col = None, element = None, value = None):
        self.right = right
        self.left = left
        self.value = value
        self.col = col
        self.element = element      

#entropy is a measure of the impurities (the different classes) are in the data
def entropy(dfy): #finds the entropy of a set of glass types
    sum = 0
    dfy_numpy = dfy.to_numpy().ravel().astype(np.int64)   # to make a 1D flat array 
    occurence = np.bincount(dfy_numpy)

    for x in occurence:
        entro  = x / len(dfy)
        if entro != 0:
            sum = sum - (entro*np.log2(entro)) 
    return sum

def split_entropy(X_col, thres, y):
    child_l = []
    child_r = []
    i = 0
    for x in X_col:
        if x < thres: ## we split the data into two groups by using a threshold to divide the X column into two groups
            child_l.append(y.iloc[i])
        else:
            child_r.append(y.iloc[i])
        i+=1
    child_l = pd.DataFrame({'y': child_l})
    child_r = pd.DataFrame({'y': child_r})
    return child_l, child_r 

def split(X_col, thres):
    index_l = []
    index_r = []
    i = 0
    for x in X_col:
        if x < thres: ## we split the data into two groups by using a threshold to divide the X column into two groups
            index_l.append(i)
        else:
            index_r.append(i)
        i+=1
    index_l = pd.DataFrame({'i': index_l})
    index_r = pd.DataFrame({'i': index_r})
    return index_l, index_r # we have two split functions as we want to return different things from both

def information_gains(dfy, X_col, thres):
    parent_entropy = entropy(dfy)

    child_l, child_r = split_entropy(X_col, thres, dfy)
    entropy_l = entropy(child_l) 
    entropy_r = entropy(child_r)
    #information gains equation:
    split_entro = ((len(child_l)*entropy_l) + (len(child_r)*entropy_r))/len(dfy)
    return parent_entropy - split_entro # this is the information gain which was determined by the threshold value

def where_to_split(X, dfy): #X we want to only use the columns necessary to find the threshold - therefore don't want index or y values, X is a dataframe
    ult_gain = 0
    ult_element = None
    for i in range(X.shape[1]): 
        col = X.iloc[:,i].values
        for element in col:
            info_gain = information_gains(dfy, col, element) # finds the optimum information gain in the dataset. This allows the data which is split to have the lowest possible entropy
            if info_gain >= ult_gain:
                ult_gain = info_gain
                ult_idx = i # which is an index number
                ult_element = element
    return ult_idx, ult_element # optimum position to split the data


def tree(X,dfy, depth):
    maxDepth = 10
    #these conditions stop the tree from being huge - this helps with time taken for training
    if len(np.unique(dfy)) <= 1 or depth == maxDepth or len(dfy) < 2:    
        occurances = np.bincount(dfy)
        max = np.max(occurances)
        index = np.where(occurances == max)
        return Treenode(value= index) #this is a leaf which holds the value of the class type which appeared the most in the split

    idx_col, idx_element = where_to_split(X,dfy)
    index_l, index_r = split(X.iloc[:,idx_col],idx_element)
    numpy_l = index_l.iloc[:,0].to_numpy()
    numpy_r = index_r.iloc[:,0].to_numpy()
    
    #finding the left and right nodes of the parent node
    left_node = tree(X.iloc[numpy_l], dfy.iloc[numpy_l], depth+1)### make x and y the same type
    right_node = tree(X.iloc[numpy_r], dfy.iloc[numpy_r], depth+1)
    return Treenode(right_node, left_node, idx_col, idx_element) 

#this traverse through the tree to find the leaf node. Once found, it return the node.value which will be the predicted class for the testing data
def search(x, node):
    if node.left == None and node.right == None:
       return node.value
    
    if x.iloc[node.col] <= node.element:
        return search(x, node.left)
    return search(x,node.right)






    
    
