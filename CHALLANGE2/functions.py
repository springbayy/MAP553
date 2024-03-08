import csv
import xgboost
import numpy as np
from progress.bar import Bar
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import tree
import pylab as P


def n_largest_eigenvectors(matrix, n):
    """
    Compute the n largest eigenvectors of a given matrix.

    Parameters:
    - matrix: 2D numpy array representing the matrix
    - n: number of largest eigenvectors to return

    Returns:
    - eigenvectors: a list of numpy arrays representing the n largest eigenvectors
    """

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the n largest eigenvectors
    n_largest_eigenvectors = sorted_eigenvectors[:, :n]

    return n_largest_eigenvectors

def plotBox(Xdata, ydata, collumn):

    valList=[]

    for i in range(1,8):
        idx=np.where(ydata==i)
        values=Xdata[idx, collumn]
        valList.append(values)
    
    P.figure()

    bp = P.boxplot(valList)

    for i in range(7):
        y = valList[i]
        x = np.random.normal(1+i, 0.04, size=len(y))
        P.plot(x, y, 'r.', alpha=0.2)

    P.show()

def box_plot_with_percentiles(datamatrix,collumn):
    """
    Create a box plot for each column of the given matrix with specified percentiles.

    Parameters:
    - matrix: 2D numpy array
    - percentiles: Tuple or list of percentiles to display (default is 25, 50, 75)

    Returns:
    - None (displays the plot)
    """

    percntiles=(5, 95)
    matrix=np.zeros((2160, 7))
    for i in range(7):
        arr=np.array(range(i*2160, (i+1)*2160))
        matrix[:,i]=datamatrix[arr, collumn]

    # Create a box plot for each column with specified percentiles
    plt.boxplot(matrix, vert=True, patch_artist=True, medianprops={'color': 'black'}, widths=0.7, showfliers=False, whis=percntiles)

    # Set labels and title
    plt.xlabel('Cover Type')
    plt.ylabel('Elevation (m)')
    plt.title('Box Plot of Elevation')

    # Show the plot
    plt.show()
    



def get_dataList(file):

    datamatrix=np.empty((0,56))

    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        data=list(spamreader)  
        datamatrix=np.array(data)

    featuresList=datamatrix[0,:]
    datamatrix=np.delete(datamatrix, 0,0)
    datamatrix = np.asarray(datamatrix, dtype=int)

    #print(np.max(datamatrix[:, 27]))
    #datamatrix=np.delete(datamatrix, 28,1)



    if 'Id' in featuresList:
        idList=datamatrix[:,0]
        datamatrix=np.delete(datamatrix, 0, 1)
    
    coverList=[]
    if 'Cover_Type' in featuresList:
        coverList=datamatrix[:,-1]
        datamatrix=np.delete(datamatrix, -1, 1)
    
    arr=np.array(range(1,41))
    arr=arr*-1
    sparseMatrix=datamatrix[:, arr]
    

    #datamatrix=np.delete(datamatrix, -1*arr, 1)
    
    return datamatrix, coverList, idList

def normalize_data(matrix):
    l=len(matrix[0,:])
    delIndex=0
    for i in range(l):
        col=matrix[:,i]
        col=col-np.mean(col)
        std=np.std(col)
        col=col/(std+0.00001)
        matrix[:,i]=col  
    
    return matrix

def createValidation(idData, Xdata, ydata, p):
    l=len(ydata)
    n=int(l*p)
    idx=np.array(range(l))
    np.random.shuffle(idx)

    idV=idData[idx[0:n]]
    XV=Xdata[:, idx[0:n]]
    yV=ydata[:, idx[0:n]]

    idT=idData[idx[n:]]
    XT=Xdata[:, idx[n:]]
    yT=ydata[:, idx[n:]]

    return idV, XV, yV, idT, XT, yT

def createSubmission(preds, idList):

    np.char.mod('%d', preds)
    np.char.mod('%d', idList)

    datamatrix=np.append(idList[None].T, preds[None].T, axis=1)
  
    descriptors=np.array(['Id', 'Cover_Type'])

    datamatrix=np.append(descriptors[None], datamatrix, axis=0)

    # convert array into dataframe
    DF = pd.DataFrame(datamatrix)
    # save the dataframe as a csv file
    DF.to_csv("data3.csv", header=None, index=None)

    return DF

def variable_importance_RF(B, m, X_train, y_train):
    """
    Inputs:
        B (int): number of trees
        m (int): number of features to be randomly chosen at each split in the tree
        X_train (Nxd matrix): training data
        y_train (N vector): response values
    Outputs:
        variable_importance (d vector): MDA of each predictor variable
    """
    #insert your code here


    importanceList=[]

    x_oob_list=[]
    y_oob_list=[]

    lenght=int(len(X_train))
    treeList=[]

    for i in range(B):
        indexList=np.random.randint(0, lenght, size=lenght)
        bootStrapY=y_train[indexList]
        bootStrapX=X_train[indexList]

        oob_indicies=np.arange(lenght)
        oob_indicies=np.setdiff1d(oob_indicies,indexList)

        x_oob=(X_train[oob_indicies])
        y_oob=(y_train[oob_indicies])
        x_oob_list.append(x_oob)
        y_oob_list.append(y_oob)


        clf=tree.DecisionTreeClassifier(criterion="entropy", max_features=m)
        clf_entropy=clf.fit(bootStrapX, bootStrapY)
        treeList.append(clf_entropy)


        
    for i in range(12):
        acc_ref=0
        acc=0
        if i==10:
            i=np.arange(10,14)
        elif i==11:
            i=np.arange(14,54)

        for j in range(B):
            prediction=treeList[j].predict(x_oob_list[j])
            acc_ref =acc_ref + np.sum(y_oob_list[j]==prediction)/len(prediction)

            featureList=x_oob_list[j][:,i]
            np.random.shuffle(featureList) 
            X_oob_permutation=x_oob_list[j]
            X_oob_permutation[:,i]=featureList

            prediction=treeList[j].predict(X_oob_permutation)
            acc =acc+ np.sum(y_oob_list[j]==prediction)/len(prediction)

        acc_ref=acc_ref/B
        acc=acc/B

        importanceList.append(acc_ref-acc)


    return(np.array(importanceList))


def plotMDA(X_train, y_train, nTrees):
    features=np.array(['Elevation','Aspect','Slope','Horz Dist to Hydro','Vert Dist to Hydro','Horz Dist to Road','Hillshade 9am','Hillshade Noon','Hillshade 3pm','Horz Dist Fire','Wilderness Area','Soil Type'])

    np.random.seed(1)
    mda = variable_importance_RF(nTrees, int(np.round(np.sqrt(56))), X_train, y_train)
    plt.figure()
    plt.bar(np.arange(1,13), mda)
    plt.ylabel('Mean Decrease Accuracy')
    plt.xticks(np.arange(1,13), features, rotation=90)
    plt.show()

def plotOccurance(y_train):
    occuranceList=[]
    typeList=[]
    l=len(y_train)

    for i in range(1, 8):
        arr=np.ones(l)*i
        occ=np.sum(np.equal(y_train, arr))
        occuranceList.append(occ)
        typeList.append(i)
    
    plt.figure()
    plt.bar(np.arange(1,8), occuranceList)
    plt.ylabel('Occurance of Cover Type')
    plt.xticks(np.arange(1,8), i)
    plt.show()
    




class XGB_model:

    def __init__(self) -> None:
        self.bft=xgboost.XGBClassifier(objective='multi:softmax')
        self.preds=None

    def fit_data(self, xdata, ydata):
        le = LabelEncoder()
        ydata = le.fit_transform(ydata)


        self.bft.fit(xdata, ydata)

    def predict(self, data):

        self.preds=self.bft.predict(data)+1

    def normalize_data(self, matrix):
        l=len(matrix[0,:])
        for i in range(l):
            col=matrix[:,i]
            col=col-np.mean(col)
            std=np.std(col)
            col=col/(std+0.00001)
            matrix[:,i]=col    
        
        return matrix