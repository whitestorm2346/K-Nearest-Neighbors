import operator
import pandas as pd
import numpy as np
from scipy import spatial

def ComputeDistance(a,b):
    return

def getNeighbors(,):
    return

# load dataset CSV file
url = 'https://raw.githubusercontent.com/whitestorm2346/neural-network/main/train.csv'
pd_df = pd.read_csv(url)

# data frame for entity name and ID
pd_df0 = 

# data frame for attributes
pd_df1 = 

# convert to 0s and 1s based on the attribute values
pd_df1 = pd_df.iloc[:, [0, 2]]

# merge pd_df0 and pd_df1 data frame
pd_df2 = pd.concat([pd_df0, pd_df1], axis=1, sort=False)

df_array = pd_df2.to_numpy()

carDict = {}

for d in df_array:
    pass

K = 
selectedID = 

print("", )

neighbors = getNeighbors(selectedID, K)

for neighbor in neighbors:
    print(
        str(neighbor[0]) + " | " 
        + carDict[neighbor[0]][0] + " | " 
        + str(neighbor[1])
    )

# https://godamber.blogspot.com/2019/07/pythoncolab.html(資料集讀取方式，若同學使用colab以外的編譯器需自行搜尋CSV檔案讀取方法)
# https://ithelp.ithome.com.tw/articles/10269826?sc=iThelpR(簡易KNN(有使用套件))
# https://www.w3schools.com/python/python_ml_knn.asp(簡易KNN(有使用套件，且有圖示))
# https://www.kaggle.com/code/prashant111/knn-classifier-tutorial/notebook(比較正規的KNN資料分析，較難)