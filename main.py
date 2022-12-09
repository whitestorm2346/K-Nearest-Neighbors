import pandas as pd  # data processing -> CSV file I/O
import numpy as np  # linear algebra
from scipy import spatial
import matplotlib.pyplot as plt  # for data visualization
from collections import Counter  # for most_common
# from sklearn.neighbors import KNeighborsClassifier


def euclidean_distance(x1, x2) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, X) -> list:
        predicted_labels = [self._predict(x) for x in X]

        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.x_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]


# load dataset CSV file
DATA_SET_URL = 'https://raw.githubusercontent.com/whitestorm2346/neural-network/main/train.csv'
df = pd.read_csv(DATA_SET_URL)

# data frame for entity name and ID
df0 = 0

# data frame for attributes
df1 = 0

# convert to 0s and 1s based on the attribute values
df1 = df.iloc[:, [0, 2]]

# merge pd_df0 and pd_df1 data frame
df2 = pd.concat([df0, df1], axis=1, sort=False)

df_array = df2.to_numpy()

carDict = {}

for d in df_array:
    pass

nearest_k = 0  # k個最接近你的鄰居
selected_id = 0

print("")

neighbors = []

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
