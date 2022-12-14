import operator
import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt  # for data visualization

# load dataset CSV file
DATA_SET_URL = 'https://raw.githubusercontent.com/yfujieda/tech-cookbook/master/python/knn-example2/data/car_dataset.csv'
data_set = pd.read_csv(DATA_SET_URL)


def ComputeDistance(a, b):
    dataA = a[1]
    dataB = b[1]

    AttributeDistance = spatial.distance.cosine(dataA, dataB)

    return AttributeDistance


def getNeighbors(id, k):
    global car_dict

    distances = []
    neighbors = []

    for car in car_dict:
        if car != id:
            dist = ComputeDistance(car_dict[id], car_dict[car])
            distances.append((car, dist))

    distances.sort(key=operator.itemgetter(1))

    for x in range(k):
        neighbors.append((distances[x][0], distances[x][1]))

    return neighbors


# dataframe for entity name and ID
cars = data_set.iloc[:, [0, 2]]
# model-index | model-name

# print(pd_df0, end='\n\n\n')

# dataframe for attributes
cars_attributes = data_set.iloc[:, [3, 4, 5, 6, 7, 8]]
# fuel_type | aspiration | num_of_doors | body_style | drive_wheels | engine_location

# print(pd_df1, end='\n\n\n')

# convert to 0s and 1s based on the attribute values
cars_attributes = pd.get_dummies(cars_attributes)
# 將屬性質根據性質轉換為 0 或 1 的表

# print(pd_df1, end='\n\n\n')

# merge pd_df0 and pd_df1 dataframes
cars_info = pd.concat([cars, cars_attributes], axis=1, sort=False)
# 將兩個 dataframe 合併

# print(pd_df2, end='\n\n\n')

cars_info_array = cars_info.to_numpy()
# 將 df物件轉型回 array物件

# print(df_array, end='\n\n\n')

car_dict = {}

for car in cars_info_array:
    index = int(car[0])
    name = car[1]
    attributes = car[2:]
    attributes = map(int, attributes)
    car_dict[index] = (name, np.array(list(attributes)))

k = 10
selected_index = 5

print("selected car: ", car_dict[selected_index][0])  # print model name

neighbors = getNeighbors(selected_index, k)

for neighbor in neighbors:
    model_index = neighbor[0]
    model_name = car_dict[model_index][0]
    model_value = neighbor[1]

    print("%d | %s | %.3f" % (model_index, model_name, model_value))
