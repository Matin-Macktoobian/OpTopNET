"""
This file prepares the train-validation-test multi-class dataset partitions associated with each each robot of the network.
"""

import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

test_size_ratio = 0.1
validation_size_ratio = 0.2


"""
This function casts the normal index range [1,10] of robots to [0,9] to make dataset compatible with the indexing 
conventions of output layers in Keras models. 
"""
def reindex_dataset(csv_file_name):
    def change(value):
        if value == 1:
            return 0
        elif value == 2:
            return 1
        elif value == 3:
            return 2
        elif value == 4:
            return 3
        elif value == 5:
            return 4
        elif value == 6:
            return 5
        elif value == 7:
            return 6
        elif value == 8:
            return 7
        elif value == 9:
            return 8
        elif value == 10:
            return 9

    df = pd.read_csv(csv_file_name, delimiter=" ")

    for row_index in range(0, 2000, 1):
        for item in range(20, 30, 1):
            df.iat[row_index, item] = change(int(df.iat[row_index, item]))

    df.to_csv("cycle_Topo_dataset_10_new.csv", sep=' ', index=False)

reindex_dataset("cycle_Topo_dataset_10.csv")

df = pd.read_csv("cycle_Topo_dataset_10_new.csv", delimiter=" ")


df_1 = df.drop(["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"], axis=1)
df_2 = df.drop(["C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"], axis=1)
df_3 = df.drop(["C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9", "C10"], axis=1)
df_4 = df.drop(["C1", "C2", "C3", "C5", "C6", "C7", "C8", "C9", "C10"], axis=1)
df_5 = df.drop(["C1", "C2", "C3", "C4", "C6", "C7", "C8", "C9", "C10"], axis=1)
df_6 = df.drop(["C1", "C2", "C3", "C4", "C5", "C7", "C8", "C9", "C10"], axis=1)
df_7 = df.drop(["C1", "C2", "C3", "C4", "C5", "C6", "C8", "C9", "C10"], axis=1)
df_8 = df.drop(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9", "C10"], axis=1)
df_9 = df.drop(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C10"], axis=1)
df_10 = df.drop(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"], axis=1)

def get_df_1():
    return df_1

lb = LabelBinarizer()

df_data = df.loc[:, "X1":"Y10"]

df_1['C1'] = lb.fit_transform(df_1['C1']).tolist()
df_1 = df_1.sample(frac=1).reset_index(drop=True)
df_1_target = df.loc[:, "C1"]

df_2['C2'] = lb.fit_transform(df_2['C2']).tolist()
df_2 = df_2.sample(frac=1).reset_index(drop=True)
df_2_target = df.loc[:, "C2"]

df_3['C3'] = lb.fit_transform(df_3['C3']).tolist()
df_3 = df_3.sample(frac=1).reset_index(drop=True)
df_3_target = df.loc[:, "C3"]

df_4['C4'] = lb.fit_transform(df_4['C4']).tolist()
df_4 = df_4.sample(frac=1).reset_index(drop=True)
df_4_target = df.loc[:, "C4"]

df_5['C5'] = lb.fit_transform(df_5['C5']).tolist()
df_5 = df_5.sample(frac=1).reset_index(drop=True)
df_5_target = df.loc[:, "C5"]

df_6['C6'] = lb.fit_transform(df_6['C6']).tolist()
df_6 = df_6.sample(frac=1).reset_index(drop=True)
df_6_target = df.loc[:, "C6"]

df_7['C7'] = lb.fit_transform(df_7['C7']).tolist()
df_7 = df_7.sample(frac=1).reset_index(drop=True)
df_7_target = df.loc[:, "C7"]

df_8['C8'] = lb.fit_transform(df_8['C8']).tolist()
df_8 = df_8.sample(frac=1).reset_index(drop=True)
df_8_target = df.loc[:, "C8"]

df_9['C9'] = lb.fit_transform(df_9['C9']).tolist()
df_9 = df_9.sample(frac=1).reset_index(drop=True)
df_9_target = df.loc[:, "C9"]

df_10['C10'] = lb.fit_transform(df_10['C10']).tolist()
df_10 = df_10.sample(frac=1).reset_index(drop=True)
df_10_target = df.loc[:, "C10"]

def get_train_valid_test_1():
    X_train_full_1, X_test_1, y_train_full_1, y_test_1 = train_test_split(
        df_data, df_1_target, test_size=test_size_ratio)
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
        X_train_full_1, y_train_full_1, test_size=validation_size_ratio)
    return X_train_1, y_train_1, X_valid_1, y_valid_1, X_test_1, y_test_1

def get_train_valid_test_2():
    X_train_full_2, X_test_2, y_train_full_2, y_test_2 = train_test_split(
        df_data, df_2_target, test_size=test_size_ratio)
    X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(
        X_train_full_2, y_train_full_2, test_size=validation_size_ratio)
    return X_train_2, y_train_2, X_valid_2, y_valid_2, X_test_2, y_test_2

def get_train_valid_test_3():
    X_train_full_3, X_test_3, y_train_full_3, y_test_3 = train_test_split(
        df_data, df_3_target, test_size=test_size_ratio)
    X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(
        X_train_full_3, y_train_full_3, test_size=validation_size_ratio)
    return X_train_3, y_train_3, X_valid_3, y_valid_3, X_test_3, y_test_3

def get_train_valid_test_4():
    X_train_full_4, X_test_4, y_train_full_4, y_test_4 = train_test_split(
        df_data, df_4_target, test_size=test_size_ratio)
    X_train_4, X_valid_4, y_train_4, y_valid_4 = train_test_split(
        X_train_full_4, y_train_full_4, test_size=validation_size_ratio)
    return X_train_4, y_train_4, X_valid_4, y_valid_4, X_test_4, y_test_4

def get_train_valid_test_5():
    X_train_full_5, X_test_5, y_train_full_5, y_test_5 = train_test_split(
        df_data, df_5_target, test_size=test_size_ratio)
    X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(
        X_train_full_5, y_train_full_5, test_size=validation_size_ratio)
    return X_train_5, y_train_5, X_valid_5, y_valid_5, X_test_5, y_test_5

def get_train_valid_test_6():
    X_train_full_6, X_test_6, y_train_full_6, y_test_6 = train_test_split(
        df_data, df_6_target, test_size=test_size_ratio)
    X_train_6, X_valid_6, y_train_6, y_valid_6 = train_test_split(
        X_train_full_6, y_train_full_6, test_size=validation_size_ratio)
    return X_train_6, y_train_6, X_valid_6, y_valid_6, X_test_6, y_test_6

def get_train_valid_test_7():
    X_train_full_7, X_test_7, y_train_full_7, y_test_7 = train_test_split(
        df_data, df_7_target, test_size=test_size_ratio)
    X_train_7, X_valid_7, y_train_7, y_valid_7 = train_test_split(
        X_train_full_7, y_train_full_7, test_size=validation_size_ratio)
    return X_train_7, y_train_7, X_valid_7, y_valid_7, X_test_7, y_test_7

def get_train_valid_test_8():
    X_train_full_8, X_test_8, y_train_full_8, y_test_8 = train_test_split(
        df_data, df_8_target, test_size=test_size_ratio)
    X_train_8, X_valid_8, y_train_8, y_valid_8 = train_test_split(
        X_train_full_8, y_train_full_8, test_size=validation_size_ratio)
    return X_train_8, y_train_8, X_valid_8, y_valid_8, X_test_8, y_test_8

def get_train_valid_test_9():
    X_train_full_9, X_test_9, y_train_full_9, y_test_9 = train_test_split(
        df_data, df_9_target, test_size=test_size_ratio)
    X_train_9, X_valid_9, y_train_9, y_valid_9 = train_test_split(
        X_train_full_9, y_train_full_9, test_size=validation_size_ratio)
    return X_train_9, y_train_9, X_valid_9, y_valid_9, X_test_9, y_test_9

def get_train_valid_test_10():
    X_train_full_10, X_test_10, y_train_full_10, y_test_10 = train_test_split(
        df_data, df_10_target, test_size=test_size_ratio)
    X_train_10, X_valid_10, y_train_10, y_valid_10 = train_test_split(
        X_train_full_10, y_train_full_10, test_size=validation_size_ratio)
    return X_train_10, y_train_10, X_valid_10, y_valid_10, X_test_10, y_test_10
