import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

# 步骤1: 使用pandas读取titanic.csv文件中的数据，并查看数据的描述、信息和前几行数据
titanic = pd.read_csv("train.csv")

print("Data Description:")
print(titanic.describe())

print("\nData Information:")
print(titanic.info())

print("\nFirst Few Rows of Data:")
print(titanic.head())

# 步骤2: 特征处理
# 处理缺失值和非数值列转换
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# One-Hot 编码
titanic = pd.get_dummies(titanic, columns=['Sex', 'Pclass', 'Embarked'])

# 连续特征归一化处理
scaler = MinMaxScaler()
titanic['Age'] = scaler.fit_transform(titanic[['Age']])
titanic['Fare'] = scaler.fit_transform(titanic[['Fare']])

#处理数据
#显示出横向完整数据
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


# 显示处理后的数据集
print("\nProcessed Data:")
print(titanic.head())
