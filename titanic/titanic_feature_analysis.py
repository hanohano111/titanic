import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

titanic = pd.read_csv("train.csv")

# 删除非数值型的列（比如 'Name', 'Ticket', 'PassengerId', 'Cabin' 列）
titanic.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

# 处理缺失值和非数值列转换
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna('S')

# 使用LabelEncoder将非数值型的特征转换为数值类型
label_encoder = LabelEncoder()
titanic['Sex'] = label_encoder.fit_transform(titanic['Sex'])
titanic['Embarked'] = label_encoder.fit_transform(titanic['Embarked'])

# 准备数据集（特征和标签）
X = titanic.drop('Survived', axis=1)  # 特征集（去除标签）
y = titanic['Survived']  # 标签集

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Change test_size to 0.3 for 70% training, 30% test

# 创建SVM模型
svm_model = SVC(kernel='linear')  # 使用线性核

# 在训练集上拟合模型
svm_model.fit(X_train, y_train)

# 获取特征系数
feature_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': svm_model.coef_[0]})
feature_coefficients = feature_coefficients.sort_values(by='Coefficient', ascending=False)
print("特征系数:")
print(feature_coefficients)
