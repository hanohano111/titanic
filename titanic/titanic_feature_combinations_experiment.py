import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
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

# 获取特征组合
feature_columns = X.columns.tolist()
all_feature_combinations = []
for r in range(1, len(feature_columns) + 1):
    combinations_r = combinations(feature_columns, r)
    all_feature_combinations.extend(combinations_r)

# 训练模型和评估不同特征组合
for feature_combination in all_feature_combinations:
    X_subset = X[list(feature_combination)]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

    # 创建SVM模型
    svm_model = SVC(kernel='linear')  # 使用线性核

    # 在训练集上拟合模型
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred_svm = svm_model.predict(X_test)

    # 计算模型准确率
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"特征组合 {feature_combination} 的SVM模型准确率:", accuracy_svm)
