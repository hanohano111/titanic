import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# 读取数据
titanic = pd.read_csv("train.csv")

# 数据预处理
titanic.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# 准备特征集和标签集
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']

# 随机划分数据集为训练集和测试集（70%训练集，30%测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 不同参数的SVM模型
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    auc_roc_svm = roc_auc_score(y_test, y_pred_svm)

    print(f"\nSVM模型（kernel={kernel}）评估指标：")
    print(f"Precision: {precision_svm}")
    print(f"Recall: {recall_svm}")
    print(f"F1-score: {f1_svm}")
    print(f"AUC-ROC: {auc_roc_svm}")
