import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = pd.read_csv("iris.csv", delimiter=",")
columns=["sepal.length","sepal.width","petal.length","petal.width"];
df = pd.DataFrame(iris, columns=columns)
y = iris.variety
# print(df)
# print(y)

# print("\nKiem tra xem du lieu co bi thieu (NULL) khong?")
# print(df.isnull().sum())

# Chia dữ liệu ngẫu nhiên thành 2 tập dữ liệu con:
# training set và test set theo tỷ lệ 70/30
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

nFold=5

# Xây dựng mô hình với k = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Dự đoán nhãn tập kiểm tra
y_pred = model.predict(X_test)

# print(X_test)
# print(y_pred)

# Tính độ chính xác
# scores = cross_val_score(model, df, y, cv=nFold)
# print("Do chinh xac cua mo hinh voi nghi thuc kiem tra %d-fold = %.3f" %
# (nFold, np.mean(scores)))
