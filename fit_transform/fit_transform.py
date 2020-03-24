from sklearn.preprocessing import StandardScaler

X_train = [[5,5], [10,10], [15,15], [20,20], [25,25]]
X_test = [[50,50], [100,100], [150,150], [200,200], [250,250]]
# 标准化缩放
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print("X_train_std=", X_train_std)
print("X_test_std=", X_test_std)

# 归一化缩放
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# 在X_train上学习各列特征的参数，并将学习到的参数应用于X_test上，保持训练集、测试集按照统一参数归一化
X_train_mms = mms.fit_transform(X_train)
X_test_mms = mms.transform(X_test)


