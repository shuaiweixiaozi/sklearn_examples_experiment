import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']


# 定义映射函数，映射有序类别变量
size_mapping = {'XL': 3, 'L': 2, 'M':1}
df['size'] = df['size'].map(size_mapping)
# print(df)
# 如果还想将整型变量转换回原来的字符串表示，定义一个反映射字典
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# 对类别进行编码
import numpy as np
# class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# sklearn中提供的LabelEncoder类实现类别变量的转化
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
# fit_transform方法是fit和transform方法的合并
df['classlabel'] = class_le.fit_transform(df['classlabel'])
# df['color'] = class_le.fit_transform(df['color'])
print(df)
# 利用inverse_transform方法得到原始的字符串类型
# print(class_le.inverse_transform(df['classlabel']))

# 使用one_hot处理无须离散特征
# from sklearn.preprocessing import OneHotEncoder
# # 在初始化OneHotEncode时，通过categorical_features参数设置要进行独热编码的列。
# ohe = OneHotEncoder(categorical_features=[0])
# # OneHotEncoder的transform方法默认返回稀疏矩阵，所以我们调用toarray()方法将稀疏矩阵转为一般矩阵。
# # 也可以在初始化OneHotEncoder时通过参数sparse=False来设置返回一般矩阵。
# print(ohe.fit_transform(df).toarray())

# 使用pandas中的get_dummies方法来创建哑特征，get_dummies默认会对DataFrame中所有字符串类型的列进行独热编码
print(pd.get_dummies(df[['price', 'color', 'size']]))

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