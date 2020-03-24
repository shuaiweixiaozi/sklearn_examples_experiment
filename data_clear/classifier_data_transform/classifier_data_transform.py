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