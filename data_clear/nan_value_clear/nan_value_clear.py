from io import StringIO
import pandas as pd

csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              0.0,11.0,12.0,'''
csv_data = pd.read_csv(StringIO(csv_data))

# 使用isnull方法返回一个值为布尔类型的DataFrame，判断每个元素是否缺失，如果缺失为True
# 然后使用sum方法，得到DataFrame中每一列的缺失值个数
print(csv_data.isnull().sum())


# 去除含有nan元素的行记录
print(csv_data.dropna())

# 去除含有nan元素的列记录
print(csv_data.dropna(axis=1))

# 只去掉那些所有值为nan的行
print(csv_data.dropna(how='all'))

# 去掉那些非缺失值小于4个的行
print(csv_data.dropna(thresh=4))

# 去掉那些在特定列出现nan的行
print(csv_data.dropna(subset=['C']))

# 使用均值替代缺失值
from sklearn.preprocessing import Imputer
# axis=1: 计算每个样本的所有特征的平均值。
# strategy：取值包括median、most_frequent. most_frequent对于处理分类数据类型的缺失值很有用。
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(csv_data)
imputed_data = imr.transform(csv_data)
print(imputed_data)