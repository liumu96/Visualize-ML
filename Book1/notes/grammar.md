# API Quick Search

## Matplotlib

| api | desc |
| --- | ---- |
|     |      |

## Numpy

| api                       | desc                                                          |
| ------------------------- | ------------------------------------------------------------- |
| `numpy.linalg.cholesky()` | 计算 Cholesky 分解                                            |
| `numpy.linalg.dot()`      | 计算向量的点积                                                |
| `numpy.linalg.eig()`      | 计算矩阵的特征值和特征向量                                    |
| `numpy.linalg.inv()`      | 计算矩阵的逆                                                  |
| `numpy.linalg.lstsq()`    | 求最小二乘解                                                  |
| `numpy.linalg.norm()`     | 计算向量的范数                                                |
| `numpy.linalg.pinv()`     | 计算矩阵的 Moore-Penrose 伪逆                                 |
| `numpy.linalg.solve()`    | 求解线性方程组                                                |
| `numpy.linalg.svd()`      | 计算奇异值分解                                                |
| `numpy.average()`         | 计算平均值                                                    |
| `numpy.cov()`             | 计算协方差矩阵                                                |
| `numpy.diag()`            | 以一维数组的形式返回方阵的对角线元素,或将一维数组转换成对角阵 |
| `numpy.einsum()`          | 爱因斯坦求和约定                                              |
| `numpy.stack()`           | 将矩阵叠加                                                    |
| `numpy.sum()`             | 求和                                                          |

## Pandas

| api                                | desc                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `pandas.DataFrame()`               | 创建 Pandas 数据帧                                                              |
| `pandas.DataFrame.add_prefix()`    | 给 DataFrame 的列标签添加前缀                                                   |
| `pandas.DataFrame.add_suffix()`    | 给 DataFrame 的列标签添加后缀                                                   |
| `pandas.DataFrame.axes`            | 同时获得数据帧的行标签、列标签                                                  |
| `pandas.DataFrame.columns`         | 查询数据帧的列标签                                                              |
| `pandas.DataFrame.corr()`          | 计算 DataFrame 中列之间 Pearson 相关系数 (样本)                                 |
| `pandas.DataFrame.count()`         | 返回数据帧每列 (默认 axis=0) 非缺失值数量                                       |
| `pandas.DataFrame.cov()`           | 计算 DataFrame 中列之间的协方差矩阵 (样本)                                      |
| `pandas.DataFrame.describe()`      | 计算 DataFrame 中数值列的基本描述统计信息，如平均值、标准差、分位数等           |
| `pandas.DataFrame.drop()`          | 用于从 DataFrame 中删除指定的行或列                                             |
| `pandas.DataFrame.head()`          | 用于查看数据帧的前几行数据，默认情况下，返回数据帧的前 5 行                     |
| `pandas.DataFrame.iterrows()`      | 遍历 DataFrame 的行                                                             |
| `pandas.dataframe.iloc()`          | 通过整数索引来选择 DataFrame 的行和列的索引器                                   |
| `pandas.DataFrame.index`           | 查询数据帧的行标签                                                              |
| `pandas.DataFrame.info`            | 获取关于数据帧摘要信息                                                          |
| `pandas.DataFrame.isnull()`        | 用于检查 DataFrame 中的每个元素是否为缺失值 NaN                                 |
| `pandas.DataFrame.iteritems()`     | 遍历 DataFrame 的列                                                             |
| `pandas.DataFrame.kurt()`          | 计算 DataFrame 中列的峰度 (四阶矩)                                              |
| `pandas.DataFrame.kurtosis()`      | 计算 DataFrame 中列的峰度 (四阶矩)                                              |
| `pandas.dataframe.loc()`           | 通过标签索引来选择 DataFrame 的行和列的索引器                                   |
| `pandas.DataFrame.max()`           | 计算 DataFrame 中每列的最大值                                                   |
| `pandas.DataFrame.mean()`          | 计算 DataFrame 中每列的平均值                                                   |
| `pandas.DataFrame.median()`        | 计算 DataFrame 中每列的中位数                                                   |
| `pandas.DataFrame.min()`           | 计算 DataFrame 中每列的最小值                                                   |
| `pandas.DataFrame.mode()`          | 计算 DataFrame 中每列的数                                                       |
| `pandas.DataFrame.nunique()`       | 计算数据帧中每一列的独特值数量                                                  |
| `pandas.DataFrame.quantile()`      | 计算 DataFrame 中每列的指定分位数值，如四分位数、特定百分位等                   |
| `pandas.DataFrame.rank()`          | 计算 DataFrame 中每列元素的排序排名                                             |
| `pandas.DataFrame.reindex()`       | 用于重新排序 DataFrame 的列标签                                                 |
| `pandas.DataFrame.rename()`        | 对 DataFrame 的索引标签、列标签或者它们的组合进行重命名                         |
| `pandas.DataFrame.reset_index()`   | 将 DataFrame 的行标签重置为默认的整数索引，默认并将原来的行标签转换为 新的一列  |
| `pandas.DataFrame.set_axis()`      | 重新设置 DataFrame 的行或列标签                                                 |
| `pandas.DataFrame.set_index()`     | 改变 DataFrame 的索引结构                                                       |
| `pandas.DataFrame.shape`           | 返回一个元组，其中包含数据帧的行数、列数                                        |
| `pandas.DataFrame.size`            | 用于返回数据帧中元素，即数据单元格总数                                          |
| `pandas.DataFrame.skew()`          | 计算 DataFrame 中列的偏度 (三阶矩)                                              |
| `pandas.DataFrame.sort_index()`    | 按照索引的升序或降序对 DataFrame 进行重新排序，默认 axis = 0                    |
| `pandas.DataFrame.std()`           | 计算 DataFrame 中列的标准差 (样本)                                              |
| `pandas.DataFrame.sum()`           | 计算 DataFrame 中每列元素的总和                                                 |
| `pandas.DataFrame.tail()`          | 用于查看数据帧的后几行数据，默认情况下，返回数据帧的后 5 行                     |
| `pandas.DataFrame.to_csv()`        | 将 DataFrame 数据保存为 CSV 格式文件                                            |
| `pandas.DataFrame.to_string()`     | 将 DataFrame 数据转换为字符串格式                                               |
| `pandas.DataFrame.values`          | 返回数据帧中的实际数据部分作为一个多维 NumPy 数组                               |
| `pandas.DataFrame.var()`           | 计算 DataFrame 中列的方差 (样本)                                                |
| `pandas.Series()`                  | 创建 Pandas Series                                                              |
| `pandas.DataFrame.plot()`          | 绘制线图                                                                        |
| `pandas.DataFrame.plot.area()`     | 绘制面积图                                                                      |
| `pandas.DataFrame.plot.bar()`      | 绘制面积图                                                                      |
| `pandas.DataFrame.plot.barh()`     | 绘制水平柱状图                                                                  |
| `pandas.DataFrame.plot.box()`      | 绘制箱型图                                                                      |
| `pandas.DataFrame.plot.density()`  | 绘制 KDE 线图                                                                   |
| `pandas.DataFrame.plot.hexbin()`   | 绘制六边形图                                                                    |
| `pandas.DataFrame.plot.hist()`     | 绘制直方图                                                                      |
| `pandas.DataFrame.plot.kde()`      | 绘制 KDE 线图                                                                   |
| `pandas.DataFrame.plot.line()`     | 绘制线图                                                                        |
| `pandas.DataFrame.plot.pie()`      | 绘制饼图                                                                        |
| `pandas.DataFrame.plot.scatter()`  | 绘制散点图                                                                      |
| `pandas.plotting.scatter_matrix()` | 成对散点图矩阵                                                                  |
| `pandas.DataFrame.isin()`          | 用于检查 DataFrame 中的元素是否在给定的值序列中                                 |
| `pandas.DataFrame.query()`         | 筛选和过滤 DataFrame 数据的方法                                                 |
| `pandas.DataFrame.where()`         | 在 DataFrame 中根据条件对元素进行筛选和替换的方法                               |
| `pandas.MultiIndex.from_arrays()`  | 用于从多个数组创建多级索引的方法                                                |
| `pandas.MultiIndex.from_frame()`   | 用于从 DataFrame 创建多级索引的方法                                             |
| `pandas.MultiIndex.from_product()` | 用于从多个可迭代对象的笛卡尔积创建多级索引的方法                                |
| `pandas.MultiIndex.from_tuples()`  | 用于从元组列表创建多级索引的方法                                                |
| `pandas.concat()`                  | 将多个数据帧在特定轴 (行、列) 方向进行拼接                                      |
| `pandas.DataFrame.apply()`         | 将一个自定义函数或者 lambda 函数应用到数据帧的行或列上，实现数据的转换和处理    |
| `pandas.DataFrame.apply()`         | 删除数据帧特定列                                                                |
| `pandas.DataFrame.groupby()`       | 在分组后的数据上执行聚合、转换和其他操作，从而对数据进行更深入的分析和处理      |
| `pandas.DataFrame.join()`          | 将两个数据集按照索引或指定列进行合并                                            |
| `pandas.DataFrame.merge()`         | 按照指定的列标签或索引进行数据库风格的合并                                      |
| `pandas.DataFrame.pivot()`         | 用于将数据透视成新的行和列形式的函数                                            |
| `pandas.DataFrame.stack()`         | 将 DataFrame 中的列转换为多级索引的行形式的函数                                 |
| `pandas.DataFrame.unstack()`       | 将 DataFrame 中的多级索引行转换为列形式的函数                                   |
| `pandas.melt()`                    | 将宽格式数据转换为长格式数据的函数，将多个列“融化”成一列                        |
| `pandas.pivot_table()`             | 根据指定的索引和列对数据进行透视，并使用聚合函数合并重复值的函数                |
| `pandas.wide_to_long()`            | 将宽格式数据转换为长格式数据的函数，类似于 melt()，但可以处理多个标识符列和前缀 |

## Seaborn

| api                      | desc                    |
| ------------------------ | ----------------------- |
| `seaborn.heatmap()`      | 绘制热图                |
| `seaborn.load_dataset()` | 加载 Seaborn 示例数据集 |

![Alt text](../assets/image-26.png)
![Alt text](../assets/image-27.png)
![Alt text](../assets/image-35.png)
![Alt text](../assets/image-41.png)
![Alt text](../assets/image-48.png)
![Alt text](../assets/image-50.png)
![Alt text](../assets/image-52.png)
![Alt text](../assets/image-54.png)
![Alt text](../assets/image-57.png)
![Alt text](../assets/image-59.png)
![Alt text](../assets/image-61.png)
![Alt text](../assets/image-64.png)
![Alt text](../assets/image-65.png)
![Alt text](../assets/image-67.png)
![Alt text](../assets/image-69.png)
![Alt text](../assets/image-70.png)
![Alt text](../assets/image-71.png)
![Alt text](../assets/image-72.png)
