# Quick API Search

## Matplotlib

| API                               | Desc                                                                 |
| --------------------------------- | -------------------------------------------------------------------- |
| `matplotlib.pyplot.rcParams`      | 获取或设置全局绘图参数的默认值，如图形尺寸、字体大小、线条样式等     |
| `matplotlib.gridspec.GridSpec()`  | 创建和配置复杂的子图网格布局，以便在一个图形窗口中放置多个子图       |
| `matplotlib.gridspec.SubplotSpec` | 用于定义和控制子图在网格布局中的位置和大小                           |
| `matplotlib.pyplot.figure()`      | 创建一个新的图形窗口或图表对象，以便在其上进行绘图操作               |
| `matplotlib.pyplot.subplot()`     | 用于在当前图形窗口中创建一个子图，并定位该子图在整个图形窗口中的位置 |
| `matplotlib.pyplot.subplots()`    | 一次性创建一个包含多个子图的图形窗口，并返回一个包含子图对象的元组   |
| `matplotlib.pyplot.contour()`     | 绘制等高线图                                                         |
| `matplotlib.pyplot.contourf()`    | 绘制填充等高线图                                                     |
| `matplotlib.pyplot.scatter()`     | 绘制散点图                                                           |
| `matplotlib.pyplot.grid()`        | 在当前图表中添加网格线                                               |
| `matplotlib.pyplot.plot()`        | 绘制折线图                                                           |
| `matplotlib.pyplot.title()`       | 设置当前图表的标题，等价于 `ax.set_title()`                          |
| `matplotlib.pyplot.xlabel()`      | 设置当前图表 y 轴的标签，等价于 `ax.set_xlabel()`                    |
| `matplotlib.pyplot.xlim()`        | 设置当前图表 y 轴显示范围，等价于 `ax.set_xlim()`                    |
| `matplotlib.pyplot.xticks()`      | 设置当前图表 x 轴刻度位置，等价于 `ax.set_xticks()`                  |
| `matplotlib.pyplot.ylabel()`      | 设置当前图表 y 轴的标签，等价于 `ax.set_ylabel()`                    |
| `matplotlib.pyplot.ylim()`        | 设置当前图表 y 轴显示范围，等价于 `ax.set_ylim()`                    |
| `matplotlib.pyplot.ytick()`       | 设置当前图表 y 轴刻度位置，等价于 `ax.set_yticks()`                  |

## Numpy

| API                                  | Desc                                             |
| ------------------------------------ | ------------------------------------------------ |
| `numpy.linspace()`                   | 在指定的间隔内,返回固定步长的数据                |
| `numpy.meshgrid(`                    | 产生网格化数据                                   |
| `numpy.random.multivariate_normal()` | 用于生成多元正态分布的随机样本                   |
| `numpy.vstack()`                     | 返回竖直堆叠后的数组                             |
| `numpy.arange`                       | 创建一个具有指定范围、间隔和数据类型的等间隔数组 |
| `numpy.exp()`                        | 计算给定数组中每个元素的 e 的指数值              |
| `numpy.sin()`                        | 用于计算给定弧度数组中每个元素的正弦值           |
| ``                                   |                                                  |

## Scipy

| API                          | Desc           |
| ---------------------------- | -------------- |
| `scipy.stats.gaussian_kde()` | 高斯核密度估计 |

## Statsmodels

| API                                             | Desc         |
| ----------------------------------------------- | ------------ |
| `statsmodels.api.nonparametric.KDEUnivariate()` | 构造一元 KDE |
