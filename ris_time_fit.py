
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)


runtime_mat = sio.loadmat('runtime.mat')
runtime_data = runtime_mat['runtime']

# 存储参数和相应的运行时间
dimension_n = list(range(1, 1201))
runtimes = runtime_data[5,:]

# print(len(dimension_n), len(runtimes))

# 使用numpy进行曲线拟合
# coefficients = np.polyfit(dimension_n, runtimes, 4)  # 4次多项式拟合
# poly = np.poly1d(coefficients)
# fit_values = poly(dimension_n)
# plt.plot(dimension_n, fit_values, color='red',label='fit curve')
#
# plt.plot(dimension_n, runtimes, 'b', label='runtime', alpha=0.5)
# plt.legend()
# plt.title('runtime-dimension')
# plt.xlabel('dimension')
# plt.ylabel('runtime')
# plt.show()

# 定义拟合的阶数
order = 7  # 4次多项式拟合

# 计算数据的一阶差分，即斜率
slope = np.diff(runtimes)

# 定义变化斜率的阈值，根据实际情况调整
threshold = 10

# 找到剧烈变化的位置
change_points = np.where(np.abs(slope) > threshold)[0] + 1

# 添加首尾两个端点
change_points = np.concatenate(([0], change_points, [len(dimension_n)]))

# 进行分段拟合
coefficients = np.zeros((order + 1, len(change_points) - 1))
for i in range(len(change_points) - 1):
    # 选择每个分段的数据
    segment_x = dimension_n[change_points[i]:change_points[i + 1]]
    segment_y = runtimes[change_points[i]:change_points[i + 1]]

    # 多项式拟合
    p = np.polyfit(segment_x, segment_y, order)

    # 保存拟合系数
    coefficients[:, i] = p

# 生成拟合曲线
fit_curve = np.zeros_like(dimension_n, dtype=float)
for i in range(len(change_points) - 1):
    fit_curve[change_points[i]:change_points[i + 1]] = np.polyval(coefficients[:, i],
                                                                  dimension_n[change_points[i]:change_points[i + 1]])

# 画图
# plt.plot(dimension_n, runtimes, 'b', label='runtime', alpha=0.5)
plt.plot(dimension_n, fit_curve, 'r', label='fit curve')
plt.title('runtime-dimension')
plt.xlabel('dimension')
plt.ylabel('runtime')
plt.legend()
plt.grid(True)
plt.show()
