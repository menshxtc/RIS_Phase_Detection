import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KernelDensity

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 假设a是合法数据，b是非法数据
# np.random.seed(42)
# a = np.random.rand(10100, 1200)
# b = np.random.rand(10100, 1200) + 2  # 在b中加上一些偏移，模拟非法数据

# 定义矩阵数据处理函数
def process_data(matrix_data):
    # 提取偶数列数据作为信道信息进行训练
    num_col = matrix_data.shape[1]
    col_step = np.arange(1, num_col, 2)
    fea_data = matrix_data[:, col_step]

    # 转置矩阵
    trans_fea_data = np.transpose(fea_data)

    return trans_fea_data

# 从矩阵中导入数据进行
# 2米有ris的合法数据
mat_file_2_on='RSSTxRIS2off.mat'
mat_data_2_on = sio.loadmat(mat_file_2_on)
ris_2_on = mat_data_2_on['RSSTxRIS2off']
a = process_data(ris_2_on)

# 2米无ris的非法数据
mat_file_2_off='RSSTxRIS2none.mat'
mat_data_2_off = sio.loadmat(mat_file_2_off)
ris_2_off = mat_data_2_off['RSSTxRIS2none']
b = process_data(ris_2_off)


# 使用PCA将1200维特征降维至1维
pca = PCA(n_components=1)
a_pca = pca.fit_transform(a)

# 使用One-Class SVM算法计算异常判断边界
# ocsvm = OneClassSVM(nu=0.05)
ocsvm = OneClassSVM()
ocsvm.fit(a_pca)

# 在b数据集上进行测试
b_pca = pca.transform(b)
b_pred = ocsvm.predict(b_pca)

# 计算决策边界的x值
# x_decision_boundary = np.linspace(a_pca.min(), a_pca.max(), 500).reshape(-1, 1)
# print(x_decision_boundary)

# 计算决策边界的y值
# y_decision_boundary = ocsvm.decision_function(x_decision_boundary)


# 获取决策边界点
decision_boundary = ocsvm.decision_function(a_pca.reshape(-1, 1)).ravel()

# 获取决策边界点的 x 坐标
boundary_x = a_pca[np.argmin(decision_boundary)]

# 使用Kernel Density Estimation (KDE) 估计数据点密度
kde_a = KernelDensity(bandwidth=0.1)
kde_a.fit(a_pca)
density_a = np.exp(kde_a.score_samples(a_pca))


# 绘制图像
plt.figure(figsize=(12, 6))


# 绘制数据集 A 和 B 的直方图
plt.hist(a_pca, bins=100, density=True, alpha=0.5, color='blue', label='Normal')

plt.hist(b_pca, bins=100, density=True, alpha=0.5, color='red', label='Abnormal')

# 绘制决策边界
# plt.plot([x_decision_boundary[0][0], x_decision_boundary[-1][0]], [0, 0], color='red', linestyle='--', linewidth=2, label='Boundary')


# 绘制决策边界点
plt.scatter(boundary_x, 0, color='green', marker='x', s=100, label='Decision point')

# 在决策边界点处画一条垂直于x轴的直线
plt.axvline(x=boundary_x, color='purple', linestyle='--', label='Boundary')

plt.title('One-Class SVM')
plt.xlabel('Feature')
plt.ylabel('Density')

# 显示图例
plt.legend()

# 显示图像
plt.show()


'''
# 绘制a数据集的异常判断边界
plt.subplot(1, 2, 1)
plt.scatter(a_pca, np.zeros_like(a_pca), label='Alice')
plt.scatter(b_pca[b_pred == -1], np.zeros_like(b_pca[b_pred == -1]), label='Anomaly Detection', color='r')
plt.title('One-Class SVM - Anomaly Detection')
plt.xlabel('PCA to 1')
plt.legend()

# 绘制b数据集的测试结果
plt.subplot(1, 2, 2)
plt.scatter(b_pca, np.zeros_like(b_pca), label='Eve')
plt.scatter(b_pca[b_pred == -1], np.zeros_like(b_pca[b_pred == -1]), label='Anomaly Detection', color='r')
plt.title('Detection Result')
plt.xlabel('PCA to 1')
plt.legend()

plt.show()
'''


'''
# 将标签修改为二进制形式，1 表示正常数据，-1 表示异常数据
y_true = np.ones_like(a_pca)
y_pred_a = np.ones_like(a_pca)
y_pred_b = b_pred

# 计算混淆矩阵
conf_matrix_a = confusion_matrix(y_true, y_pred_a)
conf_matrix_b = confusion_matrix(-np.ones_like(b_pca), y_pred_b)  # 注意：将标签反转为二进制形式

# 计算其他性能指标
accuracy_a = accuracy_score(y_true, y_pred_a)
accuracy_b = accuracy_score(-np.ones_like(b_pca), y_pred_b)
precision_b = precision_score(-np.ones_like(b_pca), y_pred_b)
recall_b = recall_score(-np.ones_like(b_pca), y_pred_b)
f1_b = f1_score(-np.ones_like(b_pca), y_pred_b)

# 打印结果
print("Confusion Matrix for Dataset A:")
print(conf_matrix_a)
print("Accuracy for Dataset A:", accuracy_a)

print("\nConfusion Matrix for Dataset B:")
print(conf_matrix_b)
print("Accuracy for Dataset B:", accuracy_b)
print("Precision for Dataset B:", precision_b)
print("Recall for Dataset B:", recall_b)
print("F1 Score for Dataset B:", f1_b)
'''

# Plot ROC curve
y_true = np.ones_like(b_pca)  # 1 for normal data
y_true[b_pred == -1] = 0      # 0 for detected anomalies

fpr, tpr, _ = roc_curve(y_true, b_pca)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()