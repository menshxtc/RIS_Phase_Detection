import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import umap

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
x_train = process_data(ris_2_on)

# 2米无ris的非法数据
mat_file_2_off='RSSTxRIS2none.mat'
mat_data_2_off = sio.loadmat(mat_file_2_off)
ris_2_off = mat_data_2_off['RSSTxRIS2none']
x_test = process_data(ris_2_off)


# ocsvm_model = OneClassSVM(nu=0.5,kernel='rbf')
# ocsvm_model.fit(x_train)
#
# y_pred = ocsvm_model.predict(x_test)
# detection_rate = sum(y_pred == -1)/len(y_pred)
# print("异常检测率：",detection_rate)


# 2维无需使用t-SNE可视化
# pca = PCA(n_components=2)
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)
# embedded_data = np.vstack((x_train_pca, x_test_pca))
# embedded_legitimate_data = embedded_data[:500, :]
# embedded_test_data = embedded_data[-500:, :]
#
# # 保存数据到MAT文件
# data_dict = {
#     'legitimate_data_pca': embedded_legitimate_data,
#     'anomalous_data_pca': embedded_test_data
# }
# sio.savemat('pca_2.mat', data_dict)


# # 高纬数据使用t-SNE可视化
# pca = PCA(n_components=40)
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)
#
# # 训练oc-svm模型
# # ocsvm_model_pca = OneClassSVM()
# # ocsvm_model_pca.fit(x_train_pca)
# #
# # y_pred_pca = ocsvm_model_pca.predict(x_test_pca)
# # detection_rate_pca = sum(y_pred_pca == -1)/len(y_pred_pca)
# # print("异常检测率：",detection_rate_pca)
#
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data = tsne.fit_transform(np.vstack((x_train_pca, x_test_pca)))
#
# # 分离合法和非法数据
# mid_index = len(embedded_data) // 2
# # embedded_legitimate_data = embedded_data[:mid_index, :]
# # embedded_test_data = embedded_data[mid_index:, :]
# embedded_legitimate_data = embedded_data[:200, :]
# embedded_test_data = embedded_data[-200:, :]
#
# # oc-svm模型检测率
# # ocsvm_model = OneClassSVM(nu=0.5,kernel='rbf')
# # ocsvm_model.fit(embedded_legitimate_data)
# # y_pred = ocsvm_model.predict(embedded_test_data)
# # detection_rate = sum(y_pred == -1)/len(y_pred)
# # print("异常检测率：",detection_rate)


# 原数据t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(np.vstack((x_train, x_test)))
embedded_legitimate_data = embedded_data[:500, :]
embedded_test_data = embedded_data[-500:, :]

# # 可视化
# plt.figure(figsize=(12, 6))
#
# # 绘制合法数据集
# plt.scatter(embedded_legitimate_data[:,0], embedded_legitimate_data[:,1], label='Legitimate', marker='o', color='blue')
#
# # 绘制非法数据集
# plt.scatter(embedded_test_data[:,0], embedded_test_data[:,1], label='Anomalous', marker='x', color='red')
#
# plt.title('PCA Visualization using t-SNE')
# plt.legend()
# plt.show()


# t-SNE降维到三维
# tsne = TSNE(n_components=3, random_state=42)
# embedded_data = tsne.fit_transform(np.vstack((x_train, x_test)))

# PCA降维到三维
# pca = PCA(n_components=3)
# x_train_pca = pca.fit_transform(x_train)
# x_test_pca = pca.transform(x_test)
# embedded_data = np.vstack((x_train_pca,x_test_pca))

# 使用umap降维
# umap_model = umap.UMAP(n_components=3)
# x_train_umap = umap_model.fit_transform(x_train)
# x_test_umap = umap_model.fit_transform(x_test)
# embedded_data = np.vstack((x_train_umap,x_test_umap))

# 分离合法和非法数据
# mid_index = len(embedded_data) // 2
# embedded_legitimate_data = embedded_data[:mid_index, :]
# embedded_test_data = embedded_data[mid_index:, :]
# embedded_legitimate_data = embedded_data[:500, :]
# embedded_test_data = embedded_data[-500:, :]

# # oc-svm模型检测率
# ocsvm_model = OneClassSVM(nu=0.5,kernel='rbf')
# ocsvm_model.fit(embedded_legitimate_data)
#
# y_pred = ocsvm_model.predict(embedded_test_data)
# detection_rate = sum(y_pred == -1)/len(y_pred)
# print("异常检测率：",detection_rate)
#
# # 可视化
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制合法数据集
# ax.scatter(embedded_legitimate_data[:, 0], embedded_legitimate_data[:, 1], embedded_legitimate_data[:, 2],
#            label='Legitimate',  marker='o', edgecolors='blue')
#
# # 绘制非法数据集
# ax.scatter(embedded_test_data[:, 0], embedded_test_data[:, 1], embedded_test_data[:, 2],
#            label='Anomalous', marker='o', edgecolors='red')
#
# ax.set_title('t-SNE 3D Visualization after OC-SVM')
# ax.legend()
# plt.show()

# 保存数据到MAT文件
data_dict = {
    'legitimate_data': embedded_legitimate_data,
    'anomalous_data': embedded_test_data
}

sio.savemat('oc_svm_tsne_2d.mat', data_dict)
# sio.savemat('pca_40.mat', data_dict)

# roc曲线绘制

# y_test_binary = label_binarize(np.ones(len(y_test)), classes=[-1, 1])
# y_score = ocsvm_model.decision_function(x_test)
# fpr, tpr, thresholds = roc_curve(y_pred, y_score)
# roc_auc = auc(fpr, tpr)
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
#
# # 保存ROC曲线数据为MATLAB格式的矩阵
# roc_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}
# sio.savemat('ris_oc_svm_roc.mat', roc_data)