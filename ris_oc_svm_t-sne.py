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


# 使用T-SNE降维
tsne = TSNE(n_components=3, random_state=42)
x_train_dim_dec = tsne.fit_transform(x_train)
x_test_dim_dec = tsne.fit_transform(x_test)

# 使用umap降维
# umap_model = umap.UMAP(n_components=3)
# x_train_dim_dec = umap_model.fit_transform(x_train)
# x_test_dim_dec = umap_model.fit_transform(x_test)


ocsvm_model = OneClassSVM(nu=0.5,kernel='rbf')
ocsvm_model.fit(x_train_dim_dec)

y_pred = ocsvm_model.predict(x_test_dim_dec)
detection_rate = sum(y_pred == -1)/len(y_pred)
print("异常检测率：",detection_rate)


# roc曲线绘制

# y_test_binary = label_binarize(np.ones(len(y_test)), classes=[-1, 1])
y_score = ocsvm_model.decision_function(x_test_dim_dec)
fpr, tpr, thresholds = roc_curve(y_pred, y_score)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 保存ROC曲线数据为MATLAB格式的矩阵
# roc_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}
# sio.savemat('ris_oc_svm_roc.mat', roc_data)