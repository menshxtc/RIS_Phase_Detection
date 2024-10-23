import ris_data

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

# 引入数据
# cls_4off_7off = ris_data.risClass(ris_data.ris_4_off, ris_data.ris_7_off)
# cls_4off_7off.stand_data()
# cls_4off_7off.pca_to_2()
#
# x_train = cls_4off_7off.x_train
# x_test = cls_4off_7off.x_test
# y_train = cls_4off_7off.y_train
# y_test = cls_4off_7off.y_test
# x_train_pca = cls_4off_7off.x_train_pca_2
# x_test_pca = cls_4off_7off.x_test_pca_2


colums_2off = ris_data.ris_2_off.shape[1]
half_2off = int(colums_2off/2)


cls_2off_2off = ris_data.risClass(ris_data.ris_2_off[:,:half_2off],ris_data.ris_2_off[:,half_2off:])
cls_2off_2off.stand_data()
# cls_2off_2off.pca_to_2()
# cls_2off_2off.pca_to_n(4)

x_train = cls_2off_2off.x_train
x_test = cls_2off_2off.x_test
y_train = cls_2off_2off.y_train
y_test = cls_2off_2off.y_test
x_train_pca = cls_2off_2off.x_train_pca_600
x_test_pca = cls_2off_2off.x_test_pca_600


colums_2on = ris_data.ris_2_on.shape[1]
half_2on = int(colums_2on/2)


# cls_2on_2off = ris_data.risClass(ris_data.ris_2_on[:,:half_2on],ris_data.ris_2_off[:,half_2off:])
# cls_2on_2off.stand_data()
# cls_2on_2off.pca_to_2()
#
# x_train = cls_2on_2off.x_train
# x_test = cls_2on_2off.x_test
# y_train = cls_2on_2off.y_train
# y_test = cls_2on_2off.y_test
# x_train_pca = cls_2on_2off.x_train_pca_2
# x_test_pca = cls_2on_2off.x_test_pca_2

# 创建线性化svm分类器，正则化参数为1.0
# svm_clf = svm.SVC(kernel='linear', C=1.0)
# svm_clf.fit(x_train_pca, y_train)
# y_pred_clf = svm_clf.predict(x_test_pca)
# acc_clf = accuracy_score(y_test,y_pred_clf)
# print("线性svm准确率为:",acc_clf)

# 创建核函数svm分类器
svm_ker = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_ker.fit(x_train,y_train)
y_pred_ker = svm_ker.predict(x_test)
acc_ker = accuracy_score(y_test,y_pred_ker)
print("核函数svm准确率为:",acc_ker)


# 计算虚警率和错检率
# y_scores = svm_clf.decision_function(x_test_pca)
y_scores = svm_ker.decision_function(x_test)

# # 绘制训练集与测试集散点图
# plt.figure(figsize=(10,10))
# plt.scatter(
#     x_train_pca[:,0],
#     x_train_pca[:,1],
#     c=y_train,
#     cmap=plt.cm.Paired,
#     marker='o',
#     s=50,
#     label='train data'
# )
# plt.scatter(
#     x_test_pca[:,0],
#     x_test_pca[:,1],
#     c=y_test,
#     cmap=plt.cm.Paired,
#     marker='x',
#     s=100,
#     label='test data'
# )
#
#
# # 绘制决策边界
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
#
#
# # 创建网格点来绘制决策边界
# xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
# xy = np.column_stack([xx.ravel(), yy.ravel()])
# Z = svm_clf.decision_function(xy)
# Z = Z.reshape(xx.shape)
#
# # 绘制决策边界和间隔
# ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
#
# # 结果展示
# plt.title('SVM Decision')
# plt.legend(loc='upper right',labelcolor='black')
# plt.xlabel('Principal Feature 1')
# plt.ylabel('Principal Feature 2')
# plt.show()
#
#



y_true = y_test

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# print("False Positive Rate:",fpr,"True Positive Rate:",tpr)

# 计算AUC
auc = roc_auc_score(y_true, y_scores)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)


# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FAR)')
plt.ylabel('True Positive Rate (1 - MDR)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
