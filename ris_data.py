import numpy as np
import scipy.io as sio

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# 从矩阵中引入数据

# 2米无ris
mat_file_2_off='RSSTxRIS2none.mat'
mat_data_2_off = sio.loadmat(mat_file_2_off)
ris_2_off = mat_data_2_off['RSSTxRIS2none']

# 2米有ris
mat_file_2_on='RSSTxRIS2off.mat'
mat_data_2_on = sio.loadmat(mat_file_2_on)
ris_2_on = mat_data_2_on['RSSTxRIS2off']

# 4米无ris
mat_file_4_off='RSSTxRIS4.mat'
mat_data_4_off = sio.loadmat(mat_file_4_off)
ris_4_off = mat_data_4_off['RSSTxRIS4']

# 7米无ris
mat_file_7_off='RSSTxRIS7.mat'
mat_data_7_off = sio.loadmat(mat_file_7_off)
ris_7_off = mat_data_7_off['RSSTxRIS7']

# 选择测试数据集
# ris_on_data = ris_2_off
# ris_off_data = ris_4_off

class risClass:
    def __init__(self, ris_on_data, ris_off_data):
        self.ris_on_data = ris_on_data
        self.ris_off_data = ris_off_data

    def stand_data(self):
        # 提取偶数列数据作为信道信息进行训练
        num_col = self.ris_on_data.shape[1]
        col_step = np.arange(1, num_col, 2)
        ris_on_train = self.ris_on_data[:, col_step]
        ris_off_train = self.ris_off_data[:, col_step]

        # 直接使用数据进行训练
        # ris_on_train = self.ris_on_data
        # ris_off_train = self.ris_off_data

        # 将奇数列和偶数列数据合并作为信道信息进行训练
        # num_col = self.ris_on_data.shape[1]
        # col_step_0 = np.arange(0, num_col, 2)
        # col_step_1 = np.arange(1, num_col, 2)
        # ris_on_train_0 = self.ris_on_data[:, col_step_0]
        # ris_off_train_0 = self.ris_off_data[:, col_step_0]
        # ris_on_train_1 = self.ris_on_data[:, col_step_1]
        # ris_off_train_1 = self.ris_off_data[:, col_step_1]
        # ris_on_train = np.vstack((ris_on_train_0,ris_on_train_1))
        # ris_off_train = np.vstack((ris_off_train_0,ris_off_train_1))


        # 转置矩阵
        trans_ris_on_train = np.transpose(ris_on_train)
        trans_ris_off_train = np.transpose(ris_off_train)

        # 添加特诊标签，ris_on为1，ris_off为0
        on_rows = trans_ris_on_train.shape[0]
        self.ones_col = np.ones((on_rows, 1))
        on_data = np.hstack((trans_ris_on_train, self.ones_col))
        self.zeros_col = np.zeros((on_rows, 1))
        off_data = np.hstack((trans_ris_off_train, self.zeros_col))

        # 纵向合并两数据集
        self.data = np.vstack((on_data, off_data))

        # 划分数据集为训练集和测试集
        x = self.data[:, :-1]  # 所有列除了最后一列
        y = self.data[:, -1]  # 最后一列

        # 20%作为测试集，随机种子为42
        x_train, x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # 创建标准器
        scaler = StandardScaler()
        # 标准化训练集，服从标准正态分布
        self.x_train = scaler.fit_transform(x_train)
        # 直接使用训练集的均值与方差进行标准化
        self.x_test = scaler.transform(x_test)


    # 使用pca进行向量降维
    def pca_to_1(self):
        pca_1 = PCA(n_components=1,whiten=False)
        self.x_train_pca_1 = pca_1.fit_transform(self.x_train)
        self.x_test_pca_1 = pca_1.fit_transform(self.x_test)

        # 特诊损失率
        self.feature_loss_rate_1 = 1 - np.sum(pca_1.explained_variance_ratio_)

    def pca_to_2(self):
        pca_2 = PCA(n_components=2)
        self.x_train_pca_2 = pca_2.fit_transform(self.x_train)
        self.x_test_pca_2 = pca_2.fit_transform(self.x_test)

        # 特诊损失率
        self.feature_loss_rate_2 = 1 - np.sum(pca_2.explained_variance_ratio_)

    def pca_to_n(self,n):
        pca_n = PCA(n_components=n)
        self.x_train_pca_600 = pca_n.fit_transform(self.x_train)
        self.x_test_pca_600 = pca_n.fit_transform(self.x_test)

        # 特诊损失率
        self.feature_loss_rate_n = 1 - np.sum(pca_n.explained_variance_ratio_)
