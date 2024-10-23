import gc
import time

import memory_profiler
import numpy as np
import scipy.io as sio
from memory_profiler import profile

from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE

def process_data(matrix_data):
    # 提取偶数列数据作为信道信息进行训练
    num_col = matrix_data.shape[1]
    col_step = np.arange(1, num_col, 2)
    fea_data = matrix_data[:, col_step]

    # 转置矩阵
    trans_fea_data = np.transpose(fea_data)

    return trans_fea_data


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

# # 2米有ris的合法数据 测试数据
# mat_file_2_on='RSS_test_on.mat'
# mat_data_2_on = sio.loadmat(mat_file_2_on)
# ris_2_on = mat_data_2_on['test_matrix_on']
# x_train = process_data(ris_2_on)
#
# # 2米无ris的非法数据 测试数据
# mat_file_2_off='RSS_test_off.mat'
# mat_data_2_off = sio.loadmat(mat_file_2_off)
# ris_2_off = mat_data_2_off['test_matrix_off']
# x_test = process_data(ris_2_off)

'''
# 使用PCA降维至1维
pca = PCA(n_components=1)
a_pca = pca.fit_transform(a)
b_pca = pca.transform(b)
'''

# 不降维时检测率
# ocsvm_model = OneClassSVM(nu=0.5)
# ocsvm_model.fit(x_train)
#
# y_pred = ocsvm_model.predict(x_test)
# detection_rate = sum(y_pred == -1)/len(y_pred)
# print("异常检测率：",detection_rate)

def tsne_to_n(x_train,x_test,n):
    tsne = TSNE(n_components=n, random_state=42)
    embedded_data = tsne.fit_transform(np.vstack((x_train, x_test)))

    mid_index = len(embedded_data) // 2
    embedded_legitimate_data = embedded_data[:mid_index, :]
    embedded_test_data = embedded_data[mid_index:, :]

    ocsvm_model = OneClassSVM(nu=0.5, kernel='rbf')
    ocsvm_model.fit(embedded_legitimate_data)

    y_pred = ocsvm_model.predict(embedded_test_data)
    detection_rate = sum(y_pred == -1) / len(y_pred)
    print("异常检测率：", detection_rate)

    return detection_rate

@profile
def pca_to_n(x_train,x_test,n):
    result = []
    pca_start_time = time.time()
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    pca_end_time = time.time()
    pca_time = pca_end_time-pca_start_time
    result.append(pca_time)

    svm_start_time = time.time()
    ocsvm_model_pca = OneClassSVM()
    ocsvm_model_pca.fit(x_train_pca)
    svm_end_time = time.time()
    svm_time = svm_end_time-svm_start_time
    result.append(svm_time)

    pred_start_time = time.time()
    y_pred_pca = ocsvm_model_pca.predict(x_test_pca)
    pred_end_time = time.time()
    pred_time = pred_end_time-pred_start_time
    result.append(pred_time)

    detection_rate_pca = sum(y_pred_pca == -1) / len(y_pred_pca)
    result.append(detection_rate_pca)
    return result


# 存储参数和相应的运行时间，识别准确率
# dimension_n = list(range(1, 1201))
# dimension_n = list(range(1, 10)) + list(range(10, 1201, 50))
# dimension_n = list(range(1, 51)) + list(range(50, 1201, 50))
# dimension_n = [1]+list(range(10, 301, 10))
# dimension_n = [1]+list(range(50, 401, 50))

dimension_n = list(range(1, 101))
# dimension_n = list(range(131, 141))
# dimension_n = list(range(53, 57))
# dimension_n = [1]+list(range(50, 401, 50))
# dimension_n = [1]+list(range(20, 101, 20))

print(dimension_n)


def timeMeasure():
    runtimes = []
    accuracy = []
    memory_usage = []
    pcatimes = []
    svmtimes = []
    predtimes = []
    # 测试函数并测量运行时间
    for dimension in dimension_n:
        # 时间测量
        start_time = time.time()

        # 准确率测量
        mem_before = memory_profiler.memory_usage()[0]  # 获取降维分类前内存使用
        train_result = pca_to_n(x_train, x_test, dimension)
        mem_after = memory_profiler.memory_usage()[0]  # 获取降维分类后内存使用

        end_time = time.time()
        run_time = end_time - start_time
        train_time = train_result[0]+train_result[1]

        pcatimes.append(train_result[0])
        svmtimes.append(train_result[1])
        predtimes.append(train_result[2])
        runtimes.append(train_time)
        accuracy.append(train_result[3])

        print(f"The {dimension} dimension:")
        print("Runtime：", run_time)
        print("PCA Train Time：", train_result[0])
        print("OC-SVM Train Time：", train_result[1])
        print("OC-SVM Predict Time：", train_result[2])

        mem_increment = mem_after - mem_before
        memory_usage.append(mem_increment)
        print("Memory Increment：", mem_increment, "MiB")

        # 强制垃圾回收
        gc.collect()


    # fig_data = {
    #     'oc_dimension_1':dimension_n,
    #     'oc_runtime_1':runtimes,
    #     'oc_svm_dete_1':accuracy,
    #     'pca_train_time_1':pcatimes,
    #     'svm_train_time_1':svmtimes,
    #     'predict_time_1': predtimes
    # }
    # sio.savemat('oc_svm_runtime_pre.mat',fig_data)


if __name__ == '__main__':
    import cProfile
    cProfile.run('timeMeasure()', 'output.pstats')

    import pstats
    with open('profiling_output.txt', 'w') as f:
        stats = pstats.Stats('output.pstats', stream=f)
        stats.strip_dirs().sort_stats('cumulative').print_stats()