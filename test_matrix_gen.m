% 定义矩阵大小
rows = 1200;
cols = 2000;

% 定义种子
seed = 24;
randn(seed);

% 生成服从标准正态分布的矩阵
test_matrix_off = randn(rows, cols);

% 设置所需的均值和方差
desired_mean = 5;
desired_variance = 0;

% 对生成的矩阵进行缩放和平移
% test_matrix_on = desired_mean + sqrt(desired_variance) * matrix_std_normal;

test_matrix_on = desired_mean + test_matrix_off;