% OFDM参数
N = 64;              % 子载波数
CP = 16;             % 循环前缀长度
M = 4;               % QPSK调制
numSymbols = 10;     % OFDM符号数

% 调制
data = randi([0 M-1], N, numSymbols);
modData = pskmod(data, M);

% IFFT
ofdmSymbols = ifft(modData);

% 加入循环前缀
ofdmSymbols_CP = [ofdmSymbols(end-CP+1:end, :); ofdmSymbols];

% 定义RIS相位矩阵
theta = pi * rand(N, 1);  % 随机相位调整
Phi = diag(exp(1j * theta));  % RIS相位调整矩阵 (N x N)

% 定义信道（简单模拟，单位矩阵用于测试）
H_TX_RIS = randn(N) + 1j*randn(N);  % 发射机到RIS的信道矩阵 (N x N)
H_RIS_RX = randn(N) + 1j*randn(N);  % RIS到接收机的信道矩阵 (N x N)

% 信号通过RIS后的传播
% 由于ofdmSymbols_CP是(N + CP) x numSymbols，因此我们只对有效符号部分进行信道处理
receivedSignal = H_RIS_RX * Phi * H_TX_RIS * ofdmSymbols;  % N x numSymbols 矩阵

% 加入循环前缀后信号在接收端
receivedSymbols_CP_removed = [receivedSignal(end-CP+1:end, :); receivedSignal];

% 接收处理
receivedSymbols_freq = fft(receivedSymbols_CP_removed);  % 进行FFT回到频域

% 解调
demodData = pskdemod(receivedSymbols_freq, M);

% 输出结果
disp('原始数据:');
disp(data(:,1));  % 显示第一个符号的数据
disp('解调后的数据:');
disp(demodData(:,1));

