#!/usr/bin/env python

import os
import h5py
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

file_name = r'CXS_t2'
file_path = r'emg_rmMA'
# 提取文件名（不带扩展名）
save_dir = f"data_filtered/200/25"
window_length = 0.2  # 窗口长度（秒）
overlap_rate = 0.25  # 窗口重叠率
# 导入输入数据
with h5py.File(f'{file_path}/{file_name}.mat', 'r') as f:
    EMG_128 = f['EMG_128'][:]
    EMG_128rmMA = f['EMG_rmMA'][:]
    EMG_128MA = f['EMG_MA'][:]
fs_e = 2048  # 采样频率

def data_format(arr):
    return arr if arr.shape[0] >= arr.shape[1] else arr.T

def select_emg_channels(emg_data, threshold=5.0, target_num=32):
    """
    emg_data: shape (T, 128) 的 EMG 数据，T 为时间长度
    threshold: 用于初步筛选通道的幅值阈值
    target_num: 最终需要的通道数（默认32）

    返回：selected_channels：最终选择的32个通道编号
    """
    assert emg_data.shape[1] == 128, "EMG数据应为128通道"

    # Step 1: 初筛 - 最大绝对值大于阈值
    max_vals = np.max(np.abs(emg_data), axis=0)
    high_val_indices = np.where(max_vals > threshold)[0].tolist()

    print(f"高幅值通道数量: {len(high_val_indices)}")

    # Step 2: 如果已经超过 32 个，取前32个高幅值通道
    if len(high_val_indices) >= target_num:
        return high_val_indices[:target_num]

    # Step 3: 用方法1从剩余通道中补足
    remaining_num = target_num - len(high_val_indices)

    # 生成规则采样通道索引（方法1）
    def regular_sample():
        channels = []
        for arr_offset in [0, 64]:  # 两个8x8阵列
            for i in range(0, 8, 2):         # 行：0,2,4,6
                for j in range(0, 8, 2):     # 列：0,2,4,6
                    ch = arr_offset + i * 8 + j
                    channels.append(ch)
        return channels  # 共16+16 = 32个

    regular_channels = regular_sample()

    # 去掉已经在 high_val_indices 中的通道
    remaining_candidates = [ch for ch in regular_channels if ch not in high_val_indices]

    if len(remaining_candidates) < remaining_num:
        raise ValueError(f"规则采样通道不足以补足32个通道，请提高阈值或调整采样策略")

    selected_channels = high_val_indices + remaining_candidates[:remaining_num]

    return sorted(selected_channels)

EMG_128 = data_format(EMG_128)
EMG_128rmMA = data_format(EMG_128rmMA)
EMG_128MA = data_format(EMG_128MA)
# 频谱分析
# Y = np.fft.fft(EMG_rmMA[0, :])  # 对第 1 通道进行 FFT
# L = EMG_rmMA.shape[1]  # 信号长度
# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(L), EMG_rmMA[0, :])  # 绘制原始信号
# plt.title("Original Signal")
# plt.subplot(2, 1, 2)
# plt.plot(fs_e / L * np.arange(L // 2), np.abs(Y[:L // 2]), linewidth=1)  # 绘制频谱
# plt.title("Frequency Spectrum")
selected_channels = select_emg_channels(EMG_128, threshold=5.0, target_num=32)
print("最终选取通道索引:", selected_channels)
EMG_32 = EMG_128[:, selected_channels]
EMG_32rmMA = EMG_128rmMA[:, selected_channels]

num_emg_channels = EMG_32.shape[1]  # 通道数
num_emg_samples = EMG_32.shape[0]  # 每通道样本数
# 构造陷波器
frequencies_to_notch = [50.1, 100.2, 150.3, 200.4, 250.5, 300.6, 350.7, 400.8, 450.9, 501]  # 要抑制的频率
delta_f = 1  # 带宽为 1 Hz（±0.5 Hz）
# 设计高通滤波器，滤除直流分量
dc_cut = 1  # 高通滤波器的截止频率（1 Hz）
b_dc, a_dc = butter(6, dc_cut / (fs_e / 2), btype='high') # 设计 6 阶高通滤波器
# 设计 Butterworth 带通滤波器
lowcut = 10  # 带通滤波器下限频率
highcut = 500  # 带通滤波器上限频率
b_d, a_d = butter(6, [lowcut / (fs_e / 2), highcut / (fs_e / 2)], btype='band')  # 设计 6 阶带通滤波器

filtered_data_32 = np.zeros_like(EMG_32)  # 初始化滤波后的数据
filtered_data_32rmMA = np.zeros_like(EMG_32rmMA)  # 初始化滤波后的数据
for i in range(num_emg_channels):
    filtered_data_32[:, i] = EMG_32[:, i]
    filtered_data_32rmMA[:, i] = EMG_32rmMA[:, i]
    # 应用陷波器
    for freq in frequencies_to_notch:
        Q = 100
        b, a = iirnotch(freq / (fs_e / 2), Q)
        filtered_data_32[:, i] = filtfilt(b, a, filtered_data_32[:, i])
        filtered_data_32rmMA[:, i] = filtfilt(b, a, filtered_data_32rmMA[:, i])
     # 应用高通滤波器
    filtered_data_32[:, i] = filtfilt(b_dc, a_dc, filtered_data_32[:, i])
    filtered_data_32rmMA[:, i] = filtfilt(b_dc, a_dc, filtered_data_32rmMA[:, i])
    # 应用带通滤波器
    filtered_data_32[:, i] = filtfilt(b_d, a_d, filtered_data_32[:, i])
    filtered_data_32rmMA[:, i] = filtfilt(b_d, a_d, filtered_data_32rmMA[:, i])

# 绘制滤波后的信号和频谱
# Y2 = np.fft.fft(filtered_data[0, :])
# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(L), filtered_data[0, :])  # 绘制滤波后的信号
# plt.title("Filtered Signal")
# plt.subplot(2, 1, 2)
# plt.plot(fs_e / L * np.arange(L // 2), np.abs(Y2[:L // 2]), linewidth=1)  # 绘制频谱
# plt.title("Filtered Frequency Spectrum")

# # 使用 Hampel Identifier 剔除异常点
# window_size = 25  # 滑动窗口大小
# n_sigma = 3.0  # 异常点判断的标准差倍数
# for i in range(num_emg_channels):
#     # 剔除异常值
#     filtered_data[i, :] = hampel(filtered_data[i, :], window_size, n_sigma).filtered_data  # 替换异常值

# 数据预处理：加窗并计算 RMS

window_samples = int(window_length * fs_e)  # 窗口长度对应的样本数
overlap_samples = int(overlap_rate * window_samples)  # 窗口重叠对应的样本数
# 创建滑动窗口
windows = np.arange(0, num_emg_samples - window_samples + 1, window_samples - overlap_samples)
# 初始化 RMS 值矩阵
rms_emg_values_32 = np.zeros((len(windows), num_emg_channels, ))
rms_emg_values_32rmMA = np.zeros((len(windows), num_emg_channels))
for i in range(num_emg_channels):
    # 计算 RMS 值
    for k, start in enumerate(windows):
        window_emg_data_32 = filtered_data_32[start:start + window_samples, i]
        rms_emg_values_32[k, i] = np.sqrt(np.mean(window_emg_data_32 ** 2))
        window_emg_data_32rmMA = filtered_data_32rmMA[start:start + window_samples, i]
        rms_emg_values_32rmMA[k, i] = np.sqrt(np.mean(window_emg_data_32rmMA ** 2))      

# 8. 绘制 RMS 值的频谱分析
input_emg_32 = rms_emg_values_32  
input_emg_32rmMA = rms_emg_values_32rmMA
# Y3 = np.fft.fft(input_emg[0, :])  # 对第 1 通道 RMS 值进行 FFT
# L3 = input_emg.shape[1]  # RMS 值长度
# plt.figure(4)
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(L3), input_emg[0, :])  # 绘制 RMS 值
# plt.title("RMS Signal")
# plt.subplot(2, 1, 2)
# plt.plot(fs_e / L3 * np.arange(L3 // 2), np.abs(Y3[:L3 // 2]), linewidth=1)  # 绘制频谱
# plt.title("RMS Frequency Spectrum")

# 9. 导入输出数据
output_data = sio.loadmat(fr'force/Force_{file_name}.mat')  # 加载 .mat 文件
Finger_Force = output_data['Force']  # 获取力信号
fs_f = output_data['fs_f'][0, 0]  # 原始采样频率
# 10. 升采样
Finger_Force_UpSampled = resample_poly(Finger_Force, up=int(fs_e), down=int(fs_f), axis=0)  # 升采样到 fs_e
# 11. 加窗处理
num_force_channels = Finger_Force_UpSampled.shape[1]  # 获取列数（通道数）
num_force_samples = Finger_Force_UpSampled.shape[0]  # 升采样后的样本数
average_force_amplitude = np.zeros((len(windows), num_force_channels))  # 初始化平均幅值矩阵
for col in range(num_force_channels):
    # 平移：将信号的最小值移到 0
    Finger_Force_UpSampled[:, col] -= np.min(Finger_Force_UpSampled[:, col])
    # 对每一列信号进行加窗处理
    for j, start in enumerate(windows):
        # 提取当前窗口的信号
        window_force_data = Finger_Force_UpSampled[start:start + window_samples, col]
        # 计算幅值平均值
        average_force_amplitude[j, col] = np.mean(np.abs(window_force_data))  # 取幅值的平均值

# 12. 输出结果
output_force = average_force_amplitude
# L4 = output_force.shape[1]  # RMS 值长度
# plt.figure(5)
# plt.subplot(3, 1, 1)
# plt.plot(np.arange(L4), output_force[0, :])  # 绘制原始信号
# plt.subplot(3, 1, 2)
# plt.plot(np.arange(L4), output_force[1, :])  # 绘制原始信号
# plt.subplot(3, 1, 3)
# plt.plot(np.arange(L4), output_force[2, :])  # 绘制原始信号
# plt.title("Original Signal")

# nmf = NMF(n_components=10, init='nndsvdar', random_state=42, max_iter=10000)
# low_input_emg = nmf.fit_transform(input_emg)
# fig, axes = plt.subplots(5, 2, figsize=(12, 10))  # 创建 5x2 的网格布局
# for i in range(5):  # 假设有 5 个通道
#     for j in range(2):  # 每行有 2 个通道
#         channel_index = i * 2 + j
#         if channel_index < low_input_emg.shape[1]:  # 确保通道索引不超出范围
#             axes[i, j].plot(low_input_emg[:, channel_index])
#             axes[i, j].set_title(f"channel {channel_index + 1}")
#             axes[i, j].grid(True)

# ====== 标准化输入和输出 ======
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# train, validation, test
train_prop = 0.6
valid_prop = 0.2

sequence_length = input_emg_32rmMA.shape[0]
n_train = int(train_prop * sequence_length)
n_val = int(valid_prop * sequence_length)
n_test = sequence_length - n_train - n_val 

split1 = n_train
split2 = n_train + n_val

x_train = x_scaler.fit_transform(input_emg_32rmMA[:split1,  :])
y_train = y_scaler.fit_transform(output_force[:split1,  :])
x_valid = x_scaler.transform(input_emg_32rmMA[split1:split2,  :])
y_valid = y_scaler.transform(output_force[split1:split2,  :]) 
x_test = x_scaler.transform(input_emg_32rmMA[split2:,  :])
x_test_ma = x_scaler.transform(input_emg_32[split2:,  :])
y_test = y_scaler.transform(output_force[split2:, :])

data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_valid': x_valid,
    'y_valid': y_valid,
    'x_test': x_test,
    'x_test_ma': x_test_ma,
    'y_test': y_test,
    'file_name': file_name
}

# 创建目录（如果不存在）
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{file_name}.h5")
# 保存数据到 HDF5 文件
with h5py.File(save_path, 'w') as f:
    for key, value in data.items():
        if isinstance(value, str):
            # 字符串保存为可变长度ASCII数据集
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(key, data=value, dtype=dt)
        else:
            f.create_dataset(key, data=value)
print(f"数据已保存至：{save_path}")