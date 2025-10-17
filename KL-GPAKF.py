#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch, os, gc
import gpytorch
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.signal import correlate, correlation_lags
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter as KF, UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

torch.set_num_threads(os.cpu_count())  # 设置为 CPU 核心数
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空 GPU 缓存

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

filename = r'CXS_t2'
file_path = r'200/25'
save_dir = f"GP_KF/{file_path}"
fig_idx = 2
with h5py.File(f'data_filtered/{file_path}/{filename}.h5', 'r') as f:
    x_train = f['x_train'][:]
    y_train = f['y_train'][:]
    x_valid = f['x_valid'][:]
    y_valid = f['y_valid'][:]
    x_test = f['x_test'][:]
    x_test_ma = f['x_test_ma'][:]
    y_test = f['y_test'][:]
    file_name = f['file_name'][()].decode('utf-8')  # 读取字符串需 decode

def align_sequences(seq1, seq2):
    """
    Align seq2 to seq1 by finding the lag that maximizes their cross-correlation.
    :param seq1: Fixed sequence (numpy array).
    :param seq2: Sequence to be shifted (numpy array).
    :return: Aligned seq2 (numpy array).
    """
    # Compute cross-correlation
    seq1 = np.ravel(seq1)
    seq2 = np.ravel(seq2)
    corr = correlate(seq1, seq2, mode='full')
    lags = correlation_lags(seq1.size, seq2.size, mode="full")
    lag = lags[np.argmax(corr)]

    # Shift seq2 based on the lag
    if lag > 0:
        aligned_seq2 = np.pad(seq2, (lag, 0), mode='constant')[:len(seq1)]
    elif lag < 0:
        aligned_seq2 = np.pad(seq2, (0, -lag), mode='constant')[-lag:len(seq1) - lag]
    else:
        aligned_seq2 = seq2[:len(seq1)]
    
    return aligned_seq2

# Training data is 100 points in [0,1] inclusive regularly spaced
num_samples = x_train.shape[0] + x_valid.shape[0] + x_test.shape[0]
split1 = x_train.shape[0]
split2 = x_train.shape[0] + x_valid.shape[0]
idx = np.arange(num_samples)

train_x = torch.tensor(x_train, dtype=torch.float64)  # 测试集输入
train_y = torch.tensor(y_train[:, fig_idx], dtype=torch.float64)  # 测试集输出
valid_x = torch.tensor(x_valid, dtype=torch.float64)  # 测试集输入
valid_y = torch.tensor(y_valid[:, fig_idx], dtype=torch.float64)  # 测试集输出
test_x = torch.tensor(x_test, dtype=torch.float64)  # 测试集输入
test_x_ma = torch.tensor(x_test_ma, dtype=torch.float64)  # 测试集输入
test_y = torch.tensor(y_test[:, fig_idx], dtype=torch.float64)  # 测试集输出

if torch.cuda.is_available():
    output_device = torch.device('cuda:0')
    # make continguous
    train_x, train_y = train_x.contiguous().to(output_device), train_y.contiguous().to(output_device)
    valid_x, valid_y = valid_x.contiguous().to(output_device), valid_y.contiguous().to(output_device)
    test_x, test_x_ma, test_y = test_x.contiguous().to(output_device), test_x_ma.contiguous().to(output_device), test_y.contiguous().to(output_device)
    
# We will use the simplest form of GP model, exact inference
class SimpleGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        rbf_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=train_x.shape[1]
        )
        matern_kernel = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=train_x.shape[1]
        )
        spectral_mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=4, ard_num_dims=train_x.shape[1]
        )
         # 初始化 SpectralMixtureKernel 参数
        # spectral_mixture_kernel.initialize_from_data(train_x, train_y)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(
                matern_kernel
                + spectral_mixture_kernel  # 使用 Matern 和 Spectral Mixture 核的加法核
            )
        )

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z, factor=1.0):
        y = z - np.dot(self.H, self.x)
        S = self.R * factor + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


# # initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
model = SimpleGPModel(train_x, train_y, likelihood).double()
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# Use the adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)  # Includes GaussianLikelihood parameters
# 初始化学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3)
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 1000
early_stopping_threshold = 1e-6  # 设置早停阈值
check_interval = 10  # 每10次记录一次loss
train_loss_history = []
valid_loss_history = []
best_MSE = None
PATIENCE = 10
patience = None

for i in range(training_iter):
    model.train()
    likelihood.train()
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    train_output = model(train_x)
    # Calc loss and backprop gradients
    train_loss = -mll(train_output, train_y)
    train_loss.backward()
    optimizer.step()
    # 根据损失更新学习率
    scheduler.step(train_loss.item())
    # 检查早停条件
    if train_loss_history:
        if abs(train_loss_history[-1] - train_loss.item()) < early_stopping_threshold:
            # 如果损失变化小于阈值，或者当前损失大于前一次损失，则停止训练
            print(f"Loss has converged (change < {early_stopping_threshold}). Stopping early at iteration {i + 1}.")
            break
    if optimizer.param_groups[0]['lr'] < 1e-4:
        print("Learning rate is too small. Stopping training.")
        break
    train_loss_history.append(train_loss.item())    

    # validation
    model.eval()
    likelihood.eval()
    if train_loss.item() < 0.0:
        with torch.no_grad():
            valid_output = model(valid_x)
            valid_loss = - likelihood.log_marginal(valid_y, valid_output).mean()
            valid_loss_history.append(valid_loss.item())
            if best_MSE is None or valid_loss.item() < best_MSE:
                best_MSE = valid_loss.item()
                patience = PATIENCE
            else:
                patience -= 1
                if patience == 0:
                    break
    if i % check_interval == 0:
        if valid_loss_history:
            print(f"Iter {i + 1}/{training_iter} -Train Loss: {train_loss_history[-1]:.6f}, Valid Loss: {valid_loss_history[-1]:.6f}, best_MSE: {best_MSE: .6f}, LR: {optimizer.param_groups[0]['lr']:.6f}") 
        else:
            print(f"Iter {i + 1}/{training_iter} -Train Loss: {train_loss_history[-1]:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}") 
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Train
    train_post_pred = likelihood(model(train_x))
    train_post_mean = train_post_pred.mean
    train_post_var = train_post_pred.variance  
    train_lower, train_upper = train_post_pred.confidence_region()
    # Valid
    valid_post_pred = likelihood(model(valid_x))
    valid_post_mean = valid_post_pred.mean
    valid_post_var = valid_post_pred.variance 
    valid_lower, valid_upper = valid_post_pred.confidence_region() 
    # Test
    test_post_pred = likelihood(model(test_x))
    test_post_mean = test_post_pred.mean 
    test_post_var = test_post_pred.variance  
    test_lower, test_upper = test_post_pred.confidence_region()

    test_ma_post_pred = likelihood(model(test_x_ma))
    test_ma_post_mean = test_ma_post_pred.mean 
    test_ma_post_var = test_ma_post_pred.variance 
    test_ma_lower, test_ma_upper = test_ma_post_pred.confidence_region()
    # Prior
    noise_var =  likelihood.noise
    train_prior_var = model.covar_module(train_x, train_x).diag()[0]
    valid_prior_var = model.covar_module(valid_x, valid_x).diag()[0]
    test_prior_var = model.covar_module(test_x, test_x).diag()[0]
    test_ma_prior_var = model.covar_module(test_x_ma, test_x_ma).diag()[0]

    if torch.cuda.is_available():
        train_post_mean = train_post_mean.detach().cpu().numpy()
        train_post_var = train_post_var.detach().cpu().numpy()  
        train_lower, train_upper = train_lower.detach().cpu().numpy(), train_upper.detach().cpu().numpy()
        train_prior_var = train_prior_var.detach().cpu().numpy()

        valid_post_mean = valid_post_mean.detach().cpu().numpy()
        valid_post_var = valid_post_var.detach().cpu().numpy()
        valid_lower, valid_upper = valid_lower.detach().cpu().numpy(), valid_upper.detach().cpu().numpy()
        valid_prior_var = valid_prior_var.detach().cpu().numpy()

        test_post_mean = test_post_mean.detach().cpu().numpy()
        test_post_var = test_post_var.detach().cpu().numpy() 
        test_lower, test_upper = test_lower.detach().cpu().numpy(), test_upper.detach().cpu().numpy()
        test_prior_var = test_prior_var.detach().cpu().numpy()

        test_ma_post_mean = test_ma_post_mean.detach().cpu().numpy()    
        test_ma_post_var = test_ma_post_var.detach().cpu().numpy()
        test_ma_lower, test_ma_upper = test_ma_lower.detach().cpu().numpy(), test_ma_upper.detach().cpu().numpy()
        test_ma_prior_var = test_ma_prior_var.detach().cpu().numpy()

        noise_var = noise_var.detach().cpu().numpy()
        
        train_x, train_y = train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy()
        valid_x, valid_y = valid_x.detach().cpu().numpy(), valid_y.detach().cpu().numpy()
        test_x, test_x_ma, test_y = test_x.detach().cpu().numpy(), test_x_ma.detach().cpu().numpy(), test_y.detach().cpu().numpy()


    truth_data = np.concatenate([train_y, valid_y, test_y])
    post_mean = np.concatenate([train_post_mean, valid_post_mean, test_post_mean])
    ma_post_mean = np.concatenate([train_post_mean, valid_post_mean, test_ma_post_mean])
    # 使用 Kalman Filter 进行后处理
    def kalman_filt(x0=None, dt = 0.1):      
        
        if x0 is None:
            x0 = np.zeros((2, 1))
        else:
            x0 = np.asarray(x0).reshape(2)
        P = np.diag([1e-4, 1.0])  # 初始协方差矩阵状态第一维信心高，第二维可以稍大
        Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-2)  # 过程噪声协方差
        R = np.array([[0.01]])  # 测量噪声协方差
        F = np.array([[1.0, dt], [0.0, 1.0]])  # 状态转移矩阵
        H = np.array([[1.0, 0.0]])  # 观测矩阵
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=x0)  # 初始化 Kalman Filter
        
        return kf 
    
    def kalman_filter(x0=None, dt = 0.1):      
        kf = KF(dim_x = 2, dim_z = 1)

        kf.F = np.array([[1.0, dt], [0.0, 1.0]])  # 状态转移矩阵
        kf.H = np.array([[1.0, 0.0]])  # 观测矩阵
        if x0 is None:
            kf.x0 = np.zeros((2, 1))
        else:
            kf.x0 = np.asarray(x0).reshape(2)
        kf.P = np.diag([1e-4, 1.0])  # 初始协方差矩阵状态第一维信心高，第二维可以稍大
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-4)  # 过程噪声协方差
        kf.R = np.array([[0.01]])  # 测量噪声协方差

        return kf 
    def unscented_kalman_filter(x0=None, dt=0.1):
        # 状态转移函数 fx
        def fx(x, dt):
            f, df = x
            return np.array([f + df * dt, df])
        # 观测函数 hx
        def hx(x):
            return np.array([x[0]])  # 只观测到力
        # UKF Sigma 点
        points = MerweScaledSigmaPoints(n=2, alpha=0.01, beta=2.0, kappa=0.0)
        # 初始化 UKF
        ukf = UKF(dim_x=2, dim_z=1, fx=fx, hx=hx, dt=dt, points=points)
        # 初始状态
        if x0 is None:
            ukf.x = np.zeros(2)
        else:
            ukf.x = np.asarray(x0).reshape(2)
        # 协方差与噪声设定
        ukf.P = np.diag([1e-1, 1.0])  # 初始协方差矩阵状态第一维信心高，第二维可以稍大  
        ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-3)  # 过程噪声协方差
        ukf.R = np.array([[0.01]])            # 测量噪声协方差

        return ukf
    
    def score_power(x, alpha):
        x_clipped = np.clip(x, 1e-6, None)
        return ((1 - x_clipped) / x_clipped) ** alpha
    # 计算 Knowledge Score: G = 1 - (y_var - noise_var) / signal_var
    G_train_scores = 1 - (train_post_var - noise_var) / train_prior_var
    predictions = list()
    kf_train = kalman_filt(x0=np.array([[train_post_mean[0]], [0.0]]))
    for i, z in enumerate(train_post_mean):
        kf_train.predict()
        predictions.append(kf_train.x.flatten()[0]) 
        kf_train.update(z, score_power(G_train_scores[i], 2))  # 使用 Knowledge Score 更新 Kalman Filter
    kf_train_post_mean = np.squeeze(np.array(predictions))

    G_valid_scores = 1 - (valid_post_var - noise_var) / valid_prior_var
    predictions = list()
    kf_valid = kalman_filt(x0=np.array([[valid_post_mean[0]], [0.0]]))
    for i, z in enumerate(valid_post_mean):
        kf_valid.predict()
        predictions.append(kf_valid.x.flatten()[0]) 
        kf_valid.update(z, score_power(G_valid_scores[i], 2))
    kf_valid_post_mean = np.squeeze(np.array(predictions))

    G_test_scores = 1 - (test_post_var - noise_var) / test_prior_var
    predictions = list()
    kf_test = kalman_filt(x0=np.array([[test_post_mean[0]], [0.0]]))
    for i, z in enumerate(test_post_mean):
        kf_test.predict()
        predictions.append(kf_test.x.flatten()[0]) 
        kf_test.update(z, score_power(G_test_scores[i], 2))
    kf_test_post_mean = np.squeeze(np.array(predictions))

    G_test_ma_scores = 1 - (test_ma_post_var - noise_var) / test_ma_prior_var
    predictions = list()
    kf_test_ma = kalman_filt(x0=np.array([[test_ma_post_mean[0]], [0.0]]))
    for i, z in enumerate(test_ma_post_mean):
        kf_test_ma.predict()
        predictions.append(kf_test_ma.x.flatten()[0]) 
        kf_test_ma.update(z, score_power(G_test_ma_scores[i], 2))
    kf_test_ma_post_mean = np.squeeze(np.array(predictions))

    G_scores = np.concatenate([G_train_scores, G_valid_scores, G_test_scores])
    kf_post_mean = np.concatenate([kf_train_post_mean, kf_valid_post_mean, kf_test_post_mean])

    G_scores_ma = np.concatenate([G_train_scores, G_valid_scores, G_test_ma_scores])
    kf_ma_post_mean = np.concatenate([kf_train_post_mean, kf_valid_post_mean, kf_test_ma_post_mean])
    
    fig_labels = ['index', 'middle', 'ring']

    f1, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    legends = []
    ax0.plot(idx, truth_data, c='k', ms=5)
    legends.append(plt.Line2D([0], [0], ls='', color='k', marker='.', ms=5, label='Truth data'))
    points = np.column_stack([idx, post_mean])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    # 创建 LineCollection
    lc0 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax0.add_collection(lc0)
    legends += [
        plt.Line2D([0], [0], ls='', color='b', marker='.', ms=5, label='Train pred'),
        plt.Line2D([0], [0], ls='', color='g', marker='.', ms=5, label='Valid pred'),
        plt.Line2D([0], [0], ls='', color='r', marker='.', ms=5, label='Test pred'),
    ]
    # Shade between the lower and upper confidence bounds
    ax0.fill_between(idx[:split1], train_lower, train_upper,  color='C0', alpha=0.3)
    ax0.fill_between(idx[split1:split2], valid_lower, valid_upper,  color='C2',  alpha=0.3)
    ax0.fill_between(idx[split2:], test_lower, test_upper, color='C3',  alpha=0.3)
    legends.append(patches.Rectangle(
            (1, 1), 1, 1, fill=True, color='b', alpha=0.3, lw=0, label='95% Error Bars'
        ))
    ax0.legend(handles=legends[::-1], loc='upper right', bbox_to_anchor=(1.17, 1), borderaxespad=0.)

    rmse_train = root_mean_squared_error(train_y, train_post_mean)
    rmse_valid = root_mean_squared_error(valid_y, valid_post_mean)
    rmse_test = root_mean_squared_error(test_y, test_post_mean)
    print(f'RMSE on train data: {rmse_train:.3f}, valid data: {rmse_valid:.3f}, test data: {rmse_test:.3f}')
    r2_train = r2_score(train_y, train_post_mean)
    r2_valid= r2_score(valid_y, valid_post_mean)
    r2_test = r2_score(test_y, test_post_mean)
    print(f'R^2 on training data: {r2_train:.3f}, valid data: {r2_valid:.3f}, test data: {r2_test:.3f}')
    ax0.text(0, 0.75, f"Test: RMSE = {rmse_test:.3f}  R² = {r2_test:.3f}", transform=ax0.transAxes)


    points = np.column_stack([idx, G_scores])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    lc1 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax1.add_collection(lc1)
    ax1.set_ylim(ax0.get_ylim())

    ax2.plot(idx, truth_data, c='k', ms=5)
    points = np.column_stack([idx, kf_post_mean])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    # 创建 LineCollection
    lc2 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax2.add_collection(lc2)
    ax2.set_ylim(ax0.get_ylim())
    rmse_train = root_mean_squared_error(train_y, kf_train_post_mean)
    rmse_valid = root_mean_squared_error(valid_y, kf_valid_post_mean)
    rmse_test = root_mean_squared_error(test_y, kf_test_post_mean)
    print(f'RMSE on train data: {rmse_train:.3f}, valid data: {rmse_valid:.3f}, test data: {rmse_test:.3f}')
    r2_train = r2_score(train_y, kf_train_post_mean)
    r2_valid= r2_score(valid_y, kf_valid_post_mean)
    r2_test = r2_score(test_y, kf_test_post_mean)
    print(f'R^2 on training data: {r2_train:.3f}, valid data: {r2_valid:.3f}, test data: {r2_test:.3f}')
    ax2.text(0, 0.75, f"Test: RMSE = {rmse_test:.3f}  R² = {r2_test:.3f}", transform=ax2.transAxes, va='top')

    f2, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    legends = []
    ax0.plot(idx, truth_data, c='k', ms=5)
    legends.append(plt.Line2D([0], [0], ls='', color='k', marker='.', ms=5, label='Truth data'))
    points = np.column_stack([idx, ma_post_mean])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    # 创建 LineCollection
    lc0 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax0.add_collection(lc0)
    legends += [
        plt.Line2D([0], [0], ls='', color='b', marker='.', ms=5, label='Train pred'),
        plt.Line2D([0], [0], ls='', color='g', marker='.', ms=5, label='Valid pred'),
        plt.Line2D([0], [0], ls='', color='r', marker='.', ms=5, label='Test pred'),
    ]
    # Shade between the lower and upper confidence bounds
    ax0.fill_between(idx[:split1], train_lower, train_upper,  color='C0', alpha=0.3)
    ax0.fill_between(idx[split1:split2], valid_lower, valid_upper,  color='C2',  alpha=0.3)
    ax0.fill_between(idx[split2:], test_ma_lower, test_ma_upper, color='C3',  alpha=0.3)
    legends.append(patches.Rectangle(
            (1, 1), 1, 1, fill=True, color='b', alpha=0.3, lw=0, label='95% Error Bars'
        ))
    ax0.legend(handles=legends[::-1], loc='upper right', bbox_to_anchor=(1.17, 1), borderaxespad=0.)

    rmse_train = root_mean_squared_error(train_y, train_post_mean)
    rmse_valid = root_mean_squared_error(valid_y, valid_post_mean)
    rmse_test = root_mean_squared_error(test_y, test_ma_post_mean)
    print(f'RMSE on train data: {rmse_train:.3f}, valid data: {rmse_valid:.3f}, test data: {rmse_test:.3f}')
    r2_train = r2_score(train_y, train_post_mean)
    r2_valid= r2_score(valid_y, valid_post_mean)
    r2_test = r2_score(test_y, test_ma_post_mean)
    print(f'R^2 on training data: {r2_train:.3f}, valid data: {r2_valid:.3f}, test data: {r2_test:.3f}')
    ax0.text(0, 0.95, f"Test: RMSE = {rmse_test:.3f}  R² = {r2_test:.3f}", transform=ax0.transAxes)


    points = np.column_stack([idx, G_scores_ma])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    lc1 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax1.add_collection(lc1)
    ax1.set_ylim(ax0.get_ylim())

    ax2.plot(idx, truth_data, c='k', ms=5)
    points = np.column_stack([idx, kf_ma_post_mean])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_colors = ['b'] * (split1-1) + ['g'] * (split2-split1) + ['r'] * (num_samples - split2 - 1)
    # 创建 LineCollection
    lc2 = LineCollection(segments, colors=segment_colors, linewidths=1)
    ax2.add_collection(lc2)
    ax2.set_ylim(ax0.get_ylim())
    rmse_train = root_mean_squared_error(train_y, kf_train_post_mean)
    rmse_valid = root_mean_squared_error(valid_y, kf_valid_post_mean)
    rmse_test = root_mean_squared_error(test_y, kf_test_ma_post_mean)
    print(f'RMSE on train data: {rmse_train:.3f}, valid data: {rmse_valid:.3f}, test data: {rmse_test:.3f}')
    r2_train = r2_score(train_y, kf_train_post_mean)
    r2_valid= r2_score(valid_y, kf_valid_post_mean)
    r2_test = r2_score(test_y, kf_test_ma_post_mean)
    print(f'R^2 on training data: {r2_train:.3f}, valid data: {r2_valid:.3f}, test data: {r2_test:.3f}')
    ax2.text(0, 0.95, f"Test: RMSE = {rmse_test:.3f}  R² = {r2_test:.3f}", transform=ax2.transAxes, va='top')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path1 = os.path.join(save_dir, f"{file_name}_{fig_labels[fig_idx]}_withoutMA.svg")
    f1.savefig(save_path1, bbox_inches='tight', format='svg')
    print(f"图像已保存至：{save_path1}")
    save_path2 = os.path.join(save_dir, f"{file_name}_{fig_labels[fig_idx]}_withMA.svg")
    f2.savefig(save_path2, bbox_inches='tight', format='svg')
    print(f"图像已保存至：{save_path2}")
    
gc.collect()

