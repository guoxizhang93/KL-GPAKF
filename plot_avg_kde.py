import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from scipy.stats import ttest_rel
import seaborn as sns
from scipy.stats import gaussian_kde, bootstrap

# 数据
data = {
    'Subject': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5', 'S6', 'S6', 'S7', 'S7'],
    'Trial': ['T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2', 'T1', 'T2'],
    'Index_H_OGP': [0.101, 0.099, 0.105, 0.118, 0.078, 0.070, 0.078, 0.091, 0.077, 0.114, 0.056, 0.071, 0.064, 0.069],
    'Index_L_OGP': [0.061, 0.141, 0.048, 0.104, 0.052, 0.047, 0.056, 0.073, 0.057, 0.086, 0.061, 0.048, 0.017, 0.058],
    'Index_H_MGP': [0.085, 0.065, 0.1, 0.110, 0.074, 0.071, 0.072, 0.086, 0.073, 0.110, 0.049, 0.066, 0.052, 0.068],
    'Index_L_MGP': [0.056, 0.057, 0.048, 0.117, 0.051, 0.044, 0.058, 0.042, 0.055, 0.083, 0.052, 0.048, 0.008, 0.052],
    'middle_H_OGP': [0.063, 0.157, 0.110, 0.119, 0.117, 0.107, 0.138, 0.166, 0.071, 0.106, 0.121, 0.135, 0.117, 0.115],
    'middle_L_OGP': [0.066, 0.129, 0.059, 0.020, 0.213, 0.087, 0.031, 0.159, 0.121, 0.037, 0.168, 0.088, 0.117, 0.312],
    'middle_H_MGP': [0.058, 0.145, 0.105, 0.116, 0.116, 0.103, 0.136, 0.156, 0.058, 0.1, 0.093, 0.125, 0.117, 0.111],
    'middle_L_MGP': [0.064, 0.077, 0.053, 0.019, 0.223, 0.048, 0.035, 0.108, 0.120, 0.031, 0.047, 0.085, 0.117, 0.180],
    'ring_H_OGP': [0.110, 0.095, 0.069, 0.077, 0.109, 0.060, 0.100, 0.132, 0.174, 0.181, 0.101, 0.114, 0.090, 0.106],
    'ring_L_OGP': [0.261, 0.142, 0.125, 0.057, 0.107, 0.087, 0.204, 0.257, 0.184, 0.412, 0.167, 0.157, 0.090, 0.111],
    'ring_H_MGP': [0.107, 0.085, 0.067, 0.073, 0.092, 0.058, 0.094, 0.120, 0.171, 0.174, 0.096, 0.113, 0.082, 0.098],
    'ring_L_MGP': [0.226, 0.102, 0.119, 0.048, 0.116, 0.094, 0.104, 0.224, 0.183, 0.4, 0.1, 0.131, 0.082, 0.098]
}
df = pd.DataFrame(data)

df['Avg_RMSE_H_OGP'] = df[['Index_H_OGP', 'middle_H_OGP', 'ring_H_OGP']].mean(axis=1)
df['Avg_RMSE_L_OGP'] = df[['Index_L_OGP', 'middle_L_OGP', 'ring_L_OGP']].mean(axis=1)
df['Avg_RMSE_H_MGP'] = df[['Index_H_MGP', 'middle_H_MGP', 'ring_H_MGP']].mean(axis=1)
df['Avg_RMSE_L_MGP'] = df[['Index_L_MGP', 'middle_L_MGP', 'ring_L_MGP']].mean(axis=1)

# 定义颜色
colors = ['#5ca8d9', '#e16b6b']

def add_stat_annotation(ax, p_value, x1, x2, y_max, color='black'):
    star_label = ""
    if p_value < 0.001:
        star_label = "***"
    elif p_value < 0.01:
        star_label = "**"
    elif p_value < 0.05:
        star_label = "*"
    elif p_value < 0.1:
        star_label = "†"
    else:
        star_label = "ns"
    h = y_max * 0.015
    ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max],
            lw=1.5, c=color)
    ax.text((x1 + x2) * 0.5, y_max + h, star_label, ha='center', va='bottom',
            color=color, fontsize=16, fontweight='bold')

def add_mode_annotation(ax, data, color):
    """Adds a marker and text label for the mode (peak) of a KDE curve."""
    kde_fit = gaussian_kde(data)
    y_range = np.linspace(data.min() - 0.01, data.max() + 0.01, 200)
    x_dens = kde_fit.evaluate(y_range)
    mode_y = y_range[np.argmax(x_dens)]
    mode_x = np.max(x_dens)
    
    ax.plot(mode_x, mode_y, 'o', color=color, ms=8, zorder=10)
    ax.text(mode_x, mode_y, f' {mode_y:.3f}', ha='left', va='center', fontsize=10, color=color, fontweight='bold')

def add_stat_annotation_ci(ax, confidence_interval, x1, x2, y_max, color='black'):
    """
    基于引导法置信区间在图上添加统计显著性标注。
    :param ax: Matplotlib axes 对象
    :param confidence_interval: 引导法生成的置信区间
    :param x1: 第一个数据点的 x 坐标
    :param x2: 第二个数据点的 x 坐标
    :param y_max: 标注线的高度
    :param color: 标注线的颜色
    """
    # 检查置信区间是否包含0
    is_significant = not (confidence_interval.low <= 0 and confidence_interval.high >= 0)
    
    # 根据显著性设置标注
    star_label = ""
    if is_significant:
        star_label = "*"
    else:
        star_label = "ns" # 不显著

    # 绘制连接线
    h = y_max * 0.015
    ax.plot([x1, x1, x2, x2], [y_max, y_max + h, y_max + h, y_max],
            lw=1.5, c=color)

    # 绘制星号
    ax.text((x1 + x2) * 0.5, y_max + h, star_label, ha='center', va='bottom',
            color=color, fontsize=16, fontweight='bold')

def median_difference(x, y, axis=0):
    """
    计算两组数据的中位数差。
    """
    return np.median(x, axis=axis) - np.median(y, axis=axis)

# 创建 1x2 的子图布局
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.patch.set_visible(False)
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- 获取 Avg RMSE 的纵坐标范围 ---
avg_cols = ["Avg_RMSE_H_OGP", "Avg_RMSE_L_OGP", "Avg_RMSE_H_MGP", "Avg_RMSE_L_MGP"]
all_avg_values = df[avg_cols].values.flatten()
global_ymin = np.min(all_avg_values)
global_ymax = np.max(all_avg_values)
y_padding = (global_ymax - global_ymin) * 0.15
final_ymin = global_ymin - y_padding * 0.2
final_ymax = global_ymax + y_padding

# --- 绘制 Avg 箱线图 ---
ax_avg_boxplot = axes[0]
ax_avg_boxplot.set_title('Avg RMSE', fontsize=14)
ax_avg_boxplot.set_ylabel("RMSE")
ax_avg_boxplot.set_ylim(final_ymin, final_ymax)
ax_avg_boxplot.set_xticks([1.5, 4.5])
ax_avg_boxplot.set_xticklabels(['H', 'L'])

ogp_positions = [1, 4]
mgp_positions = [2, 5]
# 绘制 OGP (实线)
for i, col in enumerate(avg_cols[:2]):
    y = df[col]
    bp = ax_avg_boxplot.boxplot([y], positions=[ogp_positions[i]], widths=0.6, patch_artist=True, showfliers=False,
                                boxprops=dict(facecolor=to_rgba(colors[i], alpha=0.2), edgecolor=to_rgba(colors[i], alpha=1.0), linewidth=1.5, linestyle='-'),
                                whiskerprops=dict(color=colors[i], linewidth=1.5, linestyle='-'),
                                capprops=dict(color=colors[i], linewidth=1.5, linestyle='-'),
                                medianprops=dict(color=colors[i], linewidth=2, linestyle='-'))
    for median_line in bp['medians']:
        median_value = median_line.get_ydata()[0]
        x_pos = median_line.get_xdata()[0]
        ax_avg_boxplot.text(x_pos, median_value, f'{median_value:.3f}', ha='center', va='bottom', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax_avg_boxplot.scatter(np.random.normal(ogp_positions[i], 0.05, size=len(y)), y, color=colors[i], alpha=0.8, s=20, zorder=3)

# 绘制 MGP (虚线)
for i, col in enumerate(avg_cols[2:]):
    y = df[col]
    bp = ax_avg_boxplot.boxplot([y], positions=[mgp_positions[i]], widths=0.6, patch_artist=True, showfliers=False,
                                boxprops=dict(facecolor=to_rgba(colors[i], alpha=0.2), edgecolor=to_rgba(colors[i], alpha=1.0), linewidth=1.5, linestyle='--'),
                                whiskerprops=dict(color=colors[i], linewidth=1.5, linestyle='--'),
                                capprops=dict(color=colors[i], linewidth=1.5, linestyle='--'),
                                medianprops=dict(color=colors[i], linewidth=2, linestyle='--'))
    for median_line in bp['medians']:
        median_value = median_line.get_ydata()[0]
        x_pos = median_line.get_xdata()[0]
        ax_avg_boxplot.text(x_pos, median_value, f'{median_value:.3f}', ha='center', va='bottom', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax_avg_boxplot.scatter(np.random.normal(mgp_positions[i], 0.05, size=len(y)), y, color=colors[i], alpha=0.8, s=20, zorder=3)

# 统计显著性标注
y_max_annot = final_ymax - (final_ymax - global_ymax) / 2
for i in range(2):
    stat, p_value = ttest_rel(df[avg_cols[i]], df[avg_cols[i+2]])
    add_stat_annotation(ax_avg_boxplot, p_value, ogp_positions[i], mgp_positions[i], y_max_annot, colors[i])
ax_avg_boxplot.axvline(3, linestyle="--", color="black", zorder=0, alpha=0.5)


# --- 绘制 KDE 图 ---
ax_kde = axes[1]
ax_kde.spines['top'].set_visible(False)
ax_kde.spines['right'].set_visible(False)
ax_kde.spines['left'].set_visible(True)
ax_kde.grid(False)

# 设置标题和标签
ax_kde.set_title('Avg RMSE KDE', fontsize=14)
ax_kde.set_ylabel("RMSE")
ax_kde.set_xlabel('Density', fontsize=12)
ax_kde.tick_params(labelleft=False)  # 隐藏左侧 Y 轴标签

# 将 KDE 图的 Y 轴范围设置为与箱线图相同
ax_kde.set_ylim(final_ymin, final_ymax)
ax_kde.set_xlim(left=0, right = 28)

# 绘制 KDE 曲线并添加众数标注
sns.kdeplot(y=df['Avg_RMSE_H_OGP'], ax=ax_kde, color=colors[0], linestyle='-', linewidth=2, label='H-OGP', fill=True, alpha=0.3)
add_mode_annotation(ax_kde, df['Avg_RMSE_H_OGP'], colors[0])

sns.kdeplot(y=df['Avg_RMSE_L_OGP'], ax=ax_kde, color=colors[1], linestyle='-', linewidth=2, label='L-OGP', fill=True, alpha=0.3)
add_mode_annotation(ax_kde, df['Avg_RMSE_L_OGP'], colors[1])

sns.kdeplot(y=df['Avg_RMSE_H_MGP'], ax=ax_kde, color=colors[0], linestyle='--', linewidth=2, label='H-MGP', fill=True, alpha=0.3)
add_mode_annotation(ax_kde, df['Avg_RMSE_H_MGP'], colors[0])

sns.kdeplot(y=df['Avg_RMSE_L_MGP'], ax=ax_kde, color=colors[1], linestyle='--', linewidth=2, label='L-MGP', fill=True, alpha=0.3)
add_mode_annotation(ax_kde, df['Avg_RMSE_L_MGP'], colors[1])


# 创建自定义图例
custom_lines = [
    plt.Line2D([0], [0], color=colors[0], linestyle='-', lw=2),
    plt.Line2D([0], [0], color=colors[0], linestyle='--', lw=2),
    plt.Line2D([0], [0], color=colors[1], linestyle='-', lw=2),
    plt.Line2D([0], [0], color=colors[1], linestyle='--', lw=2),
]
ax_kde.legend(custom_lines, ['H-OGP', 'H-MGP', 'L-OGP', 'L-MGP'], loc='upper right', title='Group')

# --- 绘制森林图 ---
ax_forest = axes[2]
ax_forest.set_title('Avg RMSE (Forest Plot)', fontsize=14)
ax_forest.set_ylabel("RMSE")
ax_forest.spines['bottom'].set_visible(False)
ax_forest.grid(True, linestyle='--', alpha=0.6, axis='y')

# 定义绘制顺序和标签
groups = ['H-OGP', 'H-MGP', 'L-OGP', 'L-MGP']
data_columns = ['Avg_RMSE_H_OGP', 'Avg_RMSE_H_MGP', 'Avg_RMSE_L_OGP', 'Avg_RMSE_L_MGP']
plot_colors = [colors[0], colors[0], colors[1], colors[1]]
line_styles = ['-', '--', '-', '--']
x_positions = [1, 2, 3, 4]

# 绘制每个组别的中位数和置信区间
for i, col in enumerate(data_columns):
    data = df[col].to_numpy()
    median_val = np.median(data)
    
    # 引导法计算中位数置信区间
    res_median = bootstrap((data,), statistic=np.median, n_resamples=9999, paired=False, random_state=42)
    ci_low, ci_high = res_median.confidence_interval

    # 绘制置信区间（横线）
    ax_forest.vlines(x_positions[i], ci_low, ci_high, color=plot_colors[i], linestyle=line_styles[i], linewidth=2.5)
    
    # 绘制中位数（点）
    ax_forest.plot(x_positions[i], median_val, 'o', color=plot_colors[i], markersize=10, zorder=10)
    
    # 标注中位数数值
    ax_forest.text(x_positions[i], median_val + 0.005, f'{median_val:.3f}', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    
# 绘制水平分隔线
ax_forest.axhline(y=0.08, color='grey', linestyle=':', zorder=0)

# 设置Y轴范围与箱线图一致，以保持视觉统一
ax_forest.set_ylim(final_ymin, final_ymax)
ax_forest.set_xticks(x_positions)
ax_forest.set_xticklabels(groups)

plt.tight_layout()
plt.savefig('rmse_avg_kde.pdf', bbox_inches='tight')
plt.show()