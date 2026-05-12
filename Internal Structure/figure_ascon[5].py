import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# 专业SCI论文样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'font.family': 'Arial'
})

# 数据
runs = np.arange(1, 33, 1)
runcount = np.array([6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,10,10])
accuracy = np.array([100,99.34,99.98,99.38,99.00,98.44,55.41,51.03,50.85,50.67,50.45,50.39,50.35,50.34,50.32,50.27,50.25,50.21,50.21,50.17,50.16,50.14,50.35,50.27,50.18,50.09,50.07,50,50,50.12,50,50])


# 创建图形
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# 创建柱状图
colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(runs)))
bars = ax.bar(runs, accuracy, color=colors, width=0.6,
              edgecolor='black', linewidth=0.5)

# 添加误差条（示例数据，你可以替换为真实误差）
#error = np.array([1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5])
#ax.errorbar(runs, accuracy, yerr=error, fmt='none',
#            ecolor='black', elinewidth=1, capsize=3)

# 设置标签
ax.set_xlabel('Number of Runs', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Variant Ascon-p[5] Permutation Run Statistics',
             fontweight='bold', pad=12)

# 设置刻度
ax.set_xticks(runs,labels=[f'{x}' for x in runcount],fontsize=10)
ax.set_ylim(50, 105)
ax.set_yticks(np.arange(50, 105, 5))

# 添加数据标签
for i, (run, acc) in enumerate(zip(runs, accuracy)):
    ax.text(run, acc + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=3)

# 添加趋势注释
'''
slope, intercept, r_value, p_value, std_err = stats.linregress(runs, accuracy)
ax.text(0.02, 0.98, f'Slope: {slope:.2f} %/run\nR² = {r_value**2:.3f}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')
'''
plt.tight_layout()
plt.savefig('变种Ascon-p[5]置换游程统计表.pdf', format='pdf')
plt.savefig('变种Ascon-p[5]置换游程统计表.png', dpi=300, bbox_inches='tight')  # 更高DPI用于印刷
plt.show()