import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# 专业SCI论文样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'font.family': 'Arial'
})

# 数据
runs = np.array([1, 2, 3, 4])
runcount = np.array(['8\n(0101010)','8\n(1010101)','9\n(0101011)','9\n(1010100)'])
accuracy = np.array([100,59.12,50.30,50.22])

# 创建图形
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# 创建柱状图
colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(runs)))
bars = ax.bar(runs, accuracy, color=colors, width=0.4,
              edgecolor='black', linewidth=0.5)

# 添加误差条（示例数据，你可以替换为真实误差）
#error = np.array([1.2, 1.1, 1.3, 1.2, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5])
#ax.errorbar(runs, accuracy, yerr=error, fmt='none',
#            ecolor='black', elinewidth=1, capsize=3)

# 设置标签
ax.set_xlabel('Number of Runs', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
#ax.set_title('Seven round Ascon-p\' permutation Run Statistics',
#             fontweight='bold', pad=12)

# 设置刻度
ax.set_xticks(runs,labels=[f'{x}' for x in runcount],fontsize=12)
ax.set_ylim(50, 105)
ax.set_yticks(np.arange(50, 105, 5))

# 添加数据标签
for i, (run, acc) in enumerate(zip(runs, accuracy)):
    ax.text(run, acc + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=12)

# 添加趋势注释
'''
slope, intercept, r_value, p_value, std_err = stats.linregress(runs, accuracy)
ax.text(0.02, 0.98, f'Slope: {slope:.2f} %/run\nR² = {r_value**2:.3f}',
        transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')
'''
plt.tight_layout()
plt.savefig('figure12_ Ascon-p[7]_run.pdf', format='pdf')
plt.savefig('figure12_ Ascon-p[7]_run.png', dpi=300, bbox_inches='tight')  # 更高DPI用于印刷
plt.show()