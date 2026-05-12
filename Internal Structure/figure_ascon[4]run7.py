import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# 专业SCI论文样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16,
    'font.family': 'Arial'
})


# 数据
runs = np.array([1, 2, 3, 4, 5, 6])
runcount = np.array(['0\n(0001)','0\n(0011)','0\n(0111)','1\n(1100)','1\n(1110)','1\n(1000)'])
accuracy = np.array([99.91,99.71,99.58,64.42,63.17,53.12])

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
ax.set_xlabel('Number of AA Runs', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
#ax.set_title('Four-round Ascon-p\' permutation 7-run combination distinguishing performance and AA run relationship diagram',
#             fontweight='bold', pad=12)

# 设置刻度
ax.set_xticks(runs,labels=[f'{x}' for x in runcount],fontsize=16)
ax.set_ylim(50, 105)
ax.set_yticks(np.arange(50, 105, 5))

# 添加数据标签
for i, (run, acc) in enumerate(zip(runs, accuracy)):
    ax.text(run, acc + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=16)

# 添加趋势注释
'''
slope, intercept, r_value, p_value, std_err = stats.linregress(runs, accuracy)
ax.text(0.02, 0.98, f'Slope: {slope:.2f} %/run\nR² = {r_value**2:.3f}',
        transform=ax.transAxes, fontsize=16,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')
'''
plt.tight_layout()
plt.savefig('四轮Ascon-p‘置换7游程组合区分性能与AA游程关系图.eps', format='eps')
plt.savefig('四轮Ascon-p‘置换7游程组合区分性能与AA游程关系图.png', dpi=300, bbox_inches='tight')  # 更高DPI用于印刷
plt.show()