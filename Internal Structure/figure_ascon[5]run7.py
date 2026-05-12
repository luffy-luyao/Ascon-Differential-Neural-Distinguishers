import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# 专业SCI论文样式设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
    'font.family': 'Arial'
})

# 数据
#runs = np.arange(1, 33, 1)
runs = np.array([1, 2, 3, 4, 5, 6 ,7 ,8])
runcount = np.array(['1\n(01011)','1\n(01001)','1\n(00101)','1\n(01101)','2\n(11010)','2\n(10110)','2\n(10010)','2\n(10100)'])
accuracy = np.array([99.98,99.38,99.00,98.44,55.41,51.03,50.85,50.67])

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
#ax.set_title('Five round Ascon-p\' permutation 7-run combination distinguishing performance and AA run relationship diagram',
#             fontweight='bold', pad=12)

# 设置刻度
ax.set_xticks(runs,labels=[f'{x}' for x in runcount],fontsize=12)
ax.set_ylim(50, 105)
ax.set_yticks(np.arange(50, 105, 5))

# 添加数据标签
for i, (run, acc) in enumerate(zip(runs, accuracy)):
    ax.text(run, acc + 1.5, f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=14)

# 添加趋势注释
'''
slope, intercept, r_value, p_value, std_err = stats.linregress(runs, accuracy)
ax.text(0.02, 0.98, f'Slope: {slope:.2f} %/run\nR² = {r_value**2:.3f}',
        transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')
'''
plt.tight_layout()
plt.savefig('五轮Ascon-p‘置换7游程组合区分性能与AA游程关系图.eps', format='eps')
plt.savefig('五轮Ascon-p‘置换7游程组合区分性能与AA游程关系图.png', dpi=300, bbox_inches='tight')  # 更高DPI用于印刷
plt.show()