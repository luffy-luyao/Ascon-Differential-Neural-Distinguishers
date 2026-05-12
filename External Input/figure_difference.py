import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 方法1：设置字体为系统支持的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 准备数据
'''
data = {
    '汉明重量': [1, 2, 3, 4, 5, 6, 13, 15, 22, 23, 24, 29, 30],
    'S4': [0.99,0.89, 0.55,0.5045, 0.5025, 0.5013, 0.5019, 0.5028, 0.5033, 0.5004,
             0.5031, 0.5018, 0.5028],
    'S3': [0.99,  0.82, 0.53, 0.5033, 0.5026, 0.5033, 0.5038, 0.5017, 0.5027, 0.5019, 0.5015,
             0.5021, 0.5035],
    'S2': [0.99, 0.93, 0.59, 0.57, 0.5612, 0.5034, 0.5022, 0.5037, 0.5008, 0.5014, 0.5017,
             0.5013, 0.5044],
    'S1': [0.99,  0.83,0.52, 0.51, 0.5019, 0.5006, 0.5025, 0.502, 0.5044, 0.5021,
             0.5013, 0.5034, 0.5016],
    'S0': [0.99, 0.96,  0.64,  0.57, 0.5304, 0.5048, 0.5027, 0.5026, 0.5029, 0.5027,
             0.5011, 0.5045, 0.5016]
}
'''

data = {
    '汉明重量': [1, 2, 3, 4],
    'S4': [0.99,0.89, 0.55,0.5045],
    'S3': [0.99,  0.82, 0.53, 0.5033],
    'S2': [0.99, 0.93, 0.59, 0.57],
    'S1': [0.99,  0.83,0.52, 0.51],
    'S0': [0.99, 0.96,  0.64,  0.57]
}
df = pd.DataFrame(data)

# 创建图形
plt.figure(figsize=(14, 8))

# 定义颜色和标记
colors = ['#8A2BE2', '#0000FF', '#00FF00', '#FFA500', '#FF0000']
markers = ['o', 's', '^', 'D', 'v']
labels = ['S4', 'S3', 'S2', 'S1', 'S0']

# 绘制每个准确率系列的散点折线图
for i, col in enumerate(['S4', 'S3', 'S2', 'S1', 'S0']):
    # 按汉明重量排序，确保折线连接正确
    sorted_df = df.sort_values('汉明重量')

    # 绘制折线

    plt.plot(sorted_df['汉明重量'], sorted_df[col],
             color=colors[i],
             linewidth=2,
             alpha=0.7,
             label=labels[i])

    plt.scatter(df['汉明重量'], df[col],
                color=colors[i],
                marker=markers[i],
                s=80,
                edgecolors='white',
                linewidth=1.5,
                alpha=0.8,
                label=f'{labels[i]} (散点)')


# 设置图形属性
#plt.title('Average classification accuracy at different Hamming weights', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Hamming Weight', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

# 设置x轴刻度
plt.xticks(fontsize=16)
plt.yticks(np.arange(0.5, 1.05, 0.05), fontsize=16)

# 添加网格
#plt.grid(True, alpha=0.3)

# 调整图例，避免重复显示
handles, labels = plt.gca().get_legend_handles_labels()
# 只显示每个系列的第一个图例项（去掉散点的图例）
unique_labels = []
unique_handles = []
seen_labels = set()
for handle, label in zip(handles, labels):
    if not any(label.startswith(l) for l in ['S0', 'S1', 'S2', 'S3', 'S4'] if f'{l} (散点)' in label):
        base_label = label.replace(' (散点)', '')
        if base_label not in seen_labels:
            seen_labels.add(base_label)
            unique_labels.append(base_label)
            unique_handles.append(handle)

plt.legend(unique_handles, unique_labels, loc='upper right', fontsize=10)
plt.text(2.5, 0.8, 'S0', fontsize=16, color='red')
plt.text(2.5, 0.76, 'S2', fontsize=16, color='green')
plt.text(2.5, 0.72, 'S4', fontsize=16, color='purple')

# 调整布局
plt.tight_layout()

plt.savefig('不同汉明重量下平均区分准确率.pdf', format='pdf')
plt.savefig('不同汉明重量下平均区分准确率.png', dpi=300)  # 更高DPI用于印刷

plt.show()
# 也可以保存图形
#plt.savefig('汉明重量_准确率_散点折线图.png', dpi=300, bbox_inches='tight')