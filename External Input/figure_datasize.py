import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


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
    '数据量': [1000, 10000, 100000, 1000000, 10000000],
    #'数据量': [10000000, 1000000, 100000, 10000, 1000],
    #'数据量': [1, 2, 3, 4, 5],
    #'Ascon-p[3]-fix': [1,0.9917,0.9817,0.6408,0.583],
    #'Ascon-p[3]-change': [1, 0.9997, 0.9806, 0.6277, 0.552],
    #'Ascon-p[4]-fix': [0.5004, 0.5030, 0.5055, 0.5094, 0.516],
    #'Ascon-p[4]-change': [0.5005,0.5025,0.5057,0.5042,0.537],

    'Ascon-p[3]-fix': [ 0.583, 0.6408, 0.9817,0.9917, 1],
    #'Ascon-p[3]-change': [0.552, 0.6277, 0.9806,0.9997, 1],
    #'Ascon-p[4]-fix': [0.516,0.5094,  0.5055, 0.5030,0.5004],
    'Ascon-p[4]-fix': [0.537, 0.5042, 0.5057,0.5025 ,0.5005],
}
df = pd.DataFrame(data)

# 创建图形
plt.figure(figsize=(14, 8))
# 定义颜色和标记
#colors = ['#8A2BE2', '#0000FF', '#00FF00', '#FFA500', '#FF0000']
#markers = ['o', 's', '^', 'D', 'v']
#labels = ['S4', 'S3', 'S2', 'S1', 'S0']
# 定义颜色和标记
colors = ['#0000FF', '#00FF00', '#FFA500', '#FF0000']
markers = ['o', 's', '^', 'D']
#labels = ['Ascon-p[3]-fix', 'Ascon-p[3]-change', 'Ascon-p[4]-fix','Ascon-p[4]-change']
labels = ['Ascon-p[3]', 'Ascon-p[4]']

# 绘制每个准确率系列的散点折线图
#for i, col in enumerate( ['Ascon-p[3]-fix', 'Ascon-p[3]-change', 'Ascon-p[4]-fix','Ascon-p[4]-change']):
for i, col in enumerate( ['Ascon-p[3]-fix', 'Ascon-p[4]-fix']):
    # 按汉明重量排序，确保折线连接正确
    sorted_df = df.sort_values('数据量')

    # 绘制折线



    x_positions = np.arange(len(df['数据量']))
    plt.plot(x_positions, sorted_df[col],
             color=colors[i],
             linewidth=2,
             alpha=0.7,
             label=labels[i])

    plt.scatter(x_positions, df[col],
                color=colors[i],
                marker=markers[i],
                s=80,
                edgecolors='white',
                linewidth=1.5,
                alpha=0.8,
                label=f'{labels[i]} (散点)')
    '''
    plt.plot(sorted_df['数据量'], sorted_df[col],
             color=colors[i],
             linewidth=2,
             alpha=0.7,
             label=labels[i])

    plt.scatter(df['数据量'], df[col],
                color=colors[i],
                marker=markers[i],
                s=80,
                edgecolors='white',
                linewidth=1.5,
                alpha=0.8,
                label=f'{labels[i]} (散点)')
    '''


# 设置图形属性

#plt.title('Performance Differentiation of Datasets of Different Sizes', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Data scale', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
#labels_data = ['1000', '10000', '100000', '1000000','10000000'],
# 设置x轴刻度
#plt.xticks(fontsize=16)

plt.xticks(x_positions,labels=[f'{x}' for x in df['数据量']],fontsize=16)
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
    #if not any(label.startswith(l) for l in ['Ascon-p[3]-fix', 'Ascon-p[3]-change', 'Ascon-p[4]-fix','Ascon-p[4]-change'] if f'{l} (散点)' in label):
    if not any(label.startswith(l) for l in ['Ascon-p[3]-fix',  'Ascon-p[4]-fix'] if f'{l} (散点)' in label):
        base_label = label.replace(' (散点)', '')
        if base_label not in seen_labels:
            seen_labels.add(base_label)
            unique_labels.append(base_label)
            unique_handles.append(handle)

plt.legend(unique_handles, unique_labels, loc='best', fontsize=16)
#plt.text(2.5, 0.8, 'S0', fontsize=16, color='red')
#plt.text(2.5, 0.76, 'S2', fontsize=16, color='green')
#plt.text(2.5, 0.72, 'S4', fontsize=16, color='purple')

# 调整布局
plt.tight_layout()

plt.savefig('仅大小不同规模数据集区分性能.pdf', format='pdf')
plt.savefig('仅大小不同规模数据集区分性能.png', dpi=300)  # 更高DPI用于印刷

plt.show()
# 也可以保存图形
#plt.savefig('汉明重量_准确率_散点折线图.png', dpi=300, bbox_inches='tight')