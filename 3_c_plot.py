import pandas as pd
from matplotlib.font_manager import FontProperties
import seaborn as sns
import matplotlib.pyplot as plt

font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)  # 替换为您系统中的中文字体路径

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df_1 = pd.read_excel('data/表1-患者列表及临床信息.xlsx', engine='openpyxl')
df_head = df_1.iloc[:100]

df_head['预后分组'] = pd.cut(df_head['90天mRS'], bins=[-1, 3, 6], labels=['预后良好组', '预后不良组'], right=False)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_head, x='预后分组', y='年龄')

# 设置标题和标签
plt.title('不同预后分组的年龄分布')
plt.xlabel('预后分组')
plt.ylabel('年龄')
plt.savefig('ans/age_distribution.png', dpi=300)
# 显示图
plt.tight_layout()
plt.show()

# 对组和性别进行分组和计数
grouped = df_head.groupby(['预后分组', '性别']).size().unstack()
colors_tableau = ['#1f77b4', '#ff7f0e']
# 画饼图
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
grouped.loc['预后良好组'].plot(
    kind='pie', ax=ax[0], autopct='%1.1f%%',
    textprops={'fontsize': 14},  # 设置字体大小
    colors=colors_tableau # 设置颜色
)
grouped.loc['预后不良组'].plot(
    kind='pie', ax=ax[1], autopct='%1.1f%%',
    textprops={'fontsize': 14},  # 设置字体大小
    colors=colors_tableau # 设置颜色
)

ax[0].set_title('良好组的性别分布', fontsize=16)
ax[1].set_title('不良组的性别分布', fontsize=16)

# 移除y标签
ax[0].set_ylabel('')
ax[1].set_ylabel('')
plt.tight_layout()
plt.savefig('ans/gender_distribution.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("muted")
selected_columns = ['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']
df_selected = df_head[['预后分组'] + selected_columns]
grouped = df_selected.groupby('预后分组').mean()
plt.figure(figsize=(12, 8))
ax = grouped.T.plot(kind='bar', stacked=False, color=colors)

plt.title('不同预后分组的治疗比例', fontsize=16)
plt.xlabel('治疗', fontsize=14)
plt.ylabel('比例', fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig('ans/treatment_distribution_science_colors.png', dpi=300)

# 选择需要的列
colors = ['#27AE60', '#F1C40F']
selected_columns = ['高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史', '吸烟史', '饮酒史']
df_selected = df_head[['预后分组'] + selected_columns]

# 使用上面的代码来计算每个分组的病史比例并进行可视化
grouped = df_selected.groupby('预后分组').mean()

# 画图
plt.figure(figsize=(12, 8))
ax = grouped.T.plot(kind='bar', stacked=False, color=colors)

# 设置标题和标签
plt.title('不同预后分组的病史比例', fontsize=16)
plt.xlabel('病史', fontsize=14)
plt.ylabel('比例', fontsize=14)

# 显示图
plt.tight_layout()
plt.show()

# 保存图像为高分辨率
plt.savefig('ans/disease_image.png', dpi=300)

import pandas as pd
import matplotlib.pyplot as plt

df_2 = pd.read_excel('data/表2-患者影像信息血肿及水肿的体积及位置2.xlsx', engine='openpyxl')
df_1 = df_head
colors = ['#3498DB', '#D4CEF6']

column_2_data = df_2.iloc[:, 2]

df_2['Group'] = ['>30' if x > 30 else '<=30' for x in column_2_data]

merged_df = pd.concat([df_1['预后分组'], df_2['Group']], axis=1)

grouped = merged_df.groupby(['预后分组', 'Group']).size().unstack()

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
grouped.loc['预后良好组'].plot(
    kind='pie', ax=ax[0], autopct='%1.1f%%',
    textprops={'fontsize': 15},  # 设置字体大小
    colors=colors  # 设置颜色
)
grouped.loc['预后不良组'].plot(
    kind='pie', ax=ax[1], autopct='%1.1f%%',
    textprops={'fontsize': 16},  # 设置字体大小
    colors=colors  # 设置颜色
)

# 设置标题
ax[0].set_title('首次检查血肿体积', fontsize=19)
ax[1].set_title('首次检查血肿体积例', fontsize=19)

# 移除y标签
ax[0].set_ylabel('')
ax[1].set_ylabel('')

# 显示图
plt.tight_layout()
plt.show()

# 保存图像为高分辨率
plt.savefig('ans/column_2_distribution_rainbow_colors.png', dpi=300)
