import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (保持不变，精确到小数点后四位，显示时自动截取三位)
# ==========================================
categories = ['pedestrian', 'people', 'bicycle', 'car', 'van',
              'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'All class']

# YOLO11(n)
yolo_11_n = [0.218, 0.114, 0.0532, 0.661, 0.301, 0.315, 0.114, 0.142, 0.499, 0.224, 0.264]
# Drone-YOLO(n)
drone_yolo_n = [0.281, 0.166, 0.0765, 0.704, 0.327, 0.290, 0.122, 0.136, 0.496, 0.258, 0.286]
# PE-YOLO(n)
pe_yolo_n = [0.442, 0.509, 0.266, 0.653, 0.390,
             0.416, 0.265, 0.393, 0.636, 0.403, 0.437]

# ==========================================
# 2. 绘图参数
# ==========================================
x = np.arange(len(categories))
width = 0.25  # 柱宽

# 画布宽度设为 16，高度设为 8，确保水平放数字有足够空间
fig, ax = plt.subplots(figsize=(16, 8))

# ==========================================
# 3. 绘制柱状图
# ==========================================
rects1 = ax.bar(x - width, yolo_11_n, width, label='YOLO11(n)', color='#8ecfc9', edgecolor='white', linewidth=0.5)
rects2 = ax.bar(x, drone_yolo_n, width, label='Drone-YOLO(n)', color='#82b6e9', edgecolor='white', linewidth=0.5)
rects3 = ax.bar(x + width, pe_yolo_n, width, label='PE-YOLO(n)', color='#ff6b6b', edgecolor='white', linewidth=0.5)

# ==========================================
# 4. 图表细节设置
# ==========================================
ax.set_ylabel('mAP50', fontsize=14, fontweight='bold')
ax.set_xticks(x)
# X轴类别名称稍微大一点，保持倾斜以免重叠
ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=12)

# 【修改点1】将图例固定在 "右上角"
ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

# Y轴范围稍微加大，防止右上角的图例挡住最高的柱子
ax.set_ylim(0, 0.90)

ax.grid(axis='y', linestyle='--', alpha=0.3)

# ==========================================
# 5. 数值标签 (正放)
# ==========================================
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # 【修改点2】rotation=0 (正放), fontsize=8.5 (字号适中)
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8.5,  # 如果觉得数字挤，可以改小到 7 或 8
                    rotation=0)    # <--- 这里控制正放

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# ==========================================
# 6. 保存与显示
# ==========================================
plt.tight_layout()
plt.savefig('mAP_Comparison_Final.png', dpi=300, bbox_inches='tight')
plt.show()