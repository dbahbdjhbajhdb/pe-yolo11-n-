import cv2
import numpy as np
from ultralytics import YOLO

# ================= 配置区域 (请修改这里) =================
# 1. 图片路径 (找一张密密麻麻全是人的图)
IMG_PATH = r"C:\date\Visdrone2019\images\test2019\0000078_03171_d_0000009.jpg"

# 2. 基线模型路径 (Baseline / YOLOv11n 原版)
MODEL_A_PATH = r'C:\ultralytics-main\runs\detect\YOLO11(L)-VISDRONE2019-300\weights\best.pt'  # 或者你训练的基线 best.pt

# 3. 改进模型路径 (Ours / RDAttention)
MODEL_B_PATH = r'C:\ultralytics-main\runs\detect\SO-YOLO（L）-VISDRONE2019-300\weights\best.pt'

# 4. 输出文件名 (这样就不会覆盖了！)
OUTPUT_NAME = 'final_comparison_result2.jpg'


# =======================================================

def run_comparison():
    # 1. 读取原图
    origin_img = cv2.imread(IMG_PATH)
    if origin_img is None:
        print(f"❌ 找不到图片: {IMG_PATH}")
        return

    print("🚀 正在加载模型并推理...")

    # 2. 跑模型 A (基线)
    model_a = YOLO(MODEL_A_PATH)
    res_a = model_a.predict(IMG_PATH, conf=0.25)[0]
    # 关键点：labels=False, conf=False 去掉烦人的文字
    # line_width=2 让框稍微细一点，适合密集场景
    img_a = res_a.plot(labels=False, conf=False, line_width=2)

    # 3. 跑模型 B (改进版)
    model_b = YOLO(MODEL_B_PATH)
    res_b = model_b.predict(IMG_PATH, conf=0.25)[0]
    img_b = res_b.plot(labels=False, conf=False, line_width=2)

    # 4. 给图片加标题 (在图片顶部加个白条写字)
    def add_title(img, text):
        h, w = img.shape[:2]
        # 加一个白色头部区域
        header = np.full((50, w, 3), 255, dtype=np.uint8)
        # 写字 (黑色)
        cv2.putText(header, text, (int(w / 2) - 100, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return np.vstack((header, img))

    img_a_with_title = add_title(img_a, "Baseline (YOLOv11n)")
    img_b_with_title = add_title(img_b, "Ours (PE-YOLO)")

    # 5. 左右拼接 (Side-by-Side)
    # 确保两张图高度一致 (通常是一样的，为了保险起见)
    h_a, w_a = img_a_with_title.shape[:2]
    h_b, w_b = img_b_with_title.shape[:2]

    if h_a != h_b:
        img_b_with_title = cv2.resize(img_b_with_title, (w_a, h_a))

    # 拼接！
    final_comparison = np.hstack((img_a_with_title, img_b_with_title))

    # 6. 保存
    cv2.imwrite(OUTPUT_NAME, final_comparison)
    print(f"✅ 对比图已生成！请查看: {OUTPUT_NAME}")

    # (可选) 弹窗显示，按任意键关闭
    # cv2.imshow("Comparison", final_comparison)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    run_comparison()