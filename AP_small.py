import json
import os
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ================= 🔧 配置区域 =================

# 1. 你的 VisDrone 原始标签文件夹
GT_TXT_DIR = r"C:\date\visdrone2019-coco\label\test2019"

# 2. 你的 YOLO 预测结果 JSON
PRED_JSON_PATH = r"C:\ultralytics-main\runs\detect\best-test\predictions.json"


# ===============================================

def generate_robust_gt_json(txt_dir, pred_json_path, save_path='visdrone_val_gt_final_v2.json'):
    print(f"🔄 正在生成真值 (修正类别映射)...")

    if not os.path.exists(pred_json_path):
        print(f"❌ 找不到文件: {pred_json_path}")
        return None

    with open(pred_json_path, 'r') as f:
        preds = json.load(f)

    # 获取预测中所有的 image_id
    pred_image_ids = list(set(p['image_id'] for p in preds))
    pred_image_ids.sort()
    print(f"🕵️ 预测结果包含 {len(pred_image_ids)} 张图片。")

    dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 修正类别定义：直接对应原始 ID (1-10)
    # 虽然 category_id 可以是任意值，但为了对应，我们保留原始 ID
    categories = {
        1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
        6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor"
    }
    for cid, name in categories.items():
        dataset['categories'].append({"id": cid, "name": name, "supercategory": "object"})

    ann_id = 0

    for image_id in pred_image_ids:
        dataset['images'].append({
            "id": image_id,
            "file_name": str(image_id),
            "width": 1360, "height": 765
        })

        stem = os.path.splitext(str(image_id))[0]
        txt_path = os.path.join(txt_dir, stem + ".txt")

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6: continue
                    try:
                        # VisDrone: x,y,w,h,score,cls...
                        cls_raw = int(parts[5])

                        # 过滤掉 0 和 11
                        if cls_raw < 1 or cls_raw > 10: continue

                        dataset['annotations'].append({
                            "id": ann_id,
                            "image_id": image_id,
                            # 【关键修改】不再减 1，直接用原始 ID
                            "category_id": cls_raw,
                            "bbox": [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])],
                            "area": float(parts[2]) * float(parts[3]),
                            "iscrowd": 0
                        })
                        ann_id += 1
                    except ValueError:
                        continue

    print(f"✅ 真值转换完成！")
    with open(save_path, 'w') as f:
        json.dump(dataset, f)

    return save_path


def evaluate_coco(pred_json, anno_json):
    print(f"\n🚀 开始评估 COCO 指标...")
    try:
        anno = COCO(str(anno_json))
        pred = anno.loadRes(str(pred_json))
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return

    eval_bbox = COCOeval(anno, pred, 'bbox')
    eval_bbox.evaluate()
    eval_bbox.accumulate()
    eval_bbox.summarize()

    stats = eval_bbox.stats
    print("\n" + "=" * 40)
    print(f"🔥 【最终结果】")
    print(f"🏆 mAP (全尺寸): {stats[0] * 100:.2f}%")
    print(f"🦐 AP_small (小目标精度): {stats[3] * 100:.2f}%")
    print("=" * 40)


if __name__ == '__main__':
    gt_json = generate_robust_gt_json(GT_TXT_DIR, PRED_JSON_PATH)
    if gt_json:
        evaluate_coco(PRED_JSON_PATH, gt_json)