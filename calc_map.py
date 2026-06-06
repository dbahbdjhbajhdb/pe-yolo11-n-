import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =======================================================
# 1. 这里填你手头已有的 COCO 格式验证集标签路径 (绝对路径)
#    例如: r'C:\data\VisDrone\annotations\visdrone_val.json'
GT_JSON_PATH = r'C:\datasets\visdrone2019-coco\VisDrone2019-DET-val\annotations'

# 2. 这里填 YOLO 刚刚生成的预测文件路径
#    例如: r'C:\ultralytics-main\runs\detect\val3\predictions.json'
PRED_JSON_PATH = r'C:\ultralytics-main\runs\detect\val3\predictions.json'


# =======================================================

def main():
    if not os.path.exists(GT_JSON_PATH):
        print(f"错误: 找不到真值文件 -> {GT_JSON_PATH}")
        return
    if not os.path.exists(PRED_JSON_PATH):
        print(f"错误: 找不到预测文件 -> {PRED_JSON_PATH}")
        return

    print("正在加载真值 (Ground Truth)...")
    cocoGt = COCO(GT_JSON_PATH)

    print("正在加载预测值 (Predictions)...")
    # 加载预测结果
    try:
        cocoDt = cocoGt.loadRes(PRED_JSON_PATH)
    except Exception as e:
        print("\n!!! 加载预测文件失败 !!!")
        print("常见原因: image_id 不匹配。")
        print("Ultralytics 输出的 predictions.json 通常使用图片文件名(不含后缀)作为 image_id。")
        print("请检查你的真值 JSON 里的 image_id 是数字还是文件名字符串。")
        print(f"详细错误: {e}")
        return

    print("开始评测 (Evaluating)...")
    # 创建评测对象
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    # 这里的 imgIds 列表用于指定只评测哪些图片，默认是所有
    # cocoEval.params.imgIds = sorted(cocoGt.getImgIds())

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()