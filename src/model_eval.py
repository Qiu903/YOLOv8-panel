# 解决 OpenMP 库冲突警告
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ✅ best.pt复制到项目根目录，使用根目录相对路径（runs不上传github）
model_path = r"best.pt"
model = YOLO(model_path)

if __name__ == '__main__':
    # workers=0 关闭多进程，适配Windows
    print("===== 验证集评估开始 =====")
    # ✅ 增加 data 参数，写data.yaml相对路径
    val_metrics = model.val(data="dataset/data.yaml", workers=0)

    # 读取官方标准指标
    mp = val_metrics.box.mp    # 平均精确率
    mr = val_metrics.box.mr    # 平均召回率
    map50 = val_metrics.box.map50
    map_all = val_metrics.box.map

    # 整体输出
    print(f"平均精确率(mp): {mp:.4f}")
    print(f"平均召回率(mr): {mr:.4f}")
    print(f"mAP50: {map50:.4f}")
    print(f"mAP50‑95: {map_all:.4f}")

    print("\n评估图表保存路径：")
    print("runs/detect/val")