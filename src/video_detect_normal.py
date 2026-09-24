import os
import socket
import ssl

socket.setdefaulttimeout(10)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["TORCH_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

# ========= 导入库 =========
from ultralytics import YOLO
import cv2

# ========= 加载你训练好的best.pt模型 =========
# 直接用相对路径，和你的项目结构一致
model = YOLO(r"runs\detect\train1\weights\best.pt")

# ========= 检测参数 =========
CONF_THRESH = 0.303  # 你模型的最优置信度
MIN_BOX_AREA = 200   # 过滤过小误检框
MAX_BOX_AREA = 30000 # 过滤过大误检框
CLASS_NAMES = {0: "Crack", 1: "Grid", 2: "Spot"}

# ========= 打开摄像头 =========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# ========= 主循环 =========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 用YOLO自带的predict接口推理（自动处理数据类型，不会报错）
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        verbose=False,
        half=False  # 强制关闭半精度，彻底解决类型冲突
    )[0]

    # 遍历检测框 + 过滤无效框
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if MIN_BOX_AREA < area < MAX_BOX_AREA:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 画框 + 文字
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("光伏板缺陷实时检测", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()