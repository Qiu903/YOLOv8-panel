# ========= 必须放在最前面，阻断网络请求 =========
import os
import socket
import ssl
import numpy as np

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

# ===================== 西部图像处理函数 开始 =====================
def adjust_west_environment(img):
    """
    功能：将普通图像 模拟为 西部沙尘、强光、泛黄环境图像
    适用：演示效果、模拟西部拍摄场景
    """
    h, w = img.shape[:2]
    # 1. 模拟沙尘整体泛黄
    yellow_mask = np.full_like(img, (20, 25, 35), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.85, yellow_mask, 0.15, 0)
    # 2. 模拟空中薄雾/浮尘
    fog_mask = np.full_like(img, 180, dtype=np.uint8)
    img = cv2.addWeighted(img, 0.82, fog_mask, 0.18, 0)
    # 3. 添加细小沙尘噪点
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 4. 模拟正午局部强光曝光
    center_x, center_y = w // 2, h // 3
    radius = min(w, h) // 4
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    highlight = np.exp(-dist / (radius / 1.5)) * 60
    highlight = cv2.merge([highlight, highlight, highlight]).astype(np.uint8)
    img = cv2.add(img, highlight)
    return img

def enhance_west_image(img):
    """
    功能：对西部实拍沙尘/强光图像 做增强修复（去黄、去噪、修复过曝、去雾）
    适用：西部现场采集图像，提升检测准确率
    """
    # 高斯滤波去除细小沙尘噪点
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # LAB空间白平衡，修正沙尘黄色偏色
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    avg_a = np.mean(a_channel)
    avg_b = np.mean(b_channel)
    a_channel = a_channel - ((avg_a - 128) * 1.1)
    b_channel = b_channel - ((avg_b - 128) * 1.1)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # YCrCb空间CLAHE自适应均衡，修复强光、提升暗部细节
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    img_ycrcb = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)
    # 伽马校正，减轻薄雾朦胧感
    gamma = 1.2
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    img = cv2.LUT(img, look_up_table)
    return img
# ===================== 西部图像处理函数 结束 =====================

# ========= 加载你训练好的best.pt模型 =========
model = YOLO(r"runs\detect\train1\weights\best.pt")

# ========= 检测参数（针对西部场景优化） =========
# 西部沙尘多，小幅提高置信度，减少噪点误检
CONF_THRESH = 0.35
# 收紧最小框面积，过滤沙尘微小颗粒
MIN_BOX_AREA = 300
MAX_BOX_AREA = 30000
CLASS_NAMES = {0: "Crack", 1: "Grid", 2: "Spot"}

# ========= 运行模式选择（二选一） =========
# 1. "simulate"  : 模拟西部沙尘+强光环境（演示推荐）
# 2. "enhance"  : 对西部实拍图像做增强修复（实际现场检测）
# 3. "original" : 不做图像处理，使用原始画面
RUN_MODE = "simulate"

# ========= 打开摄像头 =========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# ========= 主循环 =========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 根据模式执行对应图像处理
    if RUN_MODE == "simulate":
        frame = adjust_west_environment(frame)
    elif RUN_MODE == "enhance":
        frame = enhance_west_image(frame)
    # original 模式：不做任何处理，直接原图检测

    # 用YOLO自带的predict接口推理
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

    cv2.imshow("西部光伏板缺陷实时检测", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

