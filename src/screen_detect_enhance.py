# ========== 全局离线配置（阻断网络请求） ==========
import os
import socket
import ssl

socket.setdefaulttimeout(5)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["ULTRALYTICS_AUTO_UPDATE"] = "0"
os.environ["TORCH_OFFLINE"] = "1"
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:12'

# ========== 导入依赖库 ==========
import numpy as np
import cv2
import pyautogui
from ultralytics import YOLO

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
    # 3. 添加细小沙尘噪点（优化，降低卡顿）
    noise = np.random.normal(0, 6, img.shape).astype(np.int16)
    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    # 4. 模拟正午局部强光曝光
    center_x, center_y = w // 2, h // 3
    radius = min(w, h) // 4
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    highlight = np.exp(-dist / (radius / 1.5)) * 50
    highlight = cv2.merge([highlight, highlight, highlight]).astype(np.uint8)
    img = cv2.add(img, highlight)
    return img

def enhance_west_image(img):
    """
    功能：对西部实拍图像 做增强修复（去黄、去噪、去雾、修复强光）
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
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255, 0, 255)
    img = cv2.LUT(img, look_up_table)
    return img
# ===================== 西部图像处理函数 结束 =====================

# ========== 加载模型（路径已修正） ==========
model_path = r"best.pt"
model = YOLO(model_path, task="detect")

# 缺陷类别
CLASS_NAMES = {
    0: "Crack",
    1: "Grid",
    2: "Spot"
}

# ========== 西部场景专属检测参数（已优化） ==========
CONF_THRESH = 0.35       # 提高置信度，过滤沙尘误检
MIN_BOX_AREA = 300       # 放大最小框，过滤细小沙尘颗粒
MAX_BOX_AREA = 30000
# 统一缩放尺寸，解决OpenCV绘制报错+卡顿
RESIZE_W = 960
RESIZE_H = 720

# ========== 运行模式选择 ==========
# "simulate"  : 模拟西部沙尘+强光环境（演示首选）
# "enhance"  : 对西部图像做增强修复（实测使用）
# "original" : 原始画面，不做图像处理
RUN_MODE = "simulate"

# ========== 屏幕检测主函数 ==========
def screen_detect():
    print("开始实时检测屏幕，按 q 键退出！")
    while True:
        # 截取屏幕
        img_pil = pyautogui.screenshot()
        # 转numpy + 颜色转换
        frame = np.array(img_pil)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 关键：缩放到固定大小，解决绘制报错+卡顿
        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

        # 根据模式执行对应图像处理
        if RUN_MODE == "simulate":
            frame = adjust_west_environment(frame)
        elif RUN_MODE == "enhance":
            frame = enhance_west_image(frame)
        # original 模式：不做任何处理

        # 模型推理
        results = model.predict(
            frame,
            conf=CONF_THRESH,
            verbose=False,
            half=False
        )[0]

        # 遍历检测框
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # 过滤异常大小框
            if not (MIN_BOX_AREA < area < MAX_BOX_AREA):
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls_id, f"未知({cls_id})")

            # 【修复】补全矩形两个坐标点，解决cv2.rectangle报错
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制文字
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                
            )

        # 显示画面
        cv2.imshow("西部光伏屏幕检测", frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    screen_detect()