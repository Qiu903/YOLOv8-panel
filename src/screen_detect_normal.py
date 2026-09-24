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

# ========== 加载模型（路径已修正） ==========
model_path = r"runs\detect\train1\weights\best.pt"
model = YOLO(model_path, task="detect")

# 缺陷类别
CLASS_NAMES = {
    0: "Crack",
    1: "Grid",
    2: "Spot"
}

# 检测参数
CONF_THRESH = 0.303
MIN_BOX_AREA = 200
MAX_BOX_AREA = 30000
# 统一缩放尺寸，解决OpenCV绘制报错
RESIZE_W = 960
RESIZE_H = 720

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

            # 绘制框和文字
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 显示画面
        cv2.imshow("光伏屏幕检测", frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    screen_detect()