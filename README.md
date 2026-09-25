# YOLOv8 目标检测项目
> 项目简介：基于YOLOv8实现目标检测，支持图片、屏幕画面、视频检测，包含模型训练与评估代码。本项目用于考研复试展示。

## 项目目录结构
'''text
panel-2/
├── src/ # 源代码文件夹
│ ├── train.py # 模型训练脚本
│ ├── model_eval.py # 模型评估脚本
│ ├── screen_detect_normal.py # 屏幕检测普通版
│ ├── screen_detect_enhance.py # 屏幕检测增强版
│ ├── video_detect_normal.py # 视频检测普通版
│ └── video_detect_enhance.py # 食品检测增强版
├── dataset/ # 数据集文件夹
│ └── data.yaml # 数据集配置文件（不含图片，图片体积过大不上传）
├── best.pt # 训练完成的最优权重best
├── .gitignore # Git 忽略文件配置
└── README.md # 项目说明文档
'''