# 🧠 阿尔茨海默病检测系统 (SADS v3.0)

基于深度学习的阿尔茨海默病早期检测系统，整合多种数据源和先进的机器学习模型。

## ✨ 主要功能

- **多数据源整合**
  - ALZ_Variant 遗传变异数据
  - MRI 影像数据
  - 自动数据预处理

- **集成学习模型**
  - LSTM（长短期记忆网络）
  - CNN（卷积神经网络）
  - Attention（注意力机制）
  - Hybrid（混合模型）

- **交互式Web界面**
  - 实时进度显示
  - 交互式可视化
  - 临床风险评估

## 🚀 快速开始

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/alzheimers-detection-system.git
cd alzheimers-detection-system
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 运行应用

**Windows:**
```bash
启动Web应用.bat
```

**Linux/Mac:**
```bash
streamlit run alzheimers_detection_web_app.py
```

应用会自动在浏览器中打开：`http://localhost:8501`

## 📁 项目结构

```
alzheimers-detection-system/
├── alzheimers_detection_web_app.py  # 主程序
├── 启动Web应用.bat                   # Windows启动脚本
├── requirements.txt                  # 依赖列表
├── README.md                         # 英文说明
├── README_CN.md                      # 中文说明（本文件）
└── .gitignore                        # Git忽略文件
```

## 🎯 使用方法

1. **配置数据集路径** - 在侧边栏设置
2. **选择数据源** - 选择ALZ_Variant和/或MRI数据
3. **选择模型类型** - 单模型或4模型集成
4. **调整参数** - 训练轮数和批次大小
5. **点击"开始分析"** - 开始分析

## 📊 数据集要求

- **ALZ_Variant**: `preprocessed_alz_data.npz` 位于 `Datasets/ALZ_Variant/`
- **MRI**: `train.parquet` 和 `test.parquet` 位于 `Datasets/MRI/`

## ⚠️ 重要提示

- **医疗免责声明**: 仅用于研究目的。不能替代专业医疗诊断。
- **数据隐私**: 请妥善保管医学数据。
- **首次运行**: 可能需要较长时间进行模型初始化。

## 📝 许可证

MIT许可证 - 详见LICENSE文件。

## 🙏 致谢

- 数据集：AI4Alzheimer's Hackathon
- 框架：TensorFlow, Streamlit

---

**版本**: 3.0 | **最后更新**: 2025年11月
