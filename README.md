# 🧠 Alzheimer's Disease Detection System (SADS v3.0)

A comprehensive deep learning system for early detection of Alzheimer's Disease using ensemble learning and multi-source data integration.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.12+-red.svg)](https://streamlit.io/)

## ✨ Features

- **Multi-Source Data Integration**
  - ALZ_Variant genetic variant data
  - MRI imaging data
  - Automatic data preprocessing

- **Ensemble Learning Models**
  - LSTM (Long Short-Term Memory)
  - CNN (Convolutional Neural Network)
  - Attention Mechanism
  - Hybrid Model

- **Interactive Web Interface**
  - Real-time progress tracking
  - Interactive visualizations
  - Clinical risk assessment

- **Comprehensive Evaluation**
  - ROC-AUC curves
  - Confusion matrices
  - Multiple performance metrics

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/alzheimers-detection-system.git
cd alzheimers-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

**Windows:**
```bash
启动Web应用.bat
```

**Linux/Mac:**
```bash
streamlit run alzheimers_detection_web_app.py
```

The application will automatically open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
alzheimers-detection-system/
├── alzheimers_detection_web_app.py  # Main application
├── 启动Web应用.bat                   # Windows launcher
├── requirements.txt                  # Dependencies
├── README.md                         # This file
├── README_CN.md                      # Chinese documentation
└── .gitignore                        # Git ignore file
```

## 🎯 Usage

1. **Configure Dataset Path** - Set in the sidebar
2. **Select Data Sources** - Choose ALZ_Variant and/or MRI data
3. **Choose Model Type** - Single model or 4-model ensemble
4. **Adjust Parameters** - Epochs and batch size
5. **Click "开始分析"** - Start the analysis

## 📊 Dataset Requirements

- **ALZ_Variant**: `preprocessed_alz_data.npz` in `Datasets/ALZ_Variant/`
- **MRI**: `train.parquet` and `test.parquet` in `Datasets/MRI/`

## ⚠️ Important Notes

- **Medical Disclaimer**: For research purposes only. Cannot replace professional medical diagnosis.
- **Data Privacy**: Handle medical data with care.
- **First Run**: May take longer due to model initialization.

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: AI4Alzheimer's Hackathon
- Framework: TensorFlow, Streamlit

---

**Version**: 3.0 | **Last Updated**: November 2025
