"""
阿尔茨海默病检测系统 - Web前端应用
基于Streamlit构建的交互式Web界面
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# 设置页面配置
st.set_page_config(
    page_title="阿尔茨海默病检测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🧠 阿尔茨海默病检测系统 (SADS v3.0)</h1>', unsafe_allow_html=True)
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 数据集路径配置
    st.subheader("📁 数据集路径")
    default_path = r"C:\Users\Administrator\Downloads\Datasets-20251115T200020Z-1-001\Datasets"
    dataset_path = st.text_input("数据集根目录", value=default_path)
    
    # 数据源选择
    st.subheader("📊 数据源选择")
    use_alz_variant = st.checkbox("使用 ALZ_Variant 数据", value=True)
    use_mri = st.checkbox("使用 MRI 数据", value=True)
    combine_datasets = st.checkbox("整合数据集", value=True)
    
    # 模型训练参数
    st.subheader("🎯 训练参数")
    use_ensemble = st.checkbox("使用4模型集成（推荐）", value=True)
    epochs = st.slider("训练轮数", 10, 50, 20)
    batch_size = st.slider("批次大小", 8, 32, 16)
    
    # 运行按钮
    st.markdown("---")
    run_analysis = st.button("🚀 开始分析", type="primary", use_container_width=True)
    
    # 关于信息
    st.markdown("---")
    st.markdown("### 📖 关于")
    st.info("""
    本系统整合了：
    - ALZ_Variant 遗传变异数据
    - MRI 影像数据
    - 4模型集成学习（可选）
    """)

# 主内容区域
if run_analysis:
    # 显示进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 导入主程序模块
        status_text.text("📦 正在加载模块...")
        progress_bar.progress(10)
        
        # 设置路径
        BASE_DATASET_PATH = dataset_path
        ALZ_VARIANT_PATH = os.path.join(BASE_DATASET_PATH, "ALZ_Variant")
        MRI_PATH = os.path.join(BASE_DATASET_PATH, "MRI")
        
        # 数据加载部分
        status_text.text("📂 正在加载数据集...")
        progress_bar.progress(20)
        
        X_train_final = None
        X_test_final = None
        y_train_final = None
        y_test_final = None
        data_source_info = []
        
        # 加载ALZ_Variant数据
        if use_alz_variant:
            alz_npz_path = os.path.join(ALZ_VARIANT_PATH, "preprocessed_alz_data.npz")
            if os.path.exists(alz_npz_path):
                import numpy as np
                alz_data = np.load(alz_npz_path)
                
                X_train_alz = alz_data['X_train']
                X_test_alz = alz_data['X_test']
                y_train_alz = alz_data['y_train']
                y_test_alz = alz_data['y_test']
                
                # 转换为二分类
                if len(y_train_alz.shape) > 1:
                    y_train_alz_binary = (np.argmax(y_train_alz, axis=1) >= 7).astype(int)
                    y_test_alz_binary = (np.argmax(y_test_alz, axis=1) >= 7).astype(int)
                else:
                    y_train_alz_binary = (y_train_alz > 0.5).astype(int)
                    y_test_alz_binary = (y_test_alz > 0.5).astype(int)
                
                X_train_alz_seq = np.stack([X_train_alz, X_train_alz * 0.95], axis=1)
                X_test_alz_seq = np.stack([X_test_alz, X_test_alz * 0.95], axis=1)
                
                X_train_final = X_train_alz_seq
                X_test_final = X_test_alz_seq
                y_train_final = y_train_alz_binary
                y_test_final = y_test_alz_binary
                data_source_info.append("ALZ_Variant")
        
        # 加载MRI数据
        if use_mri and os.path.exists(os.path.join(MRI_PATH, "train.parquet")):
            mri_train = pd.read_parquet(os.path.join(MRI_PATH, "train.parquet"))
            mri_test = pd.read_parquet(os.path.join(MRI_PATH, "test.parquet"))
            
            # 处理MRI数据
            target_col = mri_train.columns[-1]
            feature_cols_mri = [col for col in mri_train.columns if col != target_col]
            
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_train_mri = imputer.fit_transform(mri_train[feature_cols_mri].values)
            X_test_mri = imputer.transform(mri_test[feature_cols_mri].values)
            
            y_train_mri = mri_train[target_col].values
            y_test_mri = mri_test[target_col].values
            
            if y_train_mri.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train_mri = le.fit_transform(y_train_mri)
                y_test_mri = le.transform(y_test_mri)
            
            if len(np.unique(y_train_mri)) > 2:
                y_train_mri = (y_train_mri == np.max(y_train_mri)).astype(int)
                y_test_mri = (y_test_mri == np.max(y_test_mri)).astype(int)
            
            X_train_mri_seq = np.stack([X_train_mri, X_train_mri * 0.95], axis=1)
            X_test_mri_seq = np.stack([X_test_mri, X_test_mri * 0.95], axis=1)
            
            if combine_datasets and X_train_final is not None:
                min_features = min(X_train_final.shape[2], X_train_mri_seq.shape[2])
                X_train_final = np.concatenate([
                    X_train_final[:, :, :min_features],
                    X_train_mri_seq[:, :, :min_features]
                ], axis=0)
                X_test_final = np.concatenate([
                    X_test_final[:, :, :min_features],
                    X_test_mri_seq[:, :, :min_features]
                ], axis=0)
                y_train_final = np.concatenate([y_train_final, y_train_mri])
                y_test_final = np.concatenate([y_test_final, y_test_mri])
                data_source_info.append("MRI")
            elif X_train_final is None:
                X_train_final = X_train_mri_seq
                X_test_final = X_test_mri_seq
                y_train_final = y_train_mri
                y_test_final = y_test_mri
                data_source_info.append("MRI")
        
        if X_train_final is None:
            st.error("❌ 无法加载数据！请检查数据集路径。")
            st.stop()
        
        status_text.text("🔧 正在预处理数据...")
        progress_bar.progress(40)
        
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_2d = X_train_final.reshape(-1, X_train_final.shape[-1])
        X_test_2d = X_test_final.reshape(-1, X_test_final.shape[-1])
        scaler.fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train_final.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test_final.shape)
        
        # 显示数据信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("训练样本数", f"{X_train_final.shape[0]:,}")
        with col2:
            st.metric("测试样本数", f"{X_test_final.shape[0]:,}")
        with col3:
            st.metric("特征维度", X_train_final.shape[2])
        with col4:
            st.metric("数据源", ", ".join(data_source_info))
        
        status_text.text("🏗️ 正在构建模型...")
        progress_bar.progress(50)
        
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        n_biomarkers = X_train_scaled.shape[2]
        
        if use_ensemble:
            # 构建4模型集成
            def build_lstm_model():
                return keras.Sequential([
                    layers.LSTM(32, activation='relu', return_sequences=True, 
                               input_shape=(2, n_biomarkers)),
                    layers.Dropout(0.3),
                    layers.LSTM(16, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
            
            def build_cnn_model():
                return keras.Sequential([
                    layers.Conv1D(32, kernel_size=1, activation='relu', 
                                 input_shape=(2, n_biomarkers)),
                    layers.MaxPooling1D(pool_size=1),
                    layers.Conv1D(16, kernel_size=1, activation='relu'),
                    layers.Flatten(),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
            
            def build_attention_model():
                inputs = keras.Input(shape=(2, n_biomarkers))
                attention = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inputs, inputs)
                attention = layers.Flatten()(attention)
                x = layers.Dense(32, activation='relu')(attention)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(16, activation='relu')(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                return keras.Model(inputs=inputs, outputs=outputs)
            
            def build_hybrid_model():
                inputs = keras.Input(shape=(2, n_biomarkers))
                lstm = layers.LSTM(24, activation='relu', return_sequences=False)(inputs)
                cnn = layers.Conv1D(24, kernel_size=1, activation='relu')(inputs)
                cnn = layers.Flatten()(cnn)
                merged = layers.Concatenate()([lstm, cnn])
                x = layers.Dense(32, activation='relu')(merged)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(16, activation='relu')(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                return keras.Model(inputs=inputs, outputs=outputs)
            
            models = {
                'LSTM': build_lstm_model(),
                'CNN': build_cnn_model(),
                'Attention': build_attention_model(),
                'Hybrid': build_hybrid_model()
            }
            
            for name, model in models.items():
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            status_text.text("🎓 正在训练4模型集成...")
            progress_bar.progress(60)
            
            # 训练所有模型
            model_histories = {}
            training_placeholder = st.empty()
            
            for model_name, model in models.items():
                with training_placeholder.container():
                    st.info(f"正在训练 {model_name} 模型...")
                history = model.fit(
                    X_train_scaled, y_train_final,
                    validation_split=0.2,
                    epochs=min(epochs, 20),
                    batch_size=batch_size,
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                    )],
                    verbose=0
                )
                model_histories[model_name] = history
            
            training_placeholder.empty()
            
            status_text.text("📊 正在评估集成模型...")
            progress_bar.progress(80)
            
            # 集成预测
            ensemble_preds = []
            for model in models.values():
                pred = model.predict(X_test_scaled, verbose=0).flatten()
                ensemble_preds.append(pred)
            
            y_pred_proba = np.mean(ensemble_preds, axis=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            history = model_histories['LSTM']
            
        else:
            # 单模型版本
            model = keras.Sequential([
                layers.LSTM(32, activation='relu', return_sequences=True, 
                           input_shape=(2, n_biomarkers)),
                layers.Dropout(0.3),
                layers.LSTM(16, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            status_text.text("🎓 正在训练模型...")
            progress_bar.progress(60)
            
            with st.spinner("训练中，请稍候..."):
                history = model.fit(
                    X_train_scaled, y_train_final,
                    validation_split=0.2,
                    epochs=min(epochs, 20),
                    batch_size=batch_size,
                    verbose=0
                )
            
            status_text.text("📊 正在评估模型...")
            progress_bar.progress(80)
            
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            models = {'Single Model': model}
            ensemble_preds = [y_pred_proba]
        
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
        
        auc = roc_auc_score(y_test_final, y_pred_proba)
        accuracy = accuracy_score(y_test_final, y_pred)
        f1 = f1_score(y_test_final, y_pred)
        cm = confusion_matrix(y_test_final, y_pred)
        
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        progress_bar.progress(100)
        status_text.text("✅ 分析完成！")
        
        # 显示结果
        st.markdown("## 📈 性能指标")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("AUC-ROC", f"{auc:.4f}", delta=None)
        with col2:
            st.metric("准确率", f"{accuracy:.4f}", delta=None)
        with col3:
            st.metric("F1分数", f"{f1:.4f}", delta=None)
        with col4:
            st.metric("敏感性", f"{sensitivity:.4f}", delta=None)
        with col5:
            st.metric("特异性", f"{specificity:.4f}", delta=None)
        
        # 显示单个模型性能（如果是集成）
        if use_ensemble and len(ensemble_preds) > 1:
            st.markdown("## 🔍 单个模型性能")
            model_aucs = {}
            for model_name, pred in zip(models.keys(), ensemble_preds):
                model_auc = roc_auc_score(y_test_final, pred)
                model_aucs[model_name] = model_auc
            
            model_df = pd.DataFrame({
                '模型': list(model_aucs.keys()),
                'AUC-ROC': list(model_aucs.values())
            })
            st.dataframe(model_df, use_container_width=True)
        
        # 可视化
        st.markdown("## 📊 可视化结果")
        
        if use_ensemble and len(ensemble_preds) > 1:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROC曲线
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={auc:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 0].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 0].set_xlabel('假阳性率')
        axes[0, 0].set_ylabel('真阳性率')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False, square=True)
        axes[0, 1].set_title('混淆矩阵')
        axes[0, 1].set_ylabel('真实值')
        axes[0, 1].set_xlabel('预测值')
        
        # 预测分布
        axes[1, 0].hist(y_pred_proba[y_test_final==0], bins=15, alpha=0.6, label='正常', color='green')
        axes[1, 0].hist(y_pred_proba[y_test_final==1], bins=15, alpha=0.6, label='阿尔茨海默病', color='red')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_title('预测分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 性能指标柱状图
        if use_ensemble and len(ensemble_preds) > 1:
            metrics = ['AUC', '准确率', 'F1', '敏感性', '特异性']
            values = [auc, accuracy, f1, sensitivity, specificity]
            colors = ['#4ECDC4' if v > 0.8 else '#FF6B6B' for v in values]
            axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('分数')
            axes[1, 1].set_title('整体性能')
            axes[1, 1].set_ylim([0.5, 1.0])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # 模型比较
            model_aucs = [roc_auc_score(y_test_final, pred) for pred in ensemble_preds]
            colors_models = ['#4ECDC4' if auc_val == max(model_aucs) else '#FF6B6B' for auc_val in model_aucs]
            axes[1, 2].bar(models.keys(), model_aucs, color=colors_models, alpha=0.7, edgecolor='black')
            axes[1, 2].set_ylabel('AUC-ROC')
            axes[1, 2].set_title('单个模型性能')
            axes[1, 2].set_ylim([0.7, 1.0])
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        else:
            metrics = ['AUC', '准确率', 'F1', '敏感性', '特异性']
            values = [auc, accuracy, f1, sensitivity, specificity]
            colors = ['#4ECDC4' if v > 0.8 else '#FF6B6B' for v in values]
            axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('分数')
            axes[1, 1].set_title('整体性能')
            axes[1, 1].set_ylim([0.5, 1.0])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 训练历史
        st.markdown("## 📉 训练历史")
        history_df = pd.DataFrame(history.history)
        st.line_chart(history_df[['loss', 'val_loss']])
        
        # 临床预测示例
        st.markdown("## 🏥 临床风险评估示例")
        for i in range(min(5, len(X_test_scaled))):
            prob = y_pred_proba[i]
            if prob > 0.75:
                risk = "🔴 极高风险"
            elif prob > 0.6:
                risk = "🟠 高风险"
            elif prob > 0.4:
                risk = "🟡 中等风险"
            else:
                risk = "🟢 低风险"
            
            st.markdown(f"**患者 {i+1}**: 风险概率 = {prob:.1%} | {risk}")
        
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        st.exception(e)

else:
    # 欢迎页面
    st.markdown("""
    ## 👋 欢迎使用阿尔茨海默病检测系统
    
    这是一个基于深度学习的阿尔茨海默病早期检测系统，整合了多种数据源和先进的机器学习模型。
    
    ### ✨ 主要功能
    
    1. **多数据源整合**
       - ALZ_Variant 遗传变异数据
       - MRI 影像数据
       - 自动数据预处理
    
    2. **集成学习模型**
       - LSTM（长短期记忆网络）
       - CNN（卷积神经网络）
       - Attention（注意力机制）
       - Hybrid（混合模型）
    
    3. **全面性能评估**
       - AUC-ROC 曲线
       - 混淆矩阵
       - 多种评估指标
    
    4. **临床风险评估**
       - 患者风险概率预测
       - 可视化结果展示
    
    ### 🚀 快速开始
    
    1. 在左侧边栏配置数据集路径
    2. 选择要使用的数据源
    3. 调整训练参数（可选）
    4. 点击"开始分析"按钮
    
    ### 📊 数据集要求
    
    - **ALZ_Variant**: `preprocessed_alz_data.npz` 文件
    - **MRI**: `train.parquet` 和 `test.parquet` 文件
    
    ### ⚠️ 注意事项
    
    - 首次运行可能需要较长时间进行模型训练
    - 建议使用GPU加速训练过程
    - 结果仅供参考，不能替代专业医疗诊断
    """)
    
    # 显示数据集信息
    st.markdown("### 📁 数据集信息")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **ALZ_Variant 数据**
        - 格式: NPZ (NumPy压缩)
        - 训练集: 5076样本 × 130特征
        - 测试集: 1270样本 × 130特征
        - 标签: 9分类（已转换为二分类）
        """)
    
    with info_col2:
        st.markdown("""
        **MRI 数据**
        - 格式: Parquet (列式存储)
        - 包含训练集和测试集
        - 影像相关特征数据
        - 适合大数据分析
        """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "阿尔茨海默病检测系统 (SADS v3.0) | "
    "基于Streamlit构建 | "
    "© 2025"
    "</div>",
    unsafe_allow_html=True
)
