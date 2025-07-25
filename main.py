import numpy as np
import pandas as pd
from numba import njit, float64, int32, boolean, types
from numba.typed import List, Dict
from numba.experimental import jitclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss, 
                           log_loss, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# 导入现有的工具函数和类
from gbm_model_training_evaluation import (
    MW, estimate_gbm_parameters, 
    calculate_execution_probability_gbm, calculate_order_book_features, 
    calculate_trade_features
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM可用")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("✗ LightGBM不可用")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("✓ CatBoost可用")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("✗ CatBoost不可用")

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    GTC,
    MARKET,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    DEPTH_EVENT,
    Recorder
)
from hftbacktest.recorder import Recorder_

class ForwardLookingDataCollector:
    """前向查看数据收集器 - 两阶段处理避免时间回退问题"""
    
    def __init__(self, min_horizon_ms=0, max_horizon_ms=1000, levels=5):
        self.min_horizon_ns = min_horizon_ms * 1_000_000  
        self.max_horizon_ns = max_horizon_ms * 1_000_000
        self.levels = levels
        
    def collect_data_and_labels(self, asset_no: int):
        """一次性收集特征和标签数据"""
        
        min_horizon = self.min_horizon_ns
        max_horizon = self.max_horizon_ns
        levels = self.levels
        
        @njit
        def _collect_data(hbt: ROIVectorMarketDepthBacktest, rec: Recorder_,
                         min_h: int, max_h: int, lv: int):
            
            # 存储市场数据序列
            market_data = []  # (timestamp, best_bid, best_ask, mid_price)
            feature_samples = []  # 特征样本
            sample_metadata = []  # 样本元数据用于标签生成
            
            # 市场状态窗口
            price_window = MW(200)
            volume_window = MW(100)
            
            data_points = 0
            sample_interval = 50_000_000  # 50ms采样间隔
            
            print("第一阶段：收集市场数据和特征...")
            
            while hbt.elapse(sample_interval) == 0:
                current_time = hbt.current_timestamp
                depth = hbt.depth(asset_no)
                trades = hbt.last_trades(asset_no)
                
                if depth.best_bid <= 0 or depth.best_ask <= 0:
                    hbt.clear_last_trades(asset_no)
                    rec.record(hbt)
                    continue
                
                # 记录市场数据
                mid_price = (depth.best_bid + depth.best_ask) / 2.0
                market_data.append((current_time, depth.best_bid, depth.best_ask, mid_price))
                
                # 计算特征
                ob_features = calculate_order_book_features(depth, lv)
                trade_features = calculate_trade_features(trades, current_time)
                
                price_window.push(mid_price)
                volume_window.push(trade_features[1])
                
                # 估计GBM参数
                price_history = price_window.get_buffer()
                mu, sigma = estimate_gbm_parameters(price_history, dt=0.05)
                
                # 市场状态特征
                volatility = price_window.std() / mid_price if mid_price > 0 else 0.0
                volume_rate = volume_window.mean()
                
                # 为每个档位和方向生成特征样本
                for side in [True, False]:  # True=bid, False=ask
                    for level in range(1, lv + 1):
                        if side:  # bid side
                            order_price = depth.best_bid - depth.tick_size * level
                        else:  # ask side  
                            order_price = depth.best_ask + depth.tick_size * level
                            
                        if order_price <= 0:
                            continue
                        
                        # 创建特征向量
                        feature_vector = np.zeros(25)
                        feature_vector[:10] = ob_features
                        feature_vector[10:18] = trade_features
                        feature_vector[18] = mu
                        feature_vector[19] = sigma
                        feature_vector[20] = volatility
                        feature_vector[21] = volume_rate
                        feature_vector[22] = float(level)
                        feature_vector[23] = 1.0 if side else 0.0
                        feature_vector[24] = order_price / mid_price
                        
                        feature_samples.append(feature_vector)
                        
                        # 存储元数据用于标签生成
                        metadata = np.zeros(6)
                        metadata[0] = current_time
                        metadata[1] = order_price  
                        metadata[2] = 1.0 if side else 0.0
                        metadata[3] = mid_price
                        metadata[4] = min_h
                        metadata[5] = max_h
                        sample_metadata.append(metadata)
                        
                        data_points += 1
                        if data_points % 5_000_000 == 0:
                            print(f"已收集 {data_points} 个特征样本")
                            
                hbt.clear_last_trades(asset_no)
                rec.record(hbt)
                
            print(f"收集完成: {len(market_data)} 个市场数据点, {len(feature_samples)} 个特征样本")
            return market_data, feature_samples, sample_metadata
        
        return lambda hbt, rec: _collect_data(hbt, rec, min_horizon, max_horizon, levels)

    def generate_labels_from_market_data(self, market_data, sample_metadata):
        """基于收集的市场数据生成forward looking标签"""
        print("第二阶段：生成forward looking标签...")
        
        # 转换为numpy数组以便快速查找
        market_data_array = np.array(market_data)
        timestamps = market_data_array[:, 0]
        bids = market_data_array[:, 1] 
        asks = market_data_array[:, 2]
        
        labels = []
        total_samples = len(sample_metadata)
        
        for i, meta in enumerate(sample_metadata):
            timestamp = meta[0]
            order_price = meta[1]
            is_bid = bool(meta[2])
            min_horizon = meta[4]
            max_horizon = meta[5]
            
            # 设置时间窗口
            start_time = timestamp + min_horizon
            end_time = timestamp + max_horizon
            
            # 找到时间窗口内的数据索引
            window_mask = (timestamps >= start_time) & (timestamps <= end_time)
            window_indices = np.where(window_mask)[0]
            
            label = 0.0
            
            if len(window_indices) > 0:
                if is_bid:
                    # 对于bid订单，检查ask价格是否触及我们的order_price
                    window_asks = asks[window_indices]
                    if np.any(window_asks <= order_price):
                        label = 1.0
                else:
                    # 对于ask订单，检查bid价格是否触及我们的order_price
                    window_bids = bids[window_indices]
                    if np.any(window_bids >= order_price):
                        label = 1.0
            
            labels.append(label)
            
            if (i + 1) % 100000 == 0:
                print(f"已处理 {i+1}/{total_samples} 个标签 ({(i+1)/total_samples*100:.1f}%)")
                
        return np.array(labels)

class ModelTrainer:
    """多模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.model_names = ['GBM_Baseline', 'Logistic', 'RandomForest']
        
        if LIGHTGBM_AVAILABLE:
            self.model_names.append('LightGBM')
        if CATBOOST_AVAILABLE:
            self.model_names.append('CatBoost')
    
    def create_models(self):
        """创建所有模型"""
        
        # GBM Baseline (使用理论GBM概率作为特征的逻辑回归)
        self.models['GBM_Baseline'] = LogisticRegression(
            random_state=42, max_iter=1000
        )
        
        # 逻辑回归
        self.models['Logistic'] = LogisticRegression(
            random_state=42, max_iter=1000, C=1.0
        )
        
        # 随机森林
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            self.models['CatBoost'] = cb.CatBoostClassifier(
                iterations=100,
                depth=10,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
    
    def train_all_models(self, X_train, y_train):
        """训练所有模型"""
        print("开始训练所有模型...")
        
        self.create_models()
        
        for name in self.model_names:
            print(f"训练 {name} 模型...")
            
            if name == 'GBM_Baseline':
                # 为GBM baseline添加理论概率特征
                X_train_gbm = self.add_gbm_features(X_train)
                self.models[name].fit(X_train_gbm, y_train)
            else:
                self.models[name].fit(X_train, y_train)
    
    def add_gbm_features(self, X):
        """为GBM baseline添加理论概率特征"""
        X_enhanced = np.column_stack([X, np.zeros(len(X))])
        
        for i in range(len(X)):
            # 提取相关特征
            mid_price = X[i, 0]
            mu = X[i, 18]
            sigma = X[i, 19]
            is_bid = bool(X[i, 23])
            order_price = X[i, 24] * mid_price
            
            # 计算理论GBM概率作为额外特征
            gbm_prob = calculate_execution_probability_gbm(
                order_price, mid_price, is_bid, mu, sigma, 0.5  # 500ms
            )
            X_enhanced[i, -1] = gbm_prob
            
        return X_enhanced
    
    def predict_all_models(self, X_test):
        """所有模型预测"""
        predictions = {}
        
        for name in self.model_names:
            if name == 'GBM_Baseline':
                X_test_gbm = self.add_gbm_features(X_test)
                predictions[name] = self.models[name].predict_proba(X_test_gbm)[:, 1]
            else:
                predictions[name] = self.models[name].predict_proba(X_test)[:, 1]
                
        return predictions

class ModelEvaluator:
    """全面的模型评估器"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_all_models(self, y_true, predictions):
        """评估所有模型"""
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)
        
        for model_name, y_pred in predictions.items():
            print(f"\n{model_name} 模型:")
            print("-" * 40)
            
            results = self.calculate_metrics(y_true, y_pred)
            self.results[model_name] = results
            
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            print(f"Brier Score: {results['brier_score']:.4f}")
            print(f"Log Loss: {results['log_loss']:.4f}")
            print(f"准确率: {results['accuracy']:.4f}")
            print(f"精确率: {results['precision']:.4f}")
            print(f"召回率: {results['recall']:.4f}")
            print(f"F1分数: {results['f1_score']:.4f}")
    
    def calculate_metrics(self, y_true, y_pred_proba):
        """计算全面的评估指标"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        results = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return results
    
    def plot_evaluation_results(self, y_true, predictions, save_path=None):
        """绘制评估结果图表"""
        n_models = len(predictions)
        
        # ROC曲线
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        for model_name, y_pred in predictions.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall曲线
        plt.subplot(2, 3, 2)
        for model_name, y_pred in predictions.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            plt.plot(recall, precision, label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 校准曲线
        plt.subplot(2, 3, 3)
        for model_name, y_pred in predictions.items():
            fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
            plt.plot(mean_pred, fraction_pos, marker='o', label=model_name)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('校准曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测概率分布
        plt.subplot(2, 3, 4)
        for model_name, y_pred in predictions.items():
            plt.hist(y_pred, bins=30, alpha=0.5, label=model_name, density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('预测概率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 指标对比
        plt.subplot(2, 3, 5)
        metrics = ['roc_auc', 'brier_score', 'log_loss', 'f1_score']
        model_names = list(predictions.keys())
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            plt.bar([f"{name}\n{metric}" for name in model_names], values, alpha=0.7)
        plt.xticks(rotation=45)
        plt.title('指标对比')
        plt.grid(True, alpha=0.3)
        
        # 混淆矩阵
        plt.subplot(2, 3, 6)
        # 选择最佳模型绘制混淆矩阵
        best_model = max(predictions.keys(), key=lambda x: self.results[x]['roc_auc'])
        y_pred_best = (predictions[best_model] > 0.5).astype(int)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'混淆矩阵 ({best_model})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()

def run_execution_probability_prediction(min_horizon_ms=100, max_horizon_ms=1000):
    """运行完整的执行概率预测流程"""
    
    print(f"开始执行概率预测 - 时间范围: {min_horizon_ms}ms - {max_horizon_ms}ms")
    
    # 1. 数据收集
    print("\n第一步：收集数据")
    
    asset = (
        BacktestAsset()
            .data([
                'DOGE-USDT-PERP_20250101_merged.npz',
                'DOGE-USDT-PERP_20250102_merged.npz'
            ])
            .initial_snapshot('DOGE-USDT-PERP_20250101(1)_eod_fixed.npz')
            .linear_asset(1.0)
            .intp_order_latency([
                'DOGE-USDT-PERP_20250101_merged_latency.npz',
                'DOGE-USDT-PERP_20250102_merged_latency.npz'
            ])
            .power_prob_queue_model(1.2)
            .no_partial_fill_exchange()
            .trading_value_fee_model(-0.00005, 0.0002)
            .tick_size(0.00001)
            .lot_size(1.0)
            .roi_lb(0.0)
            .roi_ub(300.0)
            .last_trades_capacity(10000)
    )

    hbt = ROIVectorMarketDepthBacktest([asset])
    recorder = Recorder(1, 5_000_000)

    # 收集特征数据
    collector = ForwardLookingDataCollector(
        min_horizon_ms=min_horizon_ms, 
        max_horizon_ms=max_horizon_ms, 
        levels=5
    )
    
    collect_task = collector.collect_data_and_labels(0)
    market_data, features_list, metadata_list = collect_task(hbt, recorder.recorder)
    
    print(f"收集到 {len(features_list)} 个样本")
    
    # 2. 生成forward looking标签
    print("\n第二步：生成标签")
    
    labels = collector.generate_labels_from_market_data(market_data, metadata_list)
    
    hbt.close()
    
    # 转换为numpy数组
    X = np.array(features_list)
    y = labels
    
    print(f"特征维度: {X.shape}")
    print(f"正样本比例: {np.mean(y):.3f}")
    
    # 3. 数据分割
    print("\n第三步：分割数据")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 4. 模型训练
    print("\n第四步：训练模型")
    
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    # 5. 模型预测
    print("\n第五步：模型预测")
    
    predictions = trainer.predict_all_models(X_test)
    
    # 6. 模型评估
    print("\n第六步：模型评估")
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(y_test, predictions)
    
    # 7. 可视化结果
    print("\n第七步：生成评估图表")
    
    evaluator.plot_evaluation_results(
        y_test, predictions, 
        save_path=f"execution_prediction_{min_horizon_ms}ms_{max_horizon_ms}ms"
    )
    
    # 8. 保存结果
    print("\n第八步：保存结果")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    with open(f"models_{min_horizon_ms}ms_{max_horizon_ms}ms_{timestamp}.pkl", 'wb') as f:
        pickle.dump(trainer.models, f)
    
    # 保存评估结果
    results_df = pd.DataFrame(evaluator.results).T
    results_df.to_csv(f"evaluation_results_{min_horizon_ms}ms_{max_horizon_ms}ms_{timestamp}.csv")
    
    print(f"结果已保存")
    print(f"最佳模型: {max(evaluator.results.keys(), key=lambda x: evaluator.results[x]['roc_auc'])}")
    
    return trainer, evaluator, predictions

if __name__ == "__main__":
    # 运行预测任务
    trainer, evaluator, predictions = run_execution_probability_prediction(
        min_horizon_ms=0,   # 100ms最小时间
        max_horizon_ms=1000   # 1000ms最大时间
    )
    
    print("\n任务完成！")
