import numpy as np
import pandas as pd
from numba import njit, float64, int32, boolean, types
from numba.typed import List, Dict
from numba.experimental import jitclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from datetime import datetime

# GPU加速库的可选导入
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as CumlRandomForestRegressor
    CUML_AVAILABLE = True
    print("✓ cuML可用 - 支持GPU加速")
except (ImportError, Exception) as e:
    CUML_AVAILABLE = False
    print("✗ cuML不可用 - 使用CPU版本")
    # print(f"  错误详情: {e}")  # 取消注释以查看详细错误

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✓ XGBoost可用")
except (ImportError, Exception) as e:
    XGB_AVAILABLE = False
    print("✗ XGBoost不可用")
    # print(f"  错误详情: {e}")  # 取消注释以查看详细错误

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
from hftbacktest.stats import LinearAssetRecord

# 复用现有的Moving Window类
spec = [
    ('buffer', float64[:]),
    ('index', int32),
    ('window_size', int32),
    ('_sum', float64),
]

@jitclass(spec=spec)
class MW:
    def __init__(self, window_size: int):
        self.buffer = np.zeros(window_size)
        self.index = 0
        self.window_size = window_size
        self._sum = 0.0

    def push(self, x: float):
        self._sum -= self.buffer[self.index]
        self._sum += x
        self.buffer[self.index] = x
        self.index = (self.index + 1) % self.window_size

    def mean(self):
        return self._sum / self.window_size

    def std(self):
        return np.std(self.buffer)

    def sum(self):
        return self._sum

    def get_buffer(self):
        return self.buffer

@njit
def estimate_gbm_parameters(price_history, dt=1.0):
    """估计GBM参数μ和σ"""
    if len(price_history) < 2:
        return 0.0, 0.01

    log_returns = np.zeros(len(price_history) - 1)
    for i in range(len(price_history) - 1):
        if price_history[i] > 0 and price_history[i+1] > 0:
            log_returns[i] = np.log(price_history[i+1] / price_history[i])

    if len(log_returns) == 0:
        return 0.0, 0.01

    mean_log_return = np.mean(log_returns)
    std_log_return = np.std(log_returns)

    volatility = std_log_return / np.sqrt(dt)
    drift = mean_log_return / dt + 0.5 * volatility * volatility

    return drift, volatility

@njit
def calculate_execution_probability_gbm(order_price, current_price, side, mu, sigma, time_horizon):
    """使用GBM模型计算订单成交概率"""
    if sigma <= 0 or current_price <= 0 or order_price <= 0:
        return 0.5

    price_ratio = order_price / current_price
    log_ratio = np.log(price_ratio)
    adjusted_drift = mu - 0.5 * sigma * sigma

    if sigma * np.sqrt(time_horizon) > 0:
        d1 = (log_ratio - adjusted_drift * time_horizon) / (sigma * np.sqrt(time_horizon))

        if side:  # bid订单
            if order_price >= current_price:
                prob = 1.0
            else:
                prob = 0.5 + 0.5 * np.tanh(-d1)
        else:  # ask订单
            if order_price <= current_price:
                prob = 1.0
            else:
                prob = 0.5 + 0.5 * np.tanh(d1)
    else:
        prob = 0.5

    return min(max(prob, 0.0), 1.0)

@njit
def calculate_order_book_features(depth, levels=5):
    """计算订单簿特征"""
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return np.zeros(10)

    features = np.zeros(10)
    best_bid = depth.best_bid
    best_ask = depth.best_ask
    mid_price = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    tick_size = depth.tick_size

    features[0] = mid_price
    features[1] = spread
    features[2] = spread / mid_price * 10000  # spread_bps

    # 计算多档位的量和不平衡性
    total_bid_qty = 0.0
    total_ask_qty = 0.0
    weighted_bid_price = 0.0
    weighted_ask_price = 0.0

    for i in range(min(levels, 10)):
        # 计算价格（通过tick偏移）
        bid_price = best_bid - tick_size * i
        ask_price = best_ask + tick_size * i

        # 获取对应tick的数量
        bid_qty = depth.bid_qty_at_tick(depth.best_bid_tick - i)
        ask_qty = depth.ask_qty_at_tick(depth.best_ask_tick + i)

        if bid_price > 0 and bid_qty > 0:
            total_bid_qty += bid_qty
            weighted_bid_price += bid_price * bid_qty
        if ask_price > 0 and ask_qty > 0:
            total_ask_qty += ask_qty
            weighted_ask_price += ask_price * ask_qty

    features[3] = total_bid_qty
    features[4] = total_ask_qty

    # 量不平衡
    if total_bid_qty + total_ask_qty > 0:
        features[5] = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)

    # 加权平均价格
    if total_bid_qty > 0:
        features[6] = weighted_bid_price / total_bid_qty
    if total_ask_qty > 0:
        features[7] = weighted_ask_price / total_ask_qty

    # 价格不平衡
    if features[6] > 0 and features[7] > 0:
        features[8] = (features[6] - features[7]) / mid_price

    # 最优档位量
    best_bid_qty = depth.bid_qty_at_tick(depth.best_bid_tick)
    best_ask_qty = depth.ask_qty_at_tick(depth.best_ask_tick)
    if best_bid_qty + best_ask_qty > 0:
        features[9] = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)

    return features

@njit
def calculate_trade_features(trades, current_time, time_window=5000000000):  # 5秒窗口
    """计算交易特征"""
    if len(trades) == 0:
        return np.zeros(8)

    features = np.zeros(8)
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    volume_weighted_price = 0.0
    recent_trades = 0

    for trade in trades:
        if current_time - trade.local_ts <= time_window:
            recent_trades += 1
            volume = trade.qty
            price = trade.px
            total_volume += volume
            volume_weighted_price += price * volume

            # 判断买卖方向（简化处理）
            if trade.ev & BUY_EVENT:
                buy_volume += volume
            else:
                sell_volume += volume

    features[0] = float(recent_trades)  # 交易次数
    features[1] = total_volume  # 总成交量
    features[2] = buy_volume  # 买方成交量
    features[3] = sell_volume  # 卖方成交量

    # 交易不平衡
    if total_volume > 0:
        features[4] = (buy_volume - sell_volume) / total_volume
        features[5] = volume_weighted_price / total_volume  # VWAP

    # 交易强度
    features[6] = total_volume / (time_window / 1000000000.0)  # 每秒成交量
    features[7] = recent_trades / (time_window / 1000000000.0)  # 每秒交易次数

    return features

class GBMExecutionDataCollector:
    """GBM执行概率数据收集器"""

    def __init__(self, time_horizons=[0.1, 0.2, 0.5, 1.0], levels=5):
        """
        初始化数据收集器

        Args:
            time_horizons: 预测时间范围（秒）- 默认100ms到1s
            levels: 订单簿档位数量
        """
        self.time_horizons = time_horizons
        self.levels = levels
        self.features = []
        self.labels_prob = {}  # 各时间范围的成交概率标签
        self.labels_time = []  # 成交时间标签

        for horizon in time_horizons:
            self.labels_prob[horizon] = []

    def collect_data(self, asset_no: int):
        """收集训练数据的numba函数 - 重构为动态时间范围"""

        # 将类属性转换为numba可处理的数组
        time_horizons_array = np.array(self.time_horizons, dtype=np.float64)
        levels = self.levels
        n_horizons = len(self.time_horizons)

        @njit
        def _collect_data(hbt: ROIVectorMarketDepthBacktest, rec: Recorder_,
                         horizons: np.ndarray, n_h: int, lv: int):
            # 存储特征和标签 - 使用列表避免动态数组问题
            features_list = []
            prob_labels_list = []  # 改为列表收集
            time_labels = []

            # 市场状态窗口
            price_window = MW(200)
            volume_window = MW(100)

            data_points = 0

            while hbt.elapse(100000000) == 0:  # 100ms间隔收集数据，提高短期预测精度
                current_time = hbt.current_timestamp
                depth = hbt.depth(asset_no)
                trades = hbt.last_trades(asset_no)

                if depth.best_bid <= 0 or depth.best_ask <= 0:
                    hbt.clear_last_trades(asset_no)
                    rec.record(hbt)
                    continue

                # 计算市场特征
                ob_features = calculate_order_book_features(depth, lv)
                trade_features = calculate_trade_features(trades, current_time)

                mid_price = ob_features[0]
                price_window.push(mid_price)

                total_volume = trade_features[1]
                volume_window.push(total_volume)

                # 估计GBM参数
                price_history = price_window.get_buffer()
                mu, sigma = estimate_gbm_parameters(price_history, dt=0.1)  # 100ms间隔

                # 市场状态特征
                volatility = price_window.std() / mid_price if mid_price > 0 else 0.0
                volume_rate = volume_window.mean()

                # 为每个档位创建特征向量和标签
                for side in [True, False]:  # True=bid, False=ask
                    for level in range(1, lv + 1):
                        if side:  # bid side
                            order_price = depth.best_bid - depth.tick_size * level
                        else:  # ask side
                            order_price = depth.best_ask + depth.tick_size * level

                        if order_price <= 0:
                            continue

                        # 特征向量组合
                        feature_vector = np.zeros(25)
                        feature_vector[:10] = ob_features  # 订单簿特征
                        feature_vector[10:18] = trade_features  # 交易特征
                        feature_vector[18] = mu  # GBM drift
                        feature_vector[19] = sigma  # GBM volatility
                        feature_vector[20] = volatility  # 历史波动率
                        feature_vector[21] = volume_rate  # 成交量率
                        feature_vector[22] = float(level)  # 档位
                        feature_vector[23] = 1.0 if side else 0.0  # 方向
                        feature_vector[24] = order_price / mid_price  # 相对价格

                        features_list.append(feature_vector)

                        # 动态计算所有时间范围的成交概率
                        prob_row = np.zeros(n_h)
                        for i in range(n_h):
                            horizon = horizons[i]
                            prob_row[i] = calculate_execution_probability_gbm(
                                order_price, mid_price, side, mu, sigma, horizon
                            )

                        # 添加到概率标签列表
                        prob_labels_list.append(prob_row)

                        # 模拟订单执行时间计算（简化版本）
                        exec_time = estimate_execution_time(
                            order_price, mid_price, side, mu, sigma
                        )
                        time_labels.append(exec_time)

                        data_points += 1
                        if data_points % 1000 == 0:
                            print(f"已收集 {data_points} 个数据点")

                hbt.clear_last_trades(asset_no)
                rec.record(hbt)

            return features_list, prob_labels_list, time_labels

        # 返回一个lambda，传递参数
        return lambda hbt, rec: _collect_data(hbt, rec, time_horizons_array, n_horizons, levels)

@njit
def estimate_execution_time(order_price, current_price, side, mu, sigma):
    """估计订单执行时间（简化版本） - 调整为短期预测"""
    if sigma <= 0:
        return 0.5  # 默认500ms

    price_ratio = order_price / current_price
    if price_ratio <= 0:
        return 1.0  # 最长1秒

    log_ratio = abs(np.log(price_ratio))
    adjusted_drift = abs(mu - 0.5 * sigma * sigma)

    # 简化的时间估计公式
    if adjusted_drift > 0:
        estimated_time = log_ratio / adjusted_drift
    else:
        estimated_time = log_ratio / (sigma * sigma)

    return min(max(estimated_time, 0.01), 1.0)  # 限制在10ms到1秒之间

class GBMExecutionModel:
    """GBM执行概率模型 - 添加GPU加速支持"""

    def __init__(self, time_horizons=[0.1, 0.2, 0.5, 1.0], use_gpu=False, model_type='randomforest'):
        """
        初始化模型

        Args:
            time_horizons: 时间范围列表
            use_gpu: 是否使用GPU加速
            model_type: 模型类型 ('randomforest', 'xgboost')
        """
        self.time_horizons = time_horizons
        self.prob_models = {}  # 概率预测模型
        self.time_model = None  # 时间预测模型
        self.feature_scaler = None
        self.use_gpu = use_gpu
        self.model_type = model_type

        # 检查GPU可用性
        if use_gpu and not CUML_AVAILABLE and model_type == 'randomforest':
            print("警告: GPU RandomForest不可用，使用CPU版本")
            self.use_gpu = False

        # 为每个时间范围创建模型
        for horizon in time_horizons:
            self.prob_models[horizon] = self._create_model()

        # 执行时间预测模型
        self.time_model = self._create_model(is_time_model=True)

    def _create_model(self, is_time_model=False):
        """创建模型实例"""
        if self.model_type == 'randomforest':
            if self.use_gpu and CUML_AVAILABLE:
                return CumlRandomForestRegressor(
                    n_estimators=100,
                    max_depth=15 if is_time_model else 10,
                    random_state=42
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15 if is_time_model else 10,
                    random_state=42,
                    n_jobs=-1  # 使用所有CPU核心
                )
        elif self.model_type == 'xgboost' and XGB_AVAILABLE:
            # 使用新的GPU配置方式（XGBoost 2.0+）
            if self.use_gpu:
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=15 if is_time_model else 10,
                    random_state=42,
                    tree_method='hist',  # 使用hist方法
                    device='cuda'       # 指定CUDA设备
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=15 if is_time_model else 10,
                    random_state=42,
                    tree_method='hist'  # CPU版本
                )
        else:
            # 默认使用sklearn RandomForest
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15 if is_time_model else 10,
                random_state=42,
                n_jobs=-1
            )

    def train_models(self, X, prob_labels_matrix, y_time):
        """训练所有模型

        Args:
            X: 特征矩阵
            prob_labels_matrix: 概率标签矩阵 (n_samples, n_horizons)
            y_time: 时间标签
        """
        print("训练执行概率预测模型...")

        # 训练概率预测模型
        for i, horizon in enumerate(self.time_horizons):
            print(f"训练 {horizon}秒 时间范围的概率模型...")
            y_prob_horizon = prob_labels_matrix[:, i]
            self.prob_models[horizon].fit(X, y_prob_horizon)

        # 训练执行时间预测模型
        print("训练执行时间预测模型...")
        if len(y_time) > 0:
            self.time_model.fit(X, y_time)

        print("模型训练完成!")

    def predict_execution_probabilities(self, X):
        """预测各时间范围的执行概率"""
        predictions = {}
        for horizon in self.time_horizons:
            if horizon in self.prob_models:
                predictions[horizon] = self.prob_models[horizon].predict(X)
        return predictions

    def predict_execution_time(self, X):
        """预测执行时间"""
        if self.time_model is not None:
            return self.time_model.predict(X)
        return None

    def save_models(self, filepath_prefix):
        """保存模型"""
        model_data = {
            'prob_models': self.prob_models,
            'time_model': self.time_model,
            'time_horizons': self.time_horizons,
            'use_gpu': self.use_gpu,
            'model_type': self.model_type
        }

        with open(f"{filepath_prefix}_gbm_execution_models.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到 {filepath_prefix}_gbm_execution_models.pkl")

    def load_models(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.prob_models = model_data['prob_models']
        self.time_model = model_data['time_model']
        self.time_horizons = model_data['time_horizons']
        self.use_gpu = model_data.get('use_gpu', False)
        self.model_type = model_data.get('model_type', 'randomforest')

        print(f"模型已从 {filepath} 加载")

class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model: GBMExecutionModel):
        self.model = model

    def evaluate_probability_models(self, X_test, prob_labels_matrix_test):
        """评估概率预测模型

        Args:
            X_test: 测试特征
            prob_labels_matrix_test: 测试概率标签矩阵
        """
        print("\n" + "="*60)
        print("执行概率预测模型评估")
        print("="*60)

        results = {}

        for i, horizon in enumerate(self.model.time_horizons):
            y_true = prob_labels_matrix_test[:, i]
            y_pred = self.model.prob_models[horizon].predict(X_test)

            # 回归指标
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            # 相关性
            correlation = np.corrcoef(y_true, y_pred)[0, 1]

            # 分类评估（将概率转换为二分类）
            y_true_binary = (y_true > 0.5).astype(int)
            y_pred_binary = (y_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            try:
                auc = roc_auc_score(y_true_binary, y_pred)
            except:
                auc = 0.5

            results[horizon] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'correlation': correlation,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }

            print(f"\n{horizon}秒时间范围:")
            print(f"  回归指标:")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    相关性: {correlation:.4f}")
            print(f"  分类指标:")
            print(f"    准确率: {accuracy:.4f}")
            print(f"    精确率: {precision:.4f}")
            print(f"    召回率: {recall:.4f}")
            print(f"    F1分数: {f1:.4f}")
            print(f"    AUC: {auc:.4f}")

        return results

    def evaluate_time_model(self, X_test, y_test_time):
        """评估执行时间预测模型"""
        print("\n" + "="*60)
        print("执行时间预测模型评估")
        print("="*60)

        if self.model.time_model is None or len(y_test_time) == 0:
            print("时间预测模型不可用或无测试数据")
            return None

        y_true = np.array(y_test_time)
        y_pred = self.model.time_model.predict(X_test)

        # 回归指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # 相关性
        correlation = np.corrcoef(y_true, y_pred)[0, 1]

        # 相对误差
        relative_errors = np.abs(y_pred - y_true) / (y_true + 1e-8)
        mean_relative_error = np.mean(relative_errors)
        median_relative_error = np.median(relative_errors)

        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation,
            'mean_relative_error': mean_relative_error,
            'median_relative_error': median_relative_error
        }

        print(f"MAE: {mae:.3f} 秒")
        print(f"RMSE: {rmse:.3f} 秒")
        print(f"相关性: {correlation:.4f}")
        print(f"平均相对误差: {mean_relative_error:.2%}")
        print(f"中位数相对误差: {median_relative_error:.2%}")

        return results

    def plot_evaluation_results(self, X_test, prob_labels_matrix_test, y_test_time, save_path=None):
        """绘制评估结果图表"""

        # 概率预测结果图
        n_horizons = len(self.model.time_horizons)
        if n_horizons > 0:
            fig, axes = plt.subplots(2, (n_horizons + 1) // 2, figsize=(15, 10))
            if n_horizons == 1:
                axes = [axes]
            elif n_horizons <= 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i, horizon in enumerate(self.model.time_horizons):
                y_true = prob_labels_matrix_test[:, i]
                y_pred = self.model.prob_models[horizon].predict(X_test)

                if i < len(axes):
                    axes[i].scatter(y_true, y_pred, alpha=0.5, s=1)
                    axes[i].plot([0, 1], [0, 1], 'r--', lw=2)
                    axes[i].set_xlabel('实际概率')
                    axes[i].set_ylabel('预测概率')
                    axes[i].set_title(f'{horizon}s执行概率预测')
                    axes[i].grid(True, alpha=0.3)

            # 隐藏多余的子图
            for j in range(n_horizons, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_probability_predictions.png", dpi=300, bbox_inches='tight')
            plt.show()

        # 执行时间预测结果图
        if self.model.time_model is not None and len(y_test_time) > 0:
            y_true_time = np.array(y_test_time)
            y_pred_time = self.model.time_model.predict(X_test)

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.scatter(y_true_time, y_pred_time, alpha=0.5, s=1)
            plt.plot([y_true_time.min(), y_true_time.max()],
                    [y_true_time.min(), y_true_time.max()], 'r--', lw=2)
            plt.xlabel('实际执行时间 (秒)')
            plt.ylabel('预测执行时间 (秒)')
            plt.title('执行时间预测结果')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            residuals = y_pred_time - y_true_time
            plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('预测误差 (秒)')
            plt.ylabel('频次')
            plt.title('预测误差分布')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_time_predictions.png", dpi=300, bbox_inches='tight')
            plt.show()

def run_gbm_model_training_evaluation(use_gpu=False, model_type='randomforest'):
    """运行完整的GBM模型训练和评估流程

    Args:
        use_gpu: 是否使用GPU加速
        model_type: 模型类型 ('randomforest', 'xgboost')
    """

    print("开始GBM执行概率模型训练和评估...")
    print(f"配置: GPU={use_gpu}, 模型类型={model_type}")
    print(f"时间范围: 100ms - 1s (超短期执行预测)")

    # 1. 数据收集
    print("\n第一步：收集训练数据")

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

    # 收集数据 - 使用100ms到1s的时间范围
    collector = GBMExecutionDataCollector(time_horizons=[0.1, 0.2, 0.5, 1.0], levels=5)
    collect_task = collector.collect_data(0)
    features_list, prob_labels_list, time_labels = collect_task(hbt, recorder.recorder)

    hbt.close()

    # 转换为numpy数组
    X = np.array(features_list)
    prob_labels_matrix = np.array(prob_labels_list)  # 将列表转换为矩阵
    y_time = np.array(time_labels)

    print(f"收集到 {len(X)} 个训练样本，特征维度: {X.shape[1]}")
    print(f"概率标签矩阵形状: {prob_labels_matrix.shape}")

    # 2. 数据分割
    print("\n第二步：分割训练和测试数据")

    # 分割数据
    X_train, X_test, prob_train, prob_test, y_train_time, y_test_time = train_test_split(
        X, prob_labels_matrix, y_time, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")

    # 3. 模型训练
    print("\n第三步：训练模型")

    model = GBMExecutionModel(
        time_horizons=collector.time_horizons,
        use_gpu=use_gpu,
        model_type=model_type
    )
    model.train_models(X_train, prob_train, y_train_time)

    # 4. 模型评估
    print("\n第四步：评估模型")

    evaluator = ModelEvaluator(model)

    # 评估概率预测
    prob_results = evaluator.evaluate_probability_models(X_test, prob_test)

    # 评估时间预测
    time_results = evaluator.evaluate_time_model(X_test, y_test_time)

    # 5. 可视化结果
    print("\n第五步：生成评估图表")

    evaluator.plot_evaluation_results(
        X_test, prob_test, y_test_time,
        save_path="gbm_model_evaluation_100ms_1s"
    )

    # 6. 保存模型
    print("\n第六步：保存模型")

    model.save_models("trained_gbm_100ms_1s")

    # 7. 生成详细报告
    print("\n第七步：生成评估报告")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"gbm_model_evaluation_report_100ms_1s_{timestamp}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GBM执行概率模型评估报告 (100ms-1s范围)\n")
        f.write("="*60 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"时间范围: 100ms - 1s\n")
        f.write(f"GPU加速: {use_gpu}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"训练样本数: {len(X_train)}\n")
        f.write(f"测试样本数: {len(X_test)}\n")
        f.write(f"特征维度: {X.shape[1]}\n\n")

        f.write("执行概率预测模型评估结果:\n")
        f.write("-" * 40 + "\n")
        for horizon, results in prob_results.items():
            f.write(f"\n{horizon}秒时间范围:\n")
            for metric, value in results.items():
                f.write(f"  {metric}: {value:.4f}\n")

        if time_results:
            f.write("\n执行时间预测模型评估结果:\n")
            f.write("-" * 40 + "\n")
            for metric, value in time_results.items():
                if 'error' in metric:
                    f.write(f"  {metric}: {value:.2%}\n")
                else:
                    f.write(f"  {metric}: {value:.4f}\n")

    print(f"评估报告已保存到: {report_path}")
    print("\nGBM模型训练和评估完成!")
    print("\n模型特点:")
    print("✓ 专注于100ms-1s超短期执行预测")
    print("✓ 100ms高频数据收集")
    print("✓ 动态时间范围支持")
    print("✓ GPU加速选项")
    print("✓ 多种模型类型支持")

    return model, prob_results, time_results

if __name__ == "__main__":
    # 运行完整的模型训练和评估流程
    # 可以选择是否使用GPU和模型类型

    # CPU版本 RandomForest
    model, prob_results, time_results = run_gbm_model_training_evaluation(
        use_gpu=False,
        model_type='randomforest'
    )

    # 如果有GPU，可以尝试：
    # model, prob_results, time_results = run_gbm_model_training_evaluation(
    #     use_gpu=True,
    #     model_type='randomforest'  # 需要安装cuML
    # )

    # 或使用XGBoost (CPU/GPU都支持):
    # model, prob_results, time_results = run_gbm_model_training_evaluation(
    #     use_gpu=True,
    #     model_type='xgboost'  # 需要安装xgboost[gpu]
    # )