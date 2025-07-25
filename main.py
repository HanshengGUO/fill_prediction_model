import numpy as np
import pandas as pd
from numba import njit, float64, int32
from numba.typed import List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
    print("✓ LightGBM可用")
except ImportError:
    LGB_AVAILABLE = False
    print("✗ LightGBM不可用 - 请运行 'pip install lightgbm'")

try:
    import catboost as cb

    CAT_AVAILABLE = True
    print("✓ CatBoost可用")
except ImportError:
    CAT_AVAILABLE = False
    print("✗ CatBoost不可用 - 请运行 'pip install catboost'")

warnings.filterwarnings("ignore", category=UserWarning, module='catboost')

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


@njit
def estimate_gbm_parameters(price_history, dt=1.0):
    if len(price_history) < 2: return 0.0, 0.01
    log_returns = np.log(price_history[1:] / price_history[:-1])
    if len(log_returns) == 0: return 0.0, 0.01
    mean_log_return = np.mean(log_returns)
    std_log_return = np.std(log_returns)
    volatility = std_log_return / np.sqrt(dt)
    drift = mean_log_return / dt + 0.5 * volatility * volatility
    return drift, max(volatility, 1e-6)


@njit
def calculate_execution_probability_gbm(order_price, current_price, side, mu, sigma, time_horizon):
    if sigma <= 1e-7 or current_price <= 0 or order_price <= 0: return 0.5
    price_ratio = order_price / current_price
    log_ratio = np.log(price_ratio)
    adjusted_drift = mu - 0.5 * sigma * sigma
    d1 = (log_ratio - adjusted_drift * time_horizon) / (sigma * np.sqrt(time_horizon))
    if side:
        prob = 0.5 * (1 - np.tanh(d1 / np.sqrt(2)))
    else:
        prob = 0.5 * (1 + np.tanh(d1 / np.sqrt(2)))
    return min(max(prob, 0.0), 1.0)


@njit
def calculate_order_book_features(depth, levels=5):
    features = np.zeros(10)
    if depth.best_bid <= 0 or depth.best_ask <= 0: return features
    best_bid, best_ask = depth.best_bid, depth.best_ask
    mid_price = (best_bid + best_ask) / 2.0
    features[0] = mid_price
    features[1] = best_ask - best_bid
    features[2] = features[1] / mid_price * 10000
    total_bid_qty, total_ask_qty = 0.0, 0.0
    for i in range(levels):
        total_bid_qty += depth.bid_qty_at_tick(depth.best_bid_tick - i)
        total_ask_qty += depth.ask_qty_at_tick(depth.best_ask_tick + i)
    features[3], features[4] = total_bid_qty, total_ask_qty
    if total_bid_qty + total_ask_qty > 0:
        features[5] = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    best_bid_qty = depth.bid_qty_at_tick(depth.best_bid_tick)
    best_ask_qty = depth.ask_qty_at_tick(depth.best_ask_tick)
    if best_bid_qty + best_ask_qty > 0:
        features[9] = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
    return features


@njit
def calculate_trade_features(trades, current_time, time_window=5_000_000_000):
    features = np.zeros(8)
    if len(trades) == 0: return features
    total_volume, buy_volume, sell_volume, recent_trades = 0.0, 0.0, 0.0, 0
    for trade in trades:
        if current_time - trade.local_ts <= time_window:
            recent_trades += 1
            total_volume += trade.qty
            if trade.ev & BUY_EVENT:
                buy_volume += trade.qty
            else:
                sell_volume += trade.qty
    features[0] = float(recent_trades)
    features[1] = total_volume
    features[2] = buy_volume
    features[3] = sell_volume
    if total_volume > 0:
        features[4] = (buy_volume - sell_volume) / total_volume
    return features


@njit
def generate_label_njit(
        order_price: float,
        side_is_buy: bool,
        current_time: int,
        min_time_horizon: int,
        max_time_horizon: int,
        future_trade_ts: np.ndarray,
        future_trade_px: np.ndarray
):
    start_idx = np.searchsorted(future_trade_ts, current_time + min_time_horizon, side='right')
    end_idx = np.searchsorted(future_trade_ts, current_time + max_time_horizon, side='right')

    label = 0
    if start_idx < len(future_trade_ts):
        future_slice_len = end_idx - start_idx
        if future_slice_len > 0:
            future_prices = future_trade_px[start_idx:end_idx]
            if side_is_buy:
                if np.any(future_prices <= order_price):
                    label = 1
            else:
                if np.any(future_prices >= order_price):
                    label = 1
    return float(label)


class SupervisedDataCollector:
    def __init__(self, time_horizon_seconds=1.0, levels=5, sample_interval_ms=100, warmup_seconds=10):
        self.time_horizon = int(time_horizon_seconds * 1_000_000_000)
        self.min_time_horizon = int(0.1 * 1_000_000_000)
        self.levels = levels
        self.sample_interval = int(sample_interval_ms * 1_000_000)
        self.warmup_period = int(warmup_seconds * 1_000_000_000)

    def collect(self, hbt: ROIVectorMarketDepthBacktest, asset_no: int, full_data: np.ndarray):
        print("开始收集监督学习数据...")

        print("正在从预加载的数据中提取未来成交信息...")
        trade_mask = (full_data['ev'] & TRADE_EVENT) == TRADE_EVENT
        future_trade_ts = full_data['local_ts'][trade_mask]
        future_trade_px = full_data['px'][trade_mask]

        features_list = []
        labels_list = []

        print(f"正在预热回测引擎 ({self.warmup_period / 1e9} 秒)...")
        if hbt.elapse(self.warmup_period) != 0:
            print("警告: 预热期间已到达数据末尾。")
            return np.array(features_list), np.array(labels_list)
        print("预热完成。")

        price_window = np.zeros(200)
        price_idx = 0
        data_points = 0
        last_print_time = hbt.current_timestamp

        while hbt.elapse(self.sample_interval) == 0:
            current_time = hbt.current_timestamp
            depth = hbt.depth(asset_no)
            trades = hbt.last_trades(asset_no)

            if depth.best_bid <= 0 or depth.best_ask <= 0:
                hbt.clear_last_trades(asset_no)
                continue

            if current_time - last_print_time > 5_000_000_000:
                print(f"  ...已处理到时间: {current_time}, 已收集样本: {data_points}")
                last_print_time = current_time

            ob_features = calculate_order_book_features(depth, self.levels)
            trade_features = calculate_trade_features(trades, current_time)
            mid_price = ob_features[0]

            price_window[price_idx] = mid_price
            price_idx = (price_idx + 1) % len(price_window)
            mu, sigma = estimate_gbm_parameters(price_window, dt=0.1)

            for side_is_buy in [True, False]:
                for level in range(self.levels):
                    if side_is_buy:
                        order_price = depth.best_bid - depth.tick_size * level
                    else:
                        order_price = depth.best_ask + depth.tick_size * level

                    if order_price <= 0: continue

                    feature_vector = np.zeros(23)
                    feature_vector[:10] = ob_features
                    feature_vector[10:18] = trade_features
                    feature_vector[18] = mu
                    feature_vector[19] = sigma
                    feature_vector[20] = float(level)
                    feature_vector[21] = 1.0 if side_is_buy else 0.0
                    feature_vector[22] = order_price / mid_price if mid_price > 0 else 1.0

                    label = generate_label_njit(
                        order_price,
                        side_is_buy,
                        current_time,
                        self.min_time_horizon,
                        self.time_horizon,
                        future_trade_ts,
                        future_trade_px
                    )

                    features_list.append(feature_vector)
                    labels_list.append(label)
                    data_points += 1

            hbt.clear_last_trades(asset_no)

        print(f"\n数据收集完成。总共收集到 {len(features_list)} 个样本。")
        return np.array(features_list), np.array(labels_list)


class ModelBenchmarker:
    def __init__(self, models_to_test, random_state=42):
        self.models = models_to_test
        self.random_state = random_state
        self.results = {}

    def run(self, X, y, data_name="Execution"):
        print(f"\n{'=' * 25} 开始对 {data_name} 数据进行基准测试 {'=' * 25}")
        if len(np.unique(y)) < 2:
            print(f"警告: {data_name} 数据只有一个类别 ({np.unique(y)}), 无法进行有意义的训练和评估。跳过。")
            return None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        print(f"成交率 (训练集): {y_train.mean():.2%}, (验证集): {y_val.mean():.2%}")

        print("\n--- 正在评估 Baseline: GBM理论模型 ---")
        gbm_probs = self._get_gbm_predictions(X_val)
        self.results['GBM (Baseline)'] = self._evaluate_predictions(y_val, gbm_probs)

        for name, model in self.models.items():
            print(f"\n--- 正在训练和评估: {name} ---")
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            self.results[name] = self._evaluate_predictions(y_val, y_pred_proba)
            self.models[name] = model

        self._print_results()
        self._plot_results(X_val, y_val)

        print(f"{'=' * 25} {data_name} 数据基准测试完成 {'=' * 25}\n")
        return self.results

    def _get_gbm_predictions(self, X_val):
        mu = X_val[:, 18]
        sigma = X_val[:, 19]
        side = X_val[:, 21].astype(bool)
        relative_price = X_val[:, 22]
        time_horizon = 0.5
        probs = np.array([
            calculate_execution_probability_gbm(relative_price[i], 1.0, side[i], mu[i], sigma[i], time_horizon)
            for i in range(len(X_val))
        ])
        return probs

    def _evaluate_predictions(self, y_true, y_pred_proba):
        return {
            'ROC AUC': roc_auc_score(y_true, y_pred_proba),
            'Brier Score': brier_score_loss(y_true, y_pred_proba),
            'Log Loss': log_loss(y_true, y_pred_proba)
        }

    def _print_results(self):
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values(by='Brier Score', ascending=True)
        print("\n--- 模型评估结果对比 ---")
        print("(Brier Score 和 Log Loss 越低越好, ROC AUC 越高越好)")
        print(results_df)

    def _plot_results(self, X_val, y_val):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.plot([0, 1], [0, 1], 'k--', label='Chance')
        for name in self.results.keys():
            if name == 'GBM (Baseline)':
                y_pred_proba = self._get_gbm_predictions(X_val)
                linestyle = ':'
            else:
                y_pred_proba = self.models[name].predict_proba(X_val)[:, 1]
                linestyle = '-'
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            auc = self.results[name]['ROC AUC']
            ax1.plot(fpr, tpr, linestyle=linestyle, label=f'{name} (AUC = {auc:.3f})')

        ax1.set_title('ROC 曲线')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for name in self.results.keys():
            if name == 'GBM (Baseline)':
                y_pred_proba = self._get_gbm_predictions(X_val)
                linestyle = ':'
            else:
                y_pred_proba = self.models[name].predict_proba(X_val)[:, 1]
                linestyle = '-'
            CalibrationDisplay.from_predictions(y_val, y_pred_proba, n_bins=10, name=name, ax=ax2, strategy='uniform',
                                                linestyle=linestyle)

        ax2.set_title('校准曲线')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def run_supervised_learning_benchmark():
    print("开始监督学习执行概率模型基准测试...")

    data_filename = 'DOGE-USDT-PERP_20250101_merged.npz'
    snapshot_filename = 'DOGE-USDT-PERP_20250101(1)_eod_fixed.npz'
    print(f"正在从文件加载数据: {data_filename}")
    try:
        full_data = np.load(data_filename)['data']
        initial_snapshot = np.load(snapshot_filename)['data']
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 {e.filename}。请确保文件路径正确。")
        return
    except KeyError:
        print("警告: 在 .npz 文件中找不到键 'data'，尝试直接加载。")
        full_data_load = np.load(data_filename)
        initial_snapshot_load = np.load(snapshot_filename)
        full_data = full_data_load[list(full_data_load.keys())[0]]
        initial_snapshot = initial_snapshot_load[list(initial_snapshot_load.keys())[0]]

    asset = (
        BacktestAsset()
        .data([full_data])
        .initial_snapshot(initial_snapshot)
        .linear_asset(1.0)
        .power_prob_queue_model(1.2)
        .no_partial_fill_exchange()
        .tick_size(0.00001)
        .lot_size(1.0)
        .last_trades_capacity(10000)
    )
    hbt = ROIVectorMarketDepthBacktest([asset])

    collector = SupervisedDataCollector()
    X, y = collector.collect(hbt, 0, full_data)
    hbt.close()

    if len(X) == 0:
        print("未能收集到任何数据，程序终止。")
        return

    print(f"数据收集完成。总样本数: {len(X)}, 总标签数: {len(y)}")
    print(f"标签类别分布: {np.unique(y, return_counts=True)}")

    buy_mask = X[:, 21] == 1.0
    sell_mask = X[:, 21] == 0.0
    X_buy, y_buy = X[buy_mask], y[buy_mask]
    X_sell, y_sell = X[sell_mask], y[sell_mask]

    print(f"\n总样本数: {len(X)}")
    print(f"买单样本数: {len(X_buy)}, 卖单样本数: {len(X_sell)}")

    models_to_test_template = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
    }
    if LGB_AVAILABLE:
        models_to_test_template['LightGBM'] = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    if CAT_AVAILABLE:
        models_to_test_template['CatBoost'] = cb.CatBoostClassifier(random_state=42, verbose=0, thread_count=-1,
                                                                    auto_class_weights='Balanced')

    if len(X_buy) > 0:
        buy_benchmarker = ModelBenchmarker(models_to_test_template.copy())
        buy_benchmarker.run(X_buy, y_buy, data_name="买单(Buy-side)")

    if len(X_sell) > 0:
        sell_benchmarker = ModelBenchmarker(models_to_test_template.copy())
        sell_benchmarker.run(X_sell, y_sell, data_name="卖单(Sell-side)")

    print("\n所有基准测试完成！")


if __name__ == "__main__":
    run_supervised_learning_benchmark()