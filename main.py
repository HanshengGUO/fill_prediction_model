import numpy as np
import pandas as pd
from numba import njit, float64, int32, boolean, types
from numba.typed import List, Dict
from numba.experimental import jitclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss, 
                           log_loss, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import pickle

# 创建必要的文件夹
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 设置日志记录
def setup_logging():
    """设置日志记录配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/execution_prediction_{timestamp}.log'
    
    # 创建logger
    logger = logging.getLogger('execution_prediction')
    logger.setLevel(logging.INFO)
    
    # 清除之前的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 全局logger
logger = setup_logging()

# 导入现有的工具函数和类
from gbm_model_training_evaluation import (
    MW, estimate_gbm_parameters, 
    calculate_execution_probability_gbm
)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info("✓ LightGBM可用")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("✗ LightGBM不可用")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    logger.info("✓ CatBoost可用")
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.info("✗ CatBoost不可用")

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
from hftbacktest.order import FILLED, PARTIALLY_FILLED

# BUY_EVENT 应该从 hftbacktest 导入
# BUY_EVENT = 1  # 这是备用定义，实际应该从hftbacktest导入 

@njit
def _calculate_slope(y):
    """一个简单的辅助函数，用于计算线性趋势（斜率）"""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    # 使用简单的线性回归公式计算斜率
    # formula: (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - (sum(x))^2)
    n = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

@njit
def calculate_market_features(depth_or_history, trades, current_time, levels=5, time_window=5000000000):
    """
    计算约50个高级市场微观结构特征。
    
    特征列表 (共49维):
    - [订单簿特征 (15维)]
      - 0: spread (价差)
      - 1: relative_spread (相对价差, bps)
      - 2-6: bid_qty_L1..L5 (买方前5档挂单量)
      - 7-11: ask_qty_L1..L5 (卖方前5档挂单量)
      - 12: top_level_qty_imbalance (一档买卖量不平衡)
      - 13: total_qty_imbalance_5L (五档总买卖量不平衡)
      - 14: top_level_liquidity_ratio (一档流动性占比)
    - [交易特征 (10维)]
      - 15: trade_count (成交笔数)
      - 16: taker_buy_volume (主动买成交量)
      - 17: taker_sell_volume (主动卖成交量)
      - 18: volume_imbalance (成交量不平衡)
      - 19: count_imbalance (成交笔数不平衡)
      - 20: vwap (成交量加权平均价)
      - 21: avg_trade_size (平均成交大小)
      - 22: max_trade_size (最大单笔成交)
      - 23: volume_per_second (每秒成交量)
      - 24: trades_per_second (每秒成交笔数)
    - [时序/波动率特征 (16维)]
      - 25: mid_price (当前中间价, 作为参考基准)
      - 26: weighted_avg_price (WAP, 当前加权平均价)
      - 27: mid_price_volatility (中间价波动率)
      - 28: wap_volatility (WAP波动率)
      - 29: spread_volatility (价差波动率)
      - 30: mid_price_trend (中间价趋势)
      - 31: wap_trend (WAP趋势)
      - 32: imbalance_trend (订单簿不平衡趋势)
      - 33-37: bid_qty_vol_L1..L5 (买方各档位量波动)
      - 38-42: ask_qty_vol_L1..L5 (卖方各档位量波动)
    - [交叉特征 (8维)]
      - 43: vwap_vs_mid_price_diff (VWAP与中间价差)
      - 44: bid_consumption_rate (买一档消耗速度)
      - 45: ask_consumption_rate (卖一档消耗速度)
      - 46: bid_qty_change_L1 (买一档量变化)
      - 47: ask_qty_change_L1 (卖一档量变化)
      - 48: market_pressure (市场压力)
      - 49: market_aggressiveness (市场侵略性)
      - 50: price_impact_proxy (价格冲击代理)
    """
    # === 0. 预处理和历史数据提取 ===
    if not isinstance(depth_or_history, list) or len(depth_or_history) < 2:
        return np.zeros(49)
    
    # 使用最新的orderbook作为当前状态
    depth = depth_or_history[-1]
    if depth.best_bid <= 0 or depth.best_ask <= 0:
        return np.zeros(49)

    # 提取历史序列用于计算趋势和波动率
    history_size = len(depth_or_history)
    mid_prices_series = np.zeros(history_size)
    waps_series = np.zeros(history_size)
    spreads_series = np.zeros(history_size)
    obi_series = np.zeros(history_size) # Order Book Imbalance
    
    # 历史各档位挂单量
    hist_bid_qty = np.zeros((history_size, levels))
    hist_ask_qty = np.zeros((history_size, levels))

    for i, h_depth in enumerate(depth_or_history):
        if h_depth.best_bid > 0 and h_depth.best_ask > 0:
            mid_p = (h_depth.best_bid + h_depth.best_ask) / 2.0
            mid_prices_series[i] = mid_p
            spreads_series[i] = h_depth.best_ask - h_depth.best_bid
            
            b_qty1 = h_depth.bid_qty_at_tick(h_depth.best_bid_tick)
            a_qty1 = h_depth.ask_qty_at_tick(h_depth.best_ask_tick)
            
            if b_qty1 + a_qty1 > 0:
                waps_series[i] = (h_depth.best_bid * a_qty1 + h_depth.best_ask * b_qty1) / (b_qty1 + a_qty1)
            else:
                waps_series[i] = mid_p

            total_b_qty, total_a_qty = 0.0, 0.0
            for j in range(levels):
                b_q = h_depth.bid_qty_at_tick(h_depth.best_bid_tick - j)
                a_q = h_depth.ask_qty_at_tick(h_depth.best_ask_tick + j)
                total_b_qty += b_q
                total_a_qty += a_q
                hist_bid_qty[i, j] = b_q
                hist_ask_qty[i, j] = a_q
            
            if total_b_qty + total_a_qty > 0:
                obi_series[i] = (total_b_qty - total_a_qty) / (total_b_qty + total_a_qty)

    # === 1. 订单簿特征 (Order Book Features) - 15维 ===
    ob_features = np.zeros(15)
    best_bid = depth.best_bid
    best_ask = depth.best_ask
    mid_price = (best_bid + best_ask) / 2.0
    
    # 基础价差
    ob_features[0] = best_ask - best_bid
    if mid_price > 0:
        ob_features[1] = ob_features[0] / mid_price * 10000  # relative_spread in bps

    # 各档位挂单量
    total_bid_qty, total_ask_qty = 0.0, 0.0
    for i in range(levels):
        bid_qty = depth.bid_qty_at_tick(depth.best_bid_tick - i)
        ask_qty = depth.ask_qty_at_tick(depth.best_ask_tick + i)
        ob_features[2 + i] = bid_qty
        ob_features[7 + i] = ask_qty
        total_bid_qty += bid_qty
        total_ask_qty += ask_qty
        
    # 不平衡指标
    best_bid_qty = ob_features[2]
    best_ask_qty = ob_features[7]
    if best_bid_qty + best_ask_qty > 0:
        ob_features[12] = (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
    if total_bid_qty + total_ask_qty > 0:
        ob_features[13] = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
        ob_features[14] = (best_bid_qty + best_ask_qty) / (total_bid_qty + total_ask_qty) # 一档流动性占比

    # === 2. 交易特征 (Trade Features) - 10维 ===
    trade_features = np.zeros(10)
    if len(trades) > 0:
        total_volume, buy_volume, sell_volume = 0.0, 0.0, 0.0
        buy_trades, sell_trades = 0, 0
        volume_weighted_price = 0.0
        max_trade_size = 0.0
        
        # 筛选时间窗口内的交易并处理
        recent_trades_count = 0
        for trade in trades:
            if current_time - trade.local_ts <= time_window:
                recent_trades_count += 1
        
        if recent_trades_count > 0:
            for trade in trades:
                if current_time - trade.local_ts <= time_window:
                    volume = trade.qty
                    price = trade.px
                    total_volume += volume
                    volume_weighted_price += price * volume
                    if volume > max_trade_size:
                        max_trade_size = volume

                    if trade.ev & BUY_EVENT:
                        buy_volume += volume
                        buy_trades += 1
                    else:
                        sell_volume += volume
                        sell_trades += 1

            trade_features[0] = float(recent_trades_count)
            trade_features[1] = buy_volume
            trade_features[2] = sell_volume
            if total_volume > 0:
                trade_features[3] = (buy_volume - sell_volume) / total_volume
                trade_features[5] = volume_weighted_price / total_volume  # VWAP
                trade_features[6] = total_volume / recent_trades_count  # Avg trade size
            if recent_trades_count > 0:
                trade_features[4] = float(buy_trades - sell_trades) / recent_trades_count
                
            trade_features[7] = max_trade_size
            
            time_delta_secs = time_window / 1e9
            trade_features[8] = total_volume / time_delta_secs
            trade_features[9] = recent_trades_count / time_delta_secs

    # === 3. 时序/波动率特征 (Time-series / Volatility Features) - 16维 ===
    ts_features = np.zeros(16)
    ts_features[0] = mid_price
    if best_bid_qty + best_ask_qty > 0:
        ts_features[1] = (best_bid * best_ask_qty + best_ask * best_bid_qty) / (best_bid_qty + best_ask_qty)
    else:
        ts_features[1] = mid_price

    if history_size > 1:
        ts_features[2] = np.std(mid_prices_series)
        ts_features[3] = np.std(waps_series)
        ts_features[4] = np.std(spreads_series)
        ts_features[5] = _calculate_slope(mid_prices_series)
        ts_features[6] = _calculate_slope(waps_series)
        ts_features[7] = _calculate_slope(obi_series)
        
        # 各档位量的波动率
        for i in range(levels):
            ts_features[8 + i] = np.std(hist_bid_qty[:, i])
            ts_features[13 + i] = np.std(hist_ask_qty[:, i])

    # === 4. 交叉特征 (Cross Features) - 8维 ===
    cross_features = np.zeros(8)
    vwap = trade_features[5]
    if vwap > 0 and mid_price > 0:
        cross_features[0] = (vwap - mid_price) / mid_price

    avg_bid_qty_L1 = np.mean(hist_bid_qty[:, 0])
    avg_ask_qty_L1 = np.mean(hist_ask_qty[:, 0])
    taker_buy_vol = trade_features[1]
    taker_sell_vol = trade_features[2]

    if avg_bid_qty_L1 > 0:
        cross_features[1] = taker_sell_vol / avg_bid_qty_L1 # 买一档消耗速度
    if avg_ask_qty_L1 > 0:
        cross_features[2] = taker_buy_vol / avg_ask_qty_L1 # 卖一档消耗速度

    # 最近订单簿变化
    prev_depth = depth_or_history[-2]
    cross_features[3] = depth.bid_qty_at_tick(depth.best_bid_tick) - prev_depth.bid_qty_at_tick(prev_depth.best_bid_tick)
    cross_features[4] = depth.ask_qty_at_tick(depth.best_ask_tick) - prev_depth.ask_qty_at_tick(prev_depth.best_ask_tick)
    
    # 综合压力指标
    # 市场压力 = 订单簿压力 x 趋势强度 (标准化)
    market_pressure_obi = ob_features[13] # total_qty_imbalance_5L
    wap_trend_norm = ts_features[6] / (np.std(waps_series) + 1e-9) # 标准化WAP趋势
    cross_features[5] = market_pressure_obi * wap_trend_norm
    
    # 市场侵略性 = 成交压力 x 趋势强度 (标准化)
    market_agg_trades = trade_features[3] # volume_imbalance
    cross_features[6] = market_agg_trades * wap_trend_norm
    
    # 价格冲击代理 = 成交不平衡 / 流动性不平衡
    if ob_features[13] != 0: # total_qty_imbalance_5L
        cross_features[7] = trade_features[3] / ob_features[13] # volume_imbalance / total_qty_imbalance

    # === 5. 合并所有特征 ===
    features = np.concatenate((ob_features, trade_features, ts_features, cross_features))
    
    # 返回最终的特征向量，并将任何可能出现的nan/inf替换为0
    # 使用numba兼容的方式处理nan和inf
    for i in range(len(features)):
        if np.isnan(features[i]) or np.isinf(features[i]):
            features[i] = 0.0
    return features

class ForwardLookingDataCollector:
    """前向查看数据收集器 - 事件驱动直接下单标签生成"""
    
    def __init__(self, max_horizon_ms=1000, levels=5, sample_interval=100):
        self.max_horizon_ns = max_horizon_ms * 1_000_000
        self.levels = levels
        self.sample_interval = sample_interval  # 每多少个事件采样一次
        
    def collect_data_and_labels(self, asset_no: int):
        """事件驱动直接下单生成特征和标签 - 单一输出版本"""
        
        max_horizon = self.max_horizon_ns
        levels = self.levels
        sample_interval = self.sample_interval
        
        @njit
        def _collect_data_with_orders(hbt: ROIVectorMarketDepthBacktest, rec: Recorder_,
                                    max_h: int, lv: int, sample_int: int):
            
            feature_samples = []  # 特征样本
            labels = []  # 标签（单一输出）
            
            # 市场状态窗口
            price_window = MW(200)
            volume_window = MW(100)
            
            # orderbook历史窗口 - 存储过去sample_int个事件的orderbook
            orderbook_history = []  # 用于存储depth对象的历史
            
            event_count = 0
            data_points = 0
            next_order_id = 1000  # 起始订单ID
            
            # 注意：在njit函数内部无法使用logger，所以保持print
            print("事件驱动采样模式：每", sample_int, "个事件采样一次")
            print("新数据格式：每个档位生成一个样本，特征包含方向和档位信息")
            
            while True:
                # 等待下一个市场数据事件
                result = hbt.wait_next_feed(
                    include_order_resp=False,
                    timeout=100_000_000_000_000
                )
                
                current_time = hbt.current_timestamp
                
                # 检查结果
                if result == 0:  # 超时
                    if event_count == 0:
                        print("警告: 没有收到任何事件，可能数据有问题")
                        break
                    continue
                elif result == 1:  # 数据结束
                    print("数据结束，当前时间:", current_time)
                    break
                elif result == 2:  # 收到市场数据feed
                    depth = hbt.depth(asset_no)
                    trades = hbt.last_trades(asset_no)

                    
                    # 检查订单簿数据有效性
                    if depth.best_bid <= 0 or depth.best_ask <= 0:
                        continue
                    
                    event_count += 1
                    
                    # 将当前orderbook添加到历史窗口中
                    orderbook_history.append(depth)
                    
                    if event_count % sample_int != 0:
                        continue
                    
                    # 进行采样：计算特征并下单
                    mid_price = (depth.best_bid + depth.best_ask) / 2.0
                    
                    # 计算基础市场特征 - 使用过去sample_int个事件的orderbook历史和交易数据
                    market_features = calculate_market_features(orderbook_history, trades, current_time, lv)
                    
                    price_window.push(mid_price)
                    # 使用买方成交量+卖方成交量作为总成交量（新特征定义中的索引16+17）
                    total_volume = market_features[16] + market_features[17]  # taker_buy_volume + taker_sell_volume
                    volume_window.push(total_volume)
                    
                    # 估计GBM参数
                    price_history = price_window.get_buffer()
                    mu, sigma = estimate_gbm_parameters(price_history, dt=0.05)
                    
                    # 市场状态特征
                    volatility = price_window.std() / mid_price if mid_price > 0 else 0.0
                    volume_rate = volume_window.mean()
                    
                    # 创建基础特征向量（49维市场特征 + 4维状态特征）
                    base_features = np.zeros(53)  # 49 + 4 = 53维
                    base_features[:49] = market_features  # 前49维：所有市场特征
                    base_features[49] = mu
                    base_features[50] = sigma
                    base_features[51] = volatility
                    base_features[52] = volume_rate
                    
                    # 存储当前订单ID，用于后续检查（买卖各5档，总共10个订单）
                    order_ids = []
                    
                    # 为bid side的前几档价格下买单（价格 <= best_bid）
                    for level in range(lv):
                        # 买单：在bid side下单，价格比best_bid低
                        order_price = depth.best_bid - depth.tick_size * level
                        if order_price > 0:
                            hbt.submit_buy_order(asset_no, next_order_id, order_price, 1.0, 
                                               False, False, False)
                            order_ids.append(next_order_id)
                            next_order_id += 1
                    
                    # 为ask side的前几档价格下卖单（价格 >= best_ask）
                    for level in range(lv):
                        # 卖单：在ask side下单，价格比best_ask高
                        order_price = depth.best_ask + depth.tick_size * level
                        hbt.submit_sell_order(asset_no, next_order_id, order_price, 1.0,
                                            False, False, False)
                        order_ids.append(next_order_id)
                        next_order_id += 1
                    
                    # 等待max_horizon时间
                    hbt.elapse(max_h)
                    
                    # 检查订单成交情况并为每个档位生成样本
                    for i, order_id in enumerate(order_ids):
                        orders = hbt.orders(asset_no)
                        order = orders.get(order_id)
                        
                        # 确定成交状态
                        if order is None:
                            # 订单不存在，假设是已成交
                            execution_label = 1.0
                        elif order.status == FILLED or order.status == PARTIALLY_FILLED:
                            # 明确已成交
                            execution_label = 1.0
                        else:
                            # 订单仍存在且未成交
                            execution_label = 0.0
                            # 取消这些未成交的测试订单
                            hbt.cancel(asset_no, order_id, False)
                        
                        # 确定交易方向和档位
                        if i < lv:  # 买方档位
                            direction = 0.0  # 0表示买方
                            level_idx = float(i)  # 档位0-4
                        else:  # 卖方档位
                            direction = 1.0  # 1表示卖方
                            level_idx = float(i - lv)  # 档位0-4
                        
                        # 创建包含方向和档位信息的特征向量（53+2=55维）
                        feature_vector = np.zeros(55)
                        feature_vector[:53] = base_features  # 基础市场特征
                        feature_vector[53] = direction       # 交易方向
                        feature_vector[54] = level_idx       # 档位
                        
                        feature_samples.append(feature_vector)
                        labels.append(execution_label)

                    data_points += 10  # 每次采样产生10个数据点（5买+5卖）
                    
                    # 进度打印
                    if (data_points % 50000 == 0):
                        print("已处理", data_points, "个样本, 事件:", event_count, ", 当前时间:", current_time)
                        print("  当前价格: $", mid_price)

                    # 清空orderbook历史窗口，为下一个采样周期做准备
                    orderbook_history.clear()

                    hbt.clear_last_trades(asset_no)
                    rec.record(hbt)

                else:
                    print("未知返回值:", result)
                
            print("收集完成:", len(feature_samples), "个特征样本,", len(labels), "个标签")
            return feature_samples, labels
        
        return lambda hbt, rec: _collect_data_with_orders(hbt, rec, max_horizon, levels, sample_interval)

class ModelTrainer:
    """单一输出模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.model_names = ['RandomForest']
        
        if LIGHTGBM_AVAILABLE:
            self.model_names.append('LightGBM')
        if CATBOOST_AVAILABLE:
            self.model_names.append('CatBoost')
    
    def create_models(self):
        """创建所有模型（单一输出版本）"""
        
        # 随机森林（使用回归版本支持概率输出）
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=300,              # 增加树的数量
            max_depth=15,                  # 增加深度
            min_samples_split=10,          # 分裂所需最小样本数
            min_samples_leaf=5,            # 叶子节点最小样本数
            max_features='sqrt',           # 每次分裂考虑的特征数
            bootstrap=True,                # 启用bootstrap采样
            oob_score=True,                # out-of-bag评分
            verbose=0,                     # 关闭详细打印
            random_state=42, 
            n_jobs=-1
        )
        
        # LightGBM（直接使用回归器，添加档位单调性约束）
        if LIGHTGBM_AVAILABLE:
            # 创建单调性约束：特征54（档位）为单调递减约束
            # 0: 无约束, 1: 单调递增, -1: 单调递减
            monotone_constraints = [0] * 55  # 55维特征
            monotone_constraints[54] = -1    # 档位特征单调递减
            
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100000,       # 大幅增加到10万轮
                max_depth=12,              # 稍微降低深度避免过拟合
                learning_rate=0.005,       # 进一步降低学习率，精细学习
                subsample=0.8,             # 行采样
                colsample_bytree=0.8,      # 列采样
                reg_alpha=0.2,             # 增加L1正则化
                reg_lambda=0.2,            # 增加L2正则化
                min_child_samples=100,     # 增加叶子节点最小样本数
                min_child_weight=1e-3,     # 叶子节点最小权重
                subsample_freq=1,          # 每轮都进行采样
                feature_fraction=0.8,      # 特征采样比例
                monotone_constraints=monotone_constraints,  # 添加单调性约束
                random_state=42,
                verbose=-1,                # 关闭所有打印
                n_jobs=-1,                 # 多线程训练
                force_col_wise=True        # 强制列wise训练（更稳定）
            )
        
        # CatBoost（直接使用回归器，添加档位单调性约束）
        if CATBOOST_AVAILABLE:
            # CatBoost的单调性约束格式：字符串 "特征索引:约束方向,..."
            # 1: 单调递增, -1: 单调递减, 0: 无约束
            monotone_constraints = "54:-1"  # 档位特征（索引54）单调递减
            
            self.models['CatBoost'] = cb.CatBoostRegressor(
                iterations=100000,         # 大幅增加到10万轮
                depth=10,                  # 适度控制深度
                learning_rate=0.005,       # 进一步降低学习率，精细学习
                l2_leaf_reg=10,            # 增强L2正则化
                border_count=128,          # 边界点数量
                bagging_temperature=1,     # 贝叶斯bootstrap温度
                random_strength=1,         # 随机强度
                subsample=0.8,             # 行采样
                colsample_bylevel=0.8,     # 每层列采样
                od_type='Iter',            # early stopping类型
                od_wait=500,               # 大幅增加early stopping等待轮数
                use_best_model=True,       # 使用最佳模型
                monotone_constraints=monotone_constraints,  # 添加单调性约束
                random_seed=42,
                verbose=False,             # 关闭所有打印
                thread_count=-1            # 多线程训练
            )
    
    def train_all_models(self, X_train, y_train):
        """训练所有模型"""
        logger.info("开始训练所有模型...")
        logger.info(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
        logger.info("特征说明: 前53维为市场特征，第54维为交易方向(0买/1卖)，第55维为档位(0-4)")
        
        # 为支持early stopping，将训练数据进一步分割为训练集和验证集
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=None
        )
        logger.info(f"分割后: 训练集 {X_train_split.shape}, 验证集 {X_val_split.shape}")
        
        self.create_models()
        
        for name in self.model_names:
            logger.info(f"训练 {name} 模型...")
            if name in ['LightGBM', 'CatBoost']:
                logger.info(f"  - {name} 使用档位单调性约束（档位越高，预测概率越低）")
            
            if name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                # LightGBM支持early stopping
                self.models[name].fit(
                    X_train_split, y_train_split,
                    eval_set=[(X_val_split, y_val_split)],
                    callbacks=[lgb.early_stopping(stopping_rounds=500, verbose=False)]  # 大幅增加patience
                )
                logger.info(f"  - LightGBM early stopping: 最佳轮数 {self.models[name].best_iteration_}")
                
            elif name == 'CatBoost' and CATBOOST_AVAILABLE:
                # CatBoost内置early stopping
                self.models[name].fit(
                    X_train_split, y_train_split,
                    eval_set=(X_val_split, y_val_split),
                    verbose=False
                )
                logger.info(f"  - CatBoost early stopping: 最佳轮数 {self.models[name].get_best_iteration()}")
                
            else:
                # RandomForest等其他模型
                self.models[name].fit(X_train, y_train)
                
    def predict_all_models(self, X_test):
        """所有模型预测（输出单一概率值）"""
        predictions = {}
        
        for name in self.model_names:
            # 所有模型都使用predict方法输出连续值（概率）
            predictions[name] = self.models[name].predict(X_test)
            
            # 将预测结果限制在[0,1]区间内（概率范围）
            predictions[name] = np.clip(predictions[name], 0.0, 1.0)
                
        return predictions

class ModelEvaluator:
    """全面的模型评估器 - 单一输出版本"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_all_models(self, y_true, predictions, X_test=None):
        """评估所有模型（单一输出版本）"""
        logger.info("="*60)
        logger.info("模型评估结果 (单一输出)")
        logger.info("="*60)
        
        for model_name, y_pred in predictions.items():
            logger.info(f"\n{model_name} 模型:")
            logger.info("-" * 40)
            
            results = self.calculate_metrics(y_true, y_pred, X_test)
            self.results[model_name] = results
            
            # 显示总体指标
            overall_results = results['overall']
            logger.info("总体指标:")
            logger.info(f"  ROC AUC: {overall_results['roc_auc']:.4f}")
            logger.info(f"  Brier Score: {overall_results['brier_score']:.4f}")
            logger.info(f"  准确率: {overall_results['accuracy']:.4f}")
            logger.info(f"  精确率: {overall_results['precision']:.4f}")
            logger.info(f"  召回率: {overall_results['recall']:.4f}")
            logger.info(f"  F1分数: {overall_results['f1_score']:.4f}")
            
            # 如果提供了特征数据，显示按方向和档位分组的指标
            if X_test is not None and 'by_direction' in results:
                logger.info("\n按交易方向分组:")
                for direction, metrics in results['by_direction'].items():
                    logger.info(f"  {direction}: AUC={metrics['roc_auc']:.3f}, F1={metrics['f1_score']:.3f}, 样本数={metrics['count']}")
                
                logger.info("\n按档位分组:")
                for level, metrics in results['by_level'].items():
                    logger.info(f"  {level}: AUC={metrics['roc_auc']:.3f}, F1={metrics['f1_score']:.3f}, 样本数={metrics['count']}")
                
                # 检查单调性
                self.check_monotonicity(y_pred, X_test, model_name)
    
    def calculate_metrics(self, y_true, y_pred_proba, X_test=None):
        """计算单一输出评估指标"""
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算总体指标
        overall_metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        results = {'overall': overall_metrics}
        
        # 如果提供了特征数据，计算分组指标
        if X_test is not None:
            # 按交易方向分组评估
            direction_results = {}
            for direction_val, direction_name in [(0, '买方'), (1, '卖方')]:
                mask = X_test[:, 53] == direction_val  # 特征53是交易方向
                if np.sum(mask) > 0:
                    direction_results[direction_name] = {
                        'roc_auc': roc_auc_score(y_true[mask], y_pred_proba[mask]),
                        'brier_score': brier_score_loss(y_true[mask], y_pred_proba[mask]),
                        'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                        'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                        'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                        'f1_score': f1_score(y_true[mask], y_pred[mask], zero_division=0),
                        'count': np.sum(mask)
                    }
            
            # 按档位分组评估
            level_results = {}
            for level in range(5):  # 档位0-4
                mask = X_test[:, 54] == level  # 特征54是档位
                if np.sum(mask) > 0:
                    level_results[f'档位{level}'] = {
                        'roc_auc': roc_auc_score(y_true[mask], y_pred_proba[mask]),
                        'brier_score': brier_score_loss(y_true[mask], y_pred_proba[mask]),
                        'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                        'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                        'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                        'f1_score': f1_score(y_true[mask], y_pred[mask], zero_division=0),
                        'count': np.sum(mask)
                    }
            
            results['by_direction'] = direction_results
            results['by_level'] = level_results
        
        return results
    
    def check_monotonicity(self, y_pred, X_test, model_name):
        """检查模型预测的单调性（档位越高，预测概率越低）"""
        logger.info(f"\n{model_name} 单调性检查:")
        
        # 按交易方向分别检查
        for direction_val, direction_name in [(0, '买方'), (1, '卖方')]:
            direction_mask = X_test[:, 53] == direction_val
            if np.sum(direction_mask) == 0:
                continue
                
            direction_data = X_test[direction_mask]
            direction_pred = y_pred[direction_mask]
            
            # 计算每个档位的平均预测概率
            level_probs = []
            for level in range(5):
                level_mask = direction_data[:, 54] == level
                if np.sum(level_mask) > 0:
                    avg_prob = np.mean(direction_pred[level_mask])
                    level_probs.append(avg_prob)
                else:
                    level_probs.append(np.nan)
            
            # 检查是否单调递减
            valid_probs = [p for p in level_probs if not np.isnan(p)]
            is_monotonic = all(valid_probs[i] >= valid_probs[i+1] for i in range(len(valid_probs)-1))
            
            logger.info(f"  {direction_name}: 档位0-4平均概率 = {[f'{p:.3f}' if not np.isnan(p) else 'N/A' for p in level_probs]}")
            logger.info(f"  {direction_name}: 单调递减 = {'是' if is_monotonic else '否'}")
    
    def plot_evaluation_results(self, y_true, predictions, X_test=None, save_path=None):
        """绘制评估结果图表（单一输出版本）"""
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
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall曲线
        plt.subplot(2, 3, 2)
        for model_name, y_pred in predictions.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            plt.plot(recall, precision, label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
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
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测概率分布
        plt.subplot(2, 3, 4)
        for model_name, y_pred in predictions.items():
            plt.hist(y_pred, bins=30, alpha=0.5, label=model_name, density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Predicted Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 指标对比
        plt.subplot(2, 3, 5)
        metrics = ['roc_auc', 'brier_score', 'f1_score']
        model_names = list(predictions.keys())
        
        metric_values = {metric: [] for metric in metrics}
        for name in model_names:
            for metric in metrics:
                metric_values[metric].append(self.results[name]['overall'][metric])
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, metric_values[metric], width, label=metric, alpha=0.7)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Metrics Comparison')
        plt.xticks(x + width, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 单调性可视化（如果有特征数据）
        plt.subplot(2, 3, 6)
        if X_test is not None:
            # 选择最佳模型
            best_model = max(predictions.keys(), key=lambda x: self.results[x]['overall']['roc_auc'])
            y_pred_best = predictions[best_model]
            
            # 绘制每个档位的平均预测概率
            for direction_val, direction_name in [(0, '买方'), (1, '卖方')]:
                direction_mask = X_test[:, 53] == direction_val
                if np.sum(direction_mask) == 0:
                    continue
                    
                direction_data = X_test[direction_mask]
                direction_pred = y_pred_best[direction_mask]
                
                level_probs = []
                for level in range(5):
                    level_mask = direction_data[:, 54] == level
                    if np.sum(level_mask) > 0:
                        avg_prob = np.mean(direction_pred[level_mask])
                        level_probs.append(avg_prob)
                    else:
                        level_probs.append(np.nan)
                
                valid_levels = [i for i, p in enumerate(level_probs) if not np.isnan(p)]
                valid_probs = [level_probs[i] for i in valid_levels]
                
                plt.plot(valid_levels, valid_probs, marker='o', label=f'{direction_name}', linewidth=2)
            
            plt.xlabel('档位')
            plt.ylabel('平均预测概率')
            plt.title(f'单调性检查 ({best_model})')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '需要特征数据\n进行单调性分析', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()

def run_execution_probability_prediction(max_horizon_ms=1000):
    """运行完整的执行概率预测流程 - 单一输出版本"""
    
    logger.info(f"开始执行概率预测 - 时间范围: 0ms - {max_horizon_ms}ms")
    logger.info("新架构: 单一模型预测，交易方向和档位作为特征输入")
    
    # 1. 数据收集
    logger.info("\n第一步：收集数据")
    
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

    # 收集特征数据和标签（一步完成）
    collector = ForwardLookingDataCollector(
        max_horizon_ms=max_horizon_ms, 
        levels=5,
        sample_interval=50  # 每50个事件采样一次
    )
    
    collect_task = collector.collect_data_and_labels(0)
    features_list, labels = collect_task(hbt, recorder.recorder)
    
    logger.info(f"收集到 {len(features_list)} 个样本")
    logger.info(f"每个时间点生成10个样本（买卖各5档），所以实际时间点数约为 {len(features_list)//10}")
    
    hbt.close()
    
    # 转换为numpy数组
    X = np.array(features_list)
    y = np.array(labels)
    
    logger.info(f"特征维度: {X.shape} (55维 = 53市场特征 + 1方向 + 1档位)")
    logger.info(f"标签维度: {y.shape} (单一输出)")
    
    # 分析标签分布
    logger.info(f"总体成交概率: {np.mean(y):.3f}")
    
    # 按方向分析
    buy_mask = X[:, 53] == 0  # 买方
    sell_mask = X[:, 53] == 1  # 卖方
    logger.info(f"买方成交概率: {np.mean(y[buy_mask]):.3f} (样本数: {np.sum(buy_mask)})")
    logger.info(f"卖方成交概率: {np.mean(y[sell_mask]):.3f} (样本数: {np.sum(sell_mask)})")
    
    # 按档位分析
    logger.info("各档位成交概率:")
    for level in range(5):
        level_mask = X[:, 54] == level
        if np.sum(level_mask) > 0:
            logger.info(f"  档位{level}: {np.mean(y[level_mask]):.3f} (样本数: {np.sum(level_mask)})")
    
    # 2. 数据分割（时间序列分割，避免未来信息泄露）
    logger.info("\n第二步：分割数据")
    
    # 按时间顺序分割：前80%作为训练集，后20%作为测试集
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    logger.info(f"训练集大小: {len(X_train)} (时间范围: 样本0到{split_idx-1})")
    logger.info(f"测试集大小: {len(X_test)} (时间范围: 样本{split_idx}到{len(X)-1})")
    logger.info("✓ 使用时间序列分割，无未来信息泄露")
    
    # 保存分割后的原始数据到data文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("\n保存原始数据（分割后，预处理前）...")
    
    train_data = {
        'X_train': X_train,
        'y_train': y_train
    }
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open(f'data/train_data_{timestamp}.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'data/test_data_{timestamp}.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    logger.info(f"✓ 训练数据已保存到: data/train_data_{timestamp}.pkl")
    logger.info(f"✓ 测试数据已保存到: data/test_data_{timestamp}.pkl")
    
    # 2.5. 数据预处理（清理和标准化）
    logger.info("\n第二步半：数据预处理")
    
    # 清理异常值
    logger.info("清理NaN和Inf值...")
    X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 检查数据范围
    logger.info(f"训练数据范围: [{np.min(X_train_clean):.3f}, {np.max(X_train_clean):.3f}]")
    
    # 只对前53维特征进行标准化，保持方向和档位特征不变
    logger.info("标准化前53维市场特征（保持方向和档位特征不变）...")
    scaler = RobustScaler()
    
    # 分别处理市场特征和方向档位特征
    X_train_market = X_train_clean[:, :53]  # 前53维市场特征
    X_train_direction_level = X_train_clean[:, 53:]  # 后2维方向档位特征
    
    X_test_market = X_test_clean[:, :53]
    X_test_direction_level = X_test_clean[:, 53:]
    
    # 只标准化市场特征
    X_train_market_scaled = scaler.fit_transform(X_train_market)
    X_test_market_scaled = scaler.transform(X_test_market)
    
    # 重新组合特征
    X_train_scaled = np.concatenate([X_train_market_scaled, X_train_direction_level], axis=1)
    X_test_scaled = np.concatenate([X_test_market_scaled, X_test_direction_level], axis=1)
    
    logger.info(f"标准化后训练数据范围: [{np.min(X_train_scaled):.3f}, {np.max(X_train_scaled):.3f}]")
    logger.info("✓ 数据预处理完成")
    
    # 2.6. 训练数据打乱
    logger.info("\n第二步六：训练数据打乱")
    logger.info("对训练集进行随机打乱（保持X和y的对应关系）...")
    X_train_shuffled, y_train_shuffled = shuffle(X_train_scaled, y_train, random_state=42)
    logger.info("✓ 训练数据打乱完成")
    
    # 3. 模型训练
    logger.info("\n第三步：训练模型")
    
    trainer = ModelTrainer()
    trainer.train_all_models(X_train_shuffled, y_train_shuffled)
    
    # 4. 模型预测
    logger.info("\n第四步：模型预测")
    
    predictions = trainer.predict_all_models(X_test_scaled)
    
    # 5. 模型评估
    logger.info("\n第五步：模型评估")
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(y_test, predictions, X_test_scaled)
    
    # 6. 保存结果
    logger.info("\n第六步：保存模型和评估结果")
    
    # 保存模型
    models_filename = f"models_single_output_0ms_{max_horizon_ms}ms_{timestamp}.pkl"
    with open(models_filename, 'wb') as f:
        pickle.dump(trainer.models, f)
    logger.info(f"✓ 模型已保存到: {models_filename}")
    
    # 保存评估结果
    results_filename = f"evaluation_results_single_output_0ms_{max_horizon_ms}ms_{timestamp}.csv"
    results_df = pd.DataFrame({name: result['overall'] for name, result in evaluator.results.items()}).T
    results_df.to_csv(results_filename)
    logger.info(f"✓ 评估结果已保存到: {results_filename}")
    
    # 保存详细评估结果（包含分组指标）
    detailed_results_filename = f"detailed_evaluation_results_{timestamp}.pkl"
    with open(detailed_results_filename, 'wb') as f:
        pickle.dump(evaluator.results, f)
    logger.info(f"✓ 详细评估结果已保存到: {detailed_results_filename}")
    
    # 确定最佳模型
    best_model = max(evaluator.results.keys(), key=lambda x: evaluator.results[x]['overall']['roc_auc'])
    logger.info(f"最佳模型: {best_model} (ROC AUC: {evaluator.results[best_model]['overall']['roc_auc']:.4f})")
    
    logger.info("\n任务完成！所有结果已保存。")
    
    return trainer, evaluator, predictions

if __name__ == "__main__":
    # 运行预测任务
    run_execution_probability_prediction(
        max_horizon_ms=5_000
    )
    
    logger.info("程序执行完毕！")
