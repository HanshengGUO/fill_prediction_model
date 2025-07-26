import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, float64, int32, boolean, types
from numba.typed import List, Dict
from numba.experimental import jitclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入您现有的工具函数和类
from gbm_model_training_evaluation import (
    MW, estimate_gbm_parameters, 
    calculate_execution_probability_gbm, calculate_order_book_features, 
    calculate_trade_features
)

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

class ExecutionProbabilityEDACollector:
    """成交概率EDA数据收集器 - 使用njit优化"""
    
    def __init__(self, max_horizon_ms=1000, levels=20, sample_interval=100):
        self.max_horizon_ns = max_horizon_ms * 1_000_000
        self.levels = levels
        self.sample_interval = sample_interval
        
    def collect_execution_data(self, asset_no: int):
        """收集成交数据 - 使用njit优化的事件循环"""
        
        max_horizon = self.max_horizon_ns
        levels = self.levels
        sample_interval = self.sample_interval
        
        @njit
        def _collect_execution_data_njit(hbt: ROIVectorMarketDepthBacktest, rec: Recorder_,
                                       max_h: int, lv: int, sample_int: int):
            
            # 存储成交结果的列表
            bid_executions = []  # 每个元素是 [level, executed]
            ask_executions = []  # 每个元素是 [level, executed]
            market_data = []     # 每个元素是 [timestamp, mid_price, spread_bps, ...]
            
            event_count = 0
            data_points = 0
            next_order_id = 1000
            
            print("开始收集EDA数据，每", sample_int, "个事件采样一次，levels=", lv)
            
            while True:
                # 等待下一个市场数据事件
                result = hbt.wait_next_feed(
                    include_order_resp=False,
                    timeout=100_000_000_000_000
                )
                
                current_time = hbt.current_timestamp
                
                if result == 0:  # 超时
                    if event_count == 0:
                        print("警告: 没有收到任何事件")
                        break
                    continue
                elif result == 1:  # 数据结束
                    print("数据结束，时间:", current_time)
                    break
                elif result == 2:  # 收到市场数据
                    depth = hbt.depth(asset_no)
                    trades = hbt.last_trades(asset_no)
                    
                    # 检查订单簿有效性
                    if depth.best_bid <= 0 or depth.best_ask <= 0:
                        continue
                    
                    event_count += 1
                    
                    # 采样控制
                    if event_count % sample_int != 0:
                        continue
                    
                    # 记录市场状态
                    mid_price = (depth.best_bid + depth.best_ask) / 2.0
                    spread = depth.best_ask - depth.best_bid
                    spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0.0
                    
                    # 计算一些基本的市场特征
                    trade_features = calculate_trade_features(trades, current_time)
                    trade_volume = trade_features[1] if len(trade_features) > 1 else 0.0
                    trade_count = trade_features[0] if len(trade_features) > 0 else 0.0
                    
                    # 存储市场数据 [timestamp, mid_price, spread_bps, volume, count]
                    market_record = np.array([
                        float(current_time), mid_price, spread_bps, trade_volume, trade_count
                    ])
                    market_data.append(market_record)
                    
                    # 存储当前批次的订单ID
                    current_orders = []
                    
                    # Bid side: 下买单（maker单，价格 <= best_bid）
                    for level in range(lv):
                        order_price = depth.best_bid - depth.tick_size * level
                        if order_price > 0:
                            hbt.submit_buy_order(asset_no, next_order_id, order_price, 1.0,
                                               False, False, False)
                            current_orders.append((next_order_id, 'bid', level + 1, order_price))
                            next_order_id += 1
                    
                    # Ask side: 下卖单（maker单，价格 >= best_ask）
                    for level in range(lv):
                        order_price = depth.best_ask + depth.tick_size * level
                        hbt.submit_sell_order(asset_no, next_order_id, order_price, 1.0,
                                            False, False, False)
                        current_orders.append((next_order_id, 'ask', level + 1, order_price))
                        next_order_id += 1
                    
                    # 等待指定时间
                    hbt.elapse(max_h)
                    
                    # 检查订单成交情况
                    orders = hbt.orders(asset_no)
                    for order_id, side, level, order_price in current_orders:
                        order = orders.get(order_id)
                        
                        if order is None:
                            # 订单不存在，假设已成交
                            executed = 1.0
                        elif order.status == FILLED or order.status == PARTIALLY_FILLED:
                            # 明确已成交
                            executed = 1.0
                        else:
                            # 订单未成交
                            executed = 0.0
                            # 取消未成交订单
                            hbt.cancel(asset_no, order_id, False)
                        
                        # 记录成交结果 [level, executed]
                        execution_record = np.array([float(level), executed])
                        if side == 'bid':
                            bid_executions.append(execution_record)
                        else:
                            ask_executions.append(execution_record)
                    
                    data_points += len(current_orders)
                    
                    # 进度报告
                    if data_points % 5000 == 0:
                        print("已处理", data_points, "个订单样本, 事件:", event_count)
                        print("  当前价格: $", mid_price, ", 价差:", spread_bps, "bps")
                
                else:
                    print("未知返回值:", result)
            
            print("收集完成:", len(bid_executions), "个bid样本,", len(ask_executions), "个ask样本")
            return bid_executions, ask_executions, market_data
        
        return lambda hbt, rec: _collect_execution_data_njit(hbt, rec, max_horizon, levels, sample_interval)

class ExecutionProbabilityEDA:
    """成交概率探索性数据分析"""
    
    def __init__(self, horizon_ms_list, max_levels=20, sample_interval=100):
        self.horizon_ms_list = horizon_ms_list
        self.max_levels = max_levels
        self.sample_interval = sample_interval
        self.results = {}
        
    def collect_data_for_horizon(self, horizon_ms):
        """为特定时间窗口收集数据"""
        
        print(f"\n开始收集时间窗口 {horizon_ms}ms 的数据...")
        
        # 设置回测环境
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
        
        # 创建数据收集器
        collector = ExecutionProbabilityEDACollector(
            max_horizon_ms=horizon_ms,
            levels=self.max_levels,
            sample_interval=self.sample_interval
        )
        
        # 收集数据
        collect_task = collector.collect_execution_data(0)
        bid_executions, ask_executions, market_data = collect_task(hbt, recorder.recorder)
        
        hbt.close()
        
        # 转换为numpy数组
        bid_executions = np.array(bid_executions)
        ask_executions = np.array(ask_executions)
        market_data = np.array(market_data)
        
        print(f"收集完成: {len(bid_executions)} bid, {len(ask_executions)} ask, {len(market_data)} market")
        
        return {
            'bid_executions': bid_executions,
            'ask_executions': ask_executions,
            'market_data': market_data
        }
    
    def run_full_eda(self):
        """运行完整的EDA分析"""
        
        print("开始执行成交概率EDA分析")
        print(f"时间窗口: {self.horizon_ms_list}")
        print(f"档位数: 1-{self.max_levels}")
        
        # 收集所有时间窗口的数据
        for horizon_ms in self.horizon_ms_list:
            self.results[horizon_ms] = self.collect_data_for_horizon(horizon_ms)
        
        # 分析和可视化
        self.analyze_execution_probabilities()
        self.plot_execution_probability_analysis()
        self.generate_summary_report()
    
    def analyze_execution_probabilities(self):
        """分析成交概率"""
        
        print("\n分析成交概率...")
        
        self.execution_prob_summary = {}
        
        for horizon_ms in self.horizon_ms_list:
            data = self.results[horizon_ms]
            
            bid_executions = data['bid_executions']
            ask_executions = data['ask_executions']
            
            # 计算各档位的成交概率
            bid_probs = {}
            ask_probs = {}
            
            for level in range(1, self.max_levels + 1):
                # Bid成交概率
                if len(bid_executions) > 0:
                    bid_level_mask = bid_executions[:, 0] == level
                    if np.any(bid_level_mask):
                        bid_probs[level] = np.mean(bid_executions[bid_level_mask, 1])
                    else:
                        bid_probs[level] = 0.0
                else:
                    bid_probs[level] = 0.0
                
                # Ask成交概率
                if len(ask_executions) > 0:
                    ask_level_mask = ask_executions[:, 0] == level
                    if np.any(ask_level_mask):
                        ask_probs[level] = np.mean(ask_executions[ask_level_mask, 1])
                    else:
                        ask_probs[level] = 0.0
                else:
                    ask_probs[level] = 0.0
            
            self.execution_prob_summary[horizon_ms] = {
                'bid_probs': bid_probs,
                'ask_probs': ask_probs,
                'sample_count': len(data['market_data'])
            }
            
            print(f"\n时间窗口 {horizon_ms}ms:")
            print(f"  样本数量: {len(data['market_data'])}")
            print(f"  Bid Level 1 成交概率: {bid_probs[1]:.3f}")
            print(f"  Ask Level 1 成交概率: {ask_probs[1]:.3f}")
            if self.max_levels >= 10:
                print(f"  Bid Level 10 成交概率: {bid_probs[10]:.3f}")
                print(f"  Ask Level 10 成交概率: {ask_probs[10]:.3f}")
    
    def plot_execution_probability_analysis(self):
        """绘制成交概率分析图表"""
        
        print("\n生成可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 成交概率 vs 档位 (不同时间窗口) - Bid Side
        plt.subplot(3, 3, 1)
        for horizon_ms in self.horizon_ms_list:
            levels = list(range(1, self.max_levels + 1))
            bid_probs = [self.execution_prob_summary[horizon_ms]['bid_probs'][level] 
                        for level in levels]
            plt.plot(levels, bid_probs, marker='o', label=f'{horizon_ms}ms', linewidth=2)
        
        plt.xlabel('Level')
        plt.ylabel('Execution Probability')
        plt.title('Bid Side: Execution Probability vs Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, self.max_levels)
        
        # 2. 成交概率 vs 档位 (不同时间窗口) - Ask Side
        plt.subplot(3, 3, 2)
        for horizon_ms in self.horizon_ms_list:
            levels = list(range(1, self.max_levels + 1))
            ask_probs = [self.execution_prob_summary[horizon_ms]['ask_probs'][level] 
                        for level in levels]
            plt.plot(levels, ask_probs, marker='o', label=f'{horizon_ms}ms', linewidth=2)
        
        plt.xlabel('Level')
        plt.ylabel('Execution Probability')
        plt.title('Ask Side: Execution Probability vs Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, self.max_levels)
        
        # 3. 成交概率 vs 时间窗口 (不同档位) - Bid Side
        plt.subplot(3, 3, 3)
        selected_levels = [1, 5, 10, 15, 20] if self.max_levels >= 20 else list(range(1, self.max_levels + 1, max(1, self.max_levels // 5)))
        for level in selected_levels:
            if level <= self.max_levels:
                probs = [self.execution_prob_summary[horizon_ms]['bid_probs'][level] 
                        for horizon_ms in self.horizon_ms_list]
                plt.plot(self.horizon_ms_list, probs, marker='s', 
                        label=f'Level {level}', linewidth=2)
        
        plt.xlabel('Time Window (ms)')
        plt.ylabel('Execution Probability')
        plt.title('Bid Side: Execution Probability vs Time Window')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 热力图: Bid成交概率 (档位 vs 时间窗口)
        plt.subplot(3, 3, 4)
        bid_heatmap_data = np.zeros((len(self.horizon_ms_list), self.max_levels))
        for i, horizon_ms in enumerate(self.horizon_ms_list):
            for j in range(self.max_levels):
                level = j + 1
                bid_heatmap_data[i, j] = self.execution_prob_summary[horizon_ms]['bid_probs'][level]
        
        sns.heatmap(bid_heatmap_data, 
                   xticklabels=list(range(1, self.max_levels + 1, max(1, self.max_levels // 10))),
                   yticklabels=[f'{h}ms' for h in self.horizon_ms_list],
                   annot=False, cmap='viridis', cbar_kws={'label': 'Execution Probability'})
        plt.title('Bid Side: Execution Probability Heatmap')
        plt.xlabel('Level')
        plt.ylabel('Time Window')
        
        # 5. 热力图: Ask成交概率 (档位 vs 时间窗口)
        plt.subplot(3, 3, 5)
        ask_heatmap_data = np.zeros((len(self.horizon_ms_list), self.max_levels))
        for i, horizon_ms in enumerate(self.horizon_ms_list):
            for j in range(self.max_levels):
                level = j + 1
                ask_heatmap_data[i, j] = self.execution_prob_summary[horizon_ms]['ask_probs'][level]
        
        sns.heatmap(ask_heatmap_data, 
                   xticklabels=list(range(1, self.max_levels + 1, max(1, self.max_levels // 10))),
                   yticklabels=[f'{h}ms' for h in self.horizon_ms_list],
                   annot=False, cmap='viridis', cbar_kws={'label': 'Execution Probability'})
        plt.title('Ask Side: Execution Probability Heatmap')
        plt.xlabel('Level')
        plt.ylabel('Time Window')
        
        # 6. Bid vs Ask 对比
        plt.subplot(3, 3, 6)
        for horizon_ms in self.horizon_ms_list:
            display_levels = min(10, self.max_levels)
            levels = list(range(1, display_levels + 1))
            bid_probs = [self.execution_prob_summary[horizon_ms]['bid_probs'][level] 
                        for level in levels]
            ask_probs = [self.execution_prob_summary[horizon_ms]['ask_probs'][level] 
                        for level in levels]
            
            diff = np.array(bid_probs) - np.array(ask_probs)
            plt.plot(levels, diff, marker='o', label=f'{horizon_ms}ms', linewidth=2)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Level')
        plt.ylabel('Execution Probability Difference (Bid - Ask)')
        plt.title('Bid vs Ask Execution Probability Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 成交概率分布
        plt.subplot(3, 3, 7)
        all_bid_probs = []
        all_ask_probs = []
        
        for horizon_ms in self.horizon_ms_list:
            bid_probs = list(self.execution_prob_summary[horizon_ms]['bid_probs'].values())
            ask_probs = list(self.execution_prob_summary[horizon_ms]['ask_probs'].values())
            all_bid_probs.extend(bid_probs)
            all_ask_probs.extend(ask_probs)
        
        plt.hist(all_bid_probs, bins=30, alpha=0.7, label='Bid', density=True)
        plt.hist(all_ask_probs, bins=30, alpha=0.7, label='Ask', density=True)
        plt.xlabel('Execution Probability')
        plt.ylabel('Density')
        plt.title('Execution Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 时间窗口效应
        plt.subplot(3, 3, 8)
        avg_bid_probs = []
        avg_ask_probs = []
        
        for horizon_ms in self.horizon_ms_list:
            bid_avg = np.mean(list(self.execution_prob_summary[horizon_ms]['bid_probs'].values()))
            ask_avg = np.mean(list(self.execution_prob_summary[horizon_ms]['ask_probs'].values()))
            avg_bid_probs.append(bid_avg)
            avg_ask_probs.append(ask_avg)
        
        plt.plot(self.horizon_ms_list, avg_bid_probs, marker='o', 
                label='Bid Average', linewidth=2)
        plt.plot(self.horizon_ms_list, avg_ask_probs, marker='s', 
                label='Ask Average', linewidth=2)
        plt.xlabel('Time Window (ms)')
        plt.ylabel('Average Execution Probability')
        plt.title('Time Window Impact on Execution Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. 边际收益递减效应
        plt.subplot(3, 3, 9)
        if len(self.horizon_ms_list) > 1:
            marginal_gains = []
            for i in range(1, len(self.horizon_ms_list)):
                prev_avg = np.mean(list(self.execution_prob_summary[self.horizon_ms_list[i-1]]['bid_probs'].values()))
                curr_avg = np.mean(list(self.execution_prob_summary[self.horizon_ms_list[i]]['bid_probs'].values()))
                marginal_gain = curr_avg - prev_avg
                marginal_gains.append(marginal_gain)
            
            time_intervals = [f'{self.horizon_ms_list[i-1]}-{self.horizon_ms_list[i]}' 
                            for i in range(1, len(self.horizon_ms_list))]
            plt.bar(time_intervals, marginal_gains, alpha=0.7)
            plt.xlabel('Time Window Interval (ms)')
            plt.ylabel('Marginal Execution Probability Gain')
            plt.title('Diminishing Returns of Time Window Extension')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'execution_probability_eda_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成汇总报告"""
        
        print("\n" + "="*80)
        print("成交概率 EDA 分析报告")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建DataFrame
        summary_data = []
        
        for horizon_ms in self.horizon_ms_list:
            for level in range(1, self.max_levels + 1):
                bid_prob = self.execution_prob_summary[horizon_ms]['bid_probs'][level]
                ask_prob = self.execution_prob_summary[horizon_ms]['ask_probs'][level]
                
                summary_data.append({
                    'Time_Window_ms': horizon_ms,
                    'Level': level,
                    'Bid_Execution_Probability': bid_prob,
                    'Ask_Execution_Probability': ask_prob,
                    'Probability_Difference_Bid_Ask': bid_prob - ask_prob,
                    'Average_Probability': (bid_prob + ask_prob) / 2
                })
        
        df = pd.DataFrame(summary_data)
        
        # 保存详细数据
        df.to_csv(f'execution_probability_detailed_{timestamp}.csv', index=False)
        
        # 汇总统计
        print(f"\n数据概览:")
        print(f"时间窗口数量: {len(self.horizon_ms_list)}")
        print(f"档位数量: {self.max_levels}")
        print(f"总数据点: {len(df)}")
        
        print(f"\n各时间窗口样本数量:")
        for horizon_ms in self.horizon_ms_list:
            sample_count = self.execution_prob_summary[horizon_ms]['sample_count']
            print(f"  {horizon_ms}ms: {sample_count} 个采样点")
        
        # 关键发现
        print(f"\n关键发现:")
        
        # 1. 最高和最低成交概率
        max_prob = df['Average_Probability'].max()
        min_prob = df['Average_Probability'].min()
        max_row = df.loc[df['Average_Probability'].idxmax()]
        min_row = df.loc[df['Average_Probability'].idxmin()]
        
        print(f"1. 最高成交概率: {max_prob:.3f} (时间窗口: {max_row['Time_Window_ms']}ms, 档位: {max_row['Level']})")
        print(f"2. 最低成交概率: {min_prob:.3f} (时间窗口: {min_row['Time_Window_ms']}ms, 档位: {min_row['Level']})")
        
        # 2. 时间窗口效应
        avg_by_horizon = df.groupby('Time_Window_ms')['Average_Probability'].mean()
        print(f"3. 时间窗口效应:")
        for horizon_ms, avg_prob in avg_by_horizon.items():
            print(f"   {horizon_ms}ms: 平均成交概率 {avg_prob:.3f}")
        
        # 3. 档位效应
        avg_by_level = df.groupby('Level')['Average_Probability'].mean()
        print(f"4. 档位效应 (前10档):")
        for level in range(1, min(11, self.max_levels + 1)):
            avg_prob = avg_by_level[level]
            print(f"   Level {level}: 平均成交概率 {avg_prob:.3f}")
        
        # 4. Bid vs Ask差异
        avg_diff = df['Probability_Difference_Bid_Ask'].mean()
        print(f"5. Bid vs Ask差异: 平均差异 {avg_diff:.4f}")
        
        # 保存汇总报告
        report_file = f'execution_probability_summary_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("成交概率 EDA 分析报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"分析时间: {datetime.now()}\n")
            f.write(f"时间窗口: {self.horizon_ms_list}\n")
            f.write(f"档位数: 1-{self.max_levels}\n\n")
            
            # 写入详细统计信息
            f.write("详细统计:\n")
            f.write(df.describe().to_string())
            f.write("\n\n")
            
            # 写入关键发现
            f.write("关键发现:\n")
            f.write(f"最高成交概率: {max_prob:.3f}\n")
            f.write(f"最低成交概率: {min_prob:.3f}\n")
            f.write(f"平均Bid-Ask差异: {avg_diff:.4f}\n")
        
        print(f"\n详细报告已保存至: {report_file}")
        print(f"详细数据已保存至: execution_probability_detailed_{timestamp}.csv")

def main():
    """主函数"""
    
    # 设置参数
    horizon_ms_list = [500, 1000, 2000, 5000]  # 您可以根据需要调整
    max_levels = 20
    sample_interval = 50  # 每50个事件采样一次，您可以调整以控制数据量
    
    print("开始成交概率 EDA 分析")
    print(f"时间窗口: {horizon_ms_list}")
    print(f"档位数: 1-{max_levels}")
    print(f"采样间隔: 每{sample_interval}个事件")
    
    # 创建EDA分析器
    eda = ExecutionProbabilityEDA(
        horizon_ms_list=horizon_ms_list,
        max_levels=max_levels,
        sample_interval=sample_interval
    )
    
    # 运行完整分析
    eda.run_full_eda()
    
    print("\n✓ EDA 分析完成！")

if __name__ == "__main__":
    main()