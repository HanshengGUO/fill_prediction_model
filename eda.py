import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import time

from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    DEPTH_EVENT,
    TRADE_EVENT,
    BUY_EVENT,
    SELL_EVENT
)

class DataExplorer:
    """高频交易数据探索分析器"""
    
    def __init__(self, max_events=None):  # 移除事件数限制，处理所有数据
        """
        数据探索器
        Args:
            max_events: 最大处理事件数，None表示处理所有数据
        """
        self.max_events = max_events
        
        # 数据存储
        self.timestamps = []
        self.mid_prices = []
        self.spreads = []
        self.time_intervals = []
        self.price_changes = []
        
        if max_events is None:
            print("将处理所有数据事件（无限制）")
        else:
            print(f"最大处理事件数: {self.max_events:,}")
        
    def analyze_event_driven_data(self):
        """使用wait_next_feed方式分析数据"""
        print("开始事件驱动的数据分析...")
        
        try:
            start_time = time.time()
            
            print("配置资产...")
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

            print("创建回测实例...")
            hbt = ROIVectorMarketDepthBacktest([asset])
            
            print("开始事件分析...")
            setup_time = time.time() - start_time
            print(f"设置时间: {setup_time:.2f} 秒")
            
            print("创建分析函数...")
            # 使用njit函数分析数据
            analyze_func = self._create_event_analysis_function()
            
            print("调用分析函数...")
            # Numba函数不能处理None，所以用一个很大的数代替
            max_events_for_numba = self.max_events if self.max_events is not None else 999999999
            print(f"max_events_for_numba: {max_events_for_numba}")
            
            results = analyze_func(hbt, max_events_for_numba)
            
            analysis_time = time.time() - start_time - setup_time
            print(f"分析时间: {analysis_time:.2f} 秒")
            
            print("关闭回测环境...")
            hbt.close()
            
            print("处理分析结果...")
            # 处理结果
            self._process_analysis_results(results)
            
            total_time = time.time() - start_time
            print(f"总时间: {total_time:.2f} 秒")
            
            return results
        
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_event_analysis_function(self):
        """创建事件分析的numba函数"""
        
        @njit
        def analyze_events(hbt, max_events):
            asset_no = 0
            
            print("初始化分析变量...")
            
            # 统计计数器
            total_events = 0
            depth_events = 0
            trade_events = 0
            valid_events = 0  # 有效的bid/ask事件
            
            # 价格和时间数据
            timestamps = []
            best_bids = []
            best_asks = []
            mid_prices = []
            spreads = []
            
            # 时间间隔统计
            prev_timestamp = 0
            time_intervals = []
            
            # 价格变化统计
            prev_mid_price = 0.0
            price_changes = []
            
            # 交易量统计
            trade_volumes = []
            trade_prices = []
            
            print("开始逐事件分析...")
            last_print_time = 0
            
            while True:
                # 等待下一个市场数据事件
                result = hbt.wait_next_feed(
                    include_order_resp=False,
                    timeout=10000000000  # 10秒超时，避免无限等待
                )
                
                current_time = hbt.current_timestamp
                
                # 检查超时
                if result == 0:
                    print("等待超时，当前时间:", current_time)
                    if total_events == 0:
                        print("警告: 没有收到任何事件，可能数据有问题")
                        break
                    continue
                elif result == 1:  # 数据结束
                    print("数据结束，当前时间:", current_time)
                    break
                elif result == 2:  # 收到市场数据feed
                    depth = hbt.depth(asset_no)
                    
                    # 基本统计
                    total_events += 1
                    
                    # 进度打印（每10000个事件或每5秒）
                    if (total_events % 10000 == 0) or (current_time - last_print_time > 5000000000):  # 5秒
                        print("已处理", total_events, "个事件, 当前时间:", current_time)
                        print("  有效事件:", valid_events, ", 交易事件:", trade_events)
                        if len(mid_prices) > 0:
                            print("  当前价格: $", mid_prices[-1])
                        last_print_time = current_time
                    
                    # 检查是否达到最大事件数（999999999表示无限制）
                    if max_events < 999999999 and total_events >= max_events:
                        print("达到最大事件数限制:", max_events)
                        break
                    
                    # 检查是否有有效的bid/ask
                    if depth.best_bid > 0 and depth.best_ask > 0:
                        valid_events += 1
                        
                        mid_price = (depth.best_bid + depth.best_ask) / 2.0
                        spread = depth.best_ask - depth.best_bid
                        
                        # 记录数据
                        timestamps.append(current_time)
                        best_bids.append(depth.best_bid)
                        best_asks.append(depth.best_ask)
                        mid_prices.append(mid_price)
                        spreads.append(spread)
                        
                        # 计算时间间隔
                        if prev_timestamp > 0:
                            interval = current_time - prev_timestamp
                            time_intervals.append(interval)
                        
                        # 计算价格变化
                        if prev_mid_price > 0:
                            price_change = (mid_price - prev_mid_price) / prev_mid_price
                            price_changes.append(price_change)
                        
                        prev_timestamp = current_time
                        prev_mid_price = mid_price
                    else:
                        if total_events % 50000 == 0:  # 减少无效事件的打印频率
                            print("  无效bid/ask: bid=", depth.best_bid, ", ask=", depth.best_ask)
                    
                    # 获取最新交易数据
                    trades = hbt.last_trades(asset_no)
                    for trade in trades:
                        trade_events += 1
                        trade_volumes.append(trade.qty)
                        trade_prices.append(trade.px)
                    
                    # 清理交易数据
                    hbt.clear_last_trades(asset_no)
                    
                else:
                    print("未知返回值:", result)
            
            print("分析完成！")
            print("  总事件数:", total_events)
            print("  有效事件数:", valid_events)
            print("  交易事件数:", trade_events)
            print("  价格数据点:", len(mid_prices))
            print("  时间间隔数据点:", len(time_intervals))
            
            # 转换为numpy数组用于返回
            return (
                total_events,
                valid_events, 
                trade_events,
                np.array(timestamps),
                np.array(best_bids),
                np.array(best_asks), 
                np.array(mid_prices),
                np.array(spreads),
                np.array(time_intervals),
                np.array(price_changes),
                np.array(trade_volumes),
                np.array(trade_prices)
            )
        
        return analyze_events
    
    def _process_analysis_results(self, results):
        """处理分析结果"""
        (total_events, valid_events, trade_events, timestamps, best_bids, 
         best_asks, mid_prices, spreads, time_intervals, price_changes,
         trade_volumes, trade_prices) = results
        
        print("计算统计指标...")
        
        # 存储基本统计
        self.stats = {
            'total_events': total_events,
            'valid_events': valid_events,
            'trade_events': trade_events,
            'unique_timestamps': len(np.unique(timestamps)) if len(timestamps) > 0 else 0
        }
        
        # 时间间隔统计（转换为毫秒）
        if len(time_intervals) > 0:
            print("计算时间间隔统计...")
            time_intervals_ms = time_intervals / 1_000_000  # 纳秒转毫秒
            self.stats['time_intervals'] = {
                'mean_ms': np.mean(time_intervals_ms),
                'median_ms': np.median(time_intervals_ms),
                'std_ms': np.std(time_intervals_ms),
                'min_ms': np.min(time_intervals_ms),
                'max_ms': np.max(time_intervals_ms),
                'percentiles': {
                    '1%': np.percentile(time_intervals_ms, 1),
                    '5%': np.percentile(time_intervals_ms, 5),
                    '25%': np.percentile(time_intervals_ms, 25),
                    '75%': np.percentile(time_intervals_ms, 75),
                    '95%': np.percentile(time_intervals_ms, 95),
                    '99%': np.percentile(time_intervals_ms, 99),
                }
            }
        
        # 价格统计
        if len(mid_prices) > 0:
            print("计算价格统计...")
            self.stats['prices'] = {
                'mean_price': np.mean(mid_prices),
                'min_price': np.min(mid_prices),
                'max_price': np.max(mid_prices),
                'price_range': np.max(mid_prices) - np.min(mid_prices),
                'price_volatility': np.std(mid_prices)
            }
        
        # 价差统计
        if len(spreads) > 0:
            print("计算价差统计...")
            spreads_bps = spreads / np.mean(mid_prices) * 10000  # 转换为基点
            self.stats['spreads'] = {
                'mean_spread': np.mean(spreads),
                'mean_spread_bps': np.mean(spreads_bps),
                'median_spread_bps': np.median(spreads_bps),
                'min_spread_bps': np.min(spreads_bps),
                'max_spread_bps': np.max(spreads_bps)
            }
        
        # 价格变化统计
        if len(price_changes) > 0:
            print("计算价格变化统计...")
            self.stats['price_changes'] = {
                'mean_change': np.mean(price_changes),
                'std_change': np.std(price_changes),
                'skewness': self._calculate_skewness(price_changes),
                'kurtosis': self._calculate_kurtosis(price_changes)
            }
        
        # 交易统计
        if len(trade_volumes) > 0:
            print("计算交易统计...")
            self.stats['trades'] = {
                'total_trades': len(trade_volumes),
                'mean_volume': np.mean(trade_volumes),
                'median_volume': np.median(trade_volumes),
                'total_volume': np.sum(trade_volumes),
                'mean_trade_price': np.mean(trade_prices)
            }
        
        # 存储原始数据用于绘图
        self.timestamps = timestamps
        self.mid_prices = mid_prices
        self.spreads = spreads
        self.time_intervals = time_intervals
        self.price_changes = price_changes
        
        print("统计计算完成!")
        
    @staticmethod
    @njit
    def _calculate_skewness(data):
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod 
    @njit
    def _calculate_kurtosis(data):
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def print_summary_statistics(self):
        """打印汇总统计信息"""
        print("\n" + "="*60)
        print("DOGE-USDT-PERP 数据探索分析报告")
        print("="*60)
        
        print(f"\n基本统计:")
        print(f"  总事件数: {self.stats['total_events']:,}")
        print(f"  有效事件数: {self.stats['valid_events']:,}")
        print(f"  交易事件数: {self.stats['trade_events']:,}")
        print(f"  唯一时间戳数: {self.stats['unique_timestamps']:,}")
        print(f"  数据利用率: {self.stats['valid_events']/max(1, self.stats['total_events'])*100:.1f}%")
        
        if 'time_intervals' in self.stats:
            ti = self.stats['time_intervals']
            print(f"\n时间间隔统计 (毫秒):")
            print(f"  平均间隔: {ti['mean_ms']:.2f} ms")
            print(f"  中位数间隔: {ti['median_ms']:.2f} ms")
            print(f"  标准差: {ti['std_ms']:.2f} ms")
            print(f"  最小间隔: {ti['min_ms']:.2f} ms")
            print(f"  最大间隔: {ti['max_ms']:.2f} ms")
            print(f"  分位数:")
            for pct, val in ti['percentiles'].items():
                print(f"    {pct}: {val:.2f} ms")
        
        if 'prices' in self.stats:
            prices = self.stats['prices']
            print(f"\n价格统计:")
            print(f"  平均价格: ${prices['mean_price']:.5f}")
            print(f"  价格范围: ${prices['min_price']:.5f} - ${prices['max_price']:.5f}")
            print(f"  价格区间: ${prices['price_range']:.5f}")
            print(f"  价格波动率: ${prices['price_volatility']:.5f}")
        
        if 'spreads' in self.stats:
            spreads = self.stats['spreads']
            print(f"\n价差统计:")
            print(f"  平均价差: ${spreads['mean_spread']:.5f}")
            print(f"  平均价差 (基点): {spreads['mean_spread_bps']:.2f} bps")
            print(f"  中位数价差 (基点): {spreads['median_spread_bps']:.2f} bps")
            print(f"  价差范围 (基点): {spreads['min_spread_bps']:.2f} - {spreads['max_spread_bps']:.2f} bps")
        
        if 'price_changes' in self.stats:
            pc = self.stats['price_changes']
            print(f"\n价格变化统计:")
            print(f"  平均变化: {pc['mean_change']:.6f}")
            print(f"  标准差: {pc['std_change']:.6f}")
            print(f"  偏度: {pc['skewness']:.3f}")
            print(f"  峰度: {pc['kurtosis']:.3f}")
        
        if 'trades' in self.stats:
            trades = self.stats['trades']
            print(f"\n交易统计:")
            print(f"  总交易数: {trades['total_trades']:,}")
            print(f"  平均交易量: {trades['mean_volume']:.2f}")
            print(f"  中位数交易量: {trades['median_volume']:.2f}")
            print(f"  总成交量: {trades['total_volume']:.2f}")
            print(f"  平均交易价格: ${trades['mean_trade_price']:.5f}")
    
    def create_visualizations(self, save_path=None):
        """创建可视化图表"""
        
        print("生成可视化图表...")
        
        if len(self.mid_prices) == 0:
            print("警告: 没有价格数据，跳过可视化")
            return
        
        # 设置中文字体
        # 由于字体问题，直接使用英文标题
        use_english = True
        
        # 设置图表样式
        plt.style.use('default')  # 改为默认样式，避免seaborn版本问题
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 定义标题（中英文版本）
        if use_english:
            titles = {
                'price_series': 'Price Time Series (Sampled)',
                'spread_dist': 'Spread Distribution',
                'interval_dist': 'Event Time Interval Distribution', 
                'change_dist': 'Price Change Distribution',
                'interval_series': 'Time Interval Series (Sampled)',
                'qq_plot': 'Price Change Q-Q Plot (Normal Dist)'
            }
            labels = {
                'time_hours': 'Time (Hours)',
                'price': 'Price ($)',
                'spread_bps': 'Spread (bps)',
                'frequency': 'Frequency',
                'interval_ms': 'Time Interval (ms)',
                'rel_change': 'Relative Price Change',
                'event_num': 'Event Number'
            }
        else:
            titles = {
                'price_series': '价格时间序列 (采样)',
                'spread_dist': '价差分布',
                'interval_dist': '事件时间间隔分布',
                'change_dist': '价格变化分布', 
                'interval_series': '时间间隔时间序列 (采样)',
                'qq_plot': '价格变化 Q-Q 图 (正态分布)'
            }
            labels = {
                'time_hours': '时间 (小时)',
                'price': '价格 ($)',
                'spread_bps': '价差 (基点)',
                'frequency': '频次',
                'interval_ms': '时间间隔 (毫秒)',
                'rel_change': '相对价格变化',
                'event_num': '事件序号'
            }
        
        # 1. 价格时间序列 (采样显示)
        if len(self.mid_prices) > 0:
            # 采样以避免图表过于密集
            step = max(1, len(self.mid_prices) // 10000)
            sample_timestamps = self.timestamps[::step]
            sample_prices = self.mid_prices[::step]
            
            # 转换时间戳为小时
            start_time = sample_timestamps[0]
            hours = (sample_timestamps - start_time) / (3600 * 1_000_000_000)
            
            axes[0, 0].plot(hours, sample_prices, linewidth=0.5, alpha=0.8)
            axes[0, 0].set_title(titles['price_series'])
            axes[0, 0].set_xlabel(labels['time_hours'])
            axes[0, 0].set_ylabel(labels['price'])
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 价差分布
        if len(self.spreads) > 0:
            spreads_bps = self.spreads / np.mean(self.mid_prices) * 10000
            axes[0, 1].hist(spreads_bps, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title(titles['spread_dist'])
            axes[0, 1].set_xlabel(labels['spread_bps'])
            axes[0, 1].set_ylabel(labels['frequency'])
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 时间间隔分布
        if len(self.time_intervals) > 0:
            intervals_ms = self.time_intervals / 1_000_000
            # 限制在合理范围内显示
            intervals_clipped = np.clip(intervals_ms, 0, np.percentile(intervals_ms, 99))
            axes[1, 0].hist(intervals_clipped, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title(titles['interval_dist'])
            axes[1, 0].set_xlabel(labels['interval_ms'])
            axes[1, 0].set_ylabel(labels['frequency'])
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 价格变化分布
        if len(self.price_changes) > 0:
            # 限制在合理范围内显示
            changes_clipped = np.clip(self.price_changes, 
                                    np.percentile(self.price_changes, 0.1),
                                    np.percentile(self.price_changes, 99.9))
            axes[1, 1].hist(changes_clipped, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title(titles['change_dist'])
            axes[1, 1].set_xlabel(labels['rel_change'])
            axes[1, 1].set_ylabel(labels['frequency'])
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 时间间隔时间序列（采样）
        if len(self.time_intervals) > 0:
            step = max(1, len(self.time_intervals) // 5000)
            sample_intervals = self.time_intervals[::step] / 1_000_000
            axes[2, 0].plot(sample_intervals, linewidth=0.5, alpha=0.8)
            axes[2, 0].set_title(titles['interval_series'])
            axes[2, 0].set_xlabel(labels['event_num'])
            axes[2, 0].set_ylabel(labels['interval_ms'])
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 价格变化Q-Q图
        if len(self.price_changes) > 0:
            try:
                from scipy import stats
                # 采样以避免计算过慢
                sample_size = min(10000, len(self.price_changes))
                sample_changes = np.random.choice(self.price_changes, sample_size, replace=False)
                stats.probplot(sample_changes, dist="norm", plot=axes[2, 1])
                axes[2, 1].set_title(titles['qq_plot'])
                axes[2, 1].grid(True, alpha=0.3)
            except ImportError:
                axes[2, 1].text(0.5, 0.5, 'scipy not available\nCannot generate Q-Q plot', 
                               ha='center', va='center', transform=axes[2, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_eda_analysis.png", dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {save_path}_eda_analysis.png")
        plt.show()
    
    def save_analysis_results(self, filepath):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存统计数据
        with open(f"{filepath}_stats_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.stats, f)
        
        # 保存为CSV
        stats_df = pd.json_normalize(self.stats, sep='_')
        stats_df.to_csv(f"{filepath}_stats_{timestamp}.csv", index=False)
        
        print(f"分析结果已保存到 {filepath}_stats_{timestamp}.*")

def run_eda_analysis(max_events=None):  # 默认处理所有数据
    """运行完整的EDA分析"""
    if max_events is None:
        print("开始运行EDA分析（处理所有数据）...")
    else:
        print(f"开始运行EDA分析（限制处理事件数: {max_events:,}）...")
    
    explorer = DataExplorer(max_events=max_events)
    
    # 分析数据
    try:
        results = explorer.analyze_event_driven_data()
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return None
    
    # 打印统计信息
    explorer.print_summary_statistics()
    
    # 创建可视化
    print("\n生成可视化图表...")
    try:
        explorer.create_visualizations(save_path="doge_usdt_perp_eda")
    except Exception as e:
        print(f"生成图表时出现错误: {e}")
    
    # 保存结果
    try:
        explorer.save_analysis_results("doge_usdt_perp_analysis")
    except Exception as e:
        print(f"保存结果时出现错误: {e}")
    
    print("\nEDA分析完成！")
    return explorer

if __name__ == "__main__":
    # 运行EDA分析，处理所有数据
    explorer = run_eda_analysis()  # 处理所有数据
