"""
Institutional Financial Backtesting and Portfolio Simulation Engine.
Evaluates the trading efficacy of the Explainable Multi-Modal Model against
a standard Buy & Hold market benchmark.
Computes:
- Cumulative Return (%)
- Annualized Return (CAGR %)
- Annualized Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown (MDD %)
- Win Rate (%) & Profit Factor
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


class PortfolioBacktester:
    """
    Simulates portfolio execution using model signals:
    - Long signal: when model predicted return > 0 and confidence > threshold
    - Cash/Neutral: when signal is bearish or below confidence threshold
    """
    def __init__(
        self,
        risk_free_rate: float = 0.04, # 4% annual risk free rate
        transaction_cost: float = 0.0005, # 5 bps transaction cost
        confidence_threshold: float = 0.52
    ):
        self.rf = risk_free_rate / 252 # Daily risk-free rate
        self.fee = transaction_cost
        self.conf_thresh = confidence_threshold

    def run_backtest(
        self,
        returns_actual: np.ndarray,
        predictions_return: np.ndarray,
        predictions_direction_prob: np.ndarray,
        dates: Optional[List] = None
    ) -> Dict:
        """
        Executes backtest over test horizon.
        Args:
            returns_actual: (T,) actual asset returns
            predictions_return: (T,) model predicted returns
            predictions_direction_prob: (T,) probability of upward trend
            dates: list of trading dates
        """
        n = len(returns_actual)
        signals = np.zeros(n)
        
        # Position logic: 1.0 (Long) or 0.0 (Cash)
        for t in range(n):
            if predictions_return[t] > 0 and predictions_direction_prob[t] >= self.conf_thresh:
                signals[t] = 1.0
            else:
                signals[t] = 0.0

        # Calculate position shifts for transaction costs
        trades = np.abs(np.diff(signals, prepend=0))
        fees = trades * self.fee
        
        # Strategy daily returns
        strat_daily_returns = (signals * returns_actual) - fees
        
        # Cumulative wealth growth (starting from $10,000)
        initial_capital = 10000.0
        strat_equity = initial_capital * np.cumprod(1.0 + strat_daily_returns)
        benchmark_equity = initial_capital * np.cumprod(1.0 + returns_actual)
        
        # Metrics calculation
        strat_cum_return = ((strat_equity[-1] - initial_capital) / initial_capital) * 100.0
        bench_cum_return = ((benchmark_equity[-1] - initial_capital) / initial_capital) * 100.0
        
        # Sharpe Ratio
        excess_returns = strat_daily_returns - self.rf
        std_returns = np.std(strat_daily_returns) + 1e-9
        sharpe_ratio = float((np.mean(excess_returns) / std_returns) * np.sqrt(252))
        
        # Sortino Ratio (downside risk only)
        downside_returns = strat_daily_returns[strat_daily_returns < 0]
        downside_std = np.std(downside_returns) + 1e-9 if len(downside_returns) > 0 else 1e-6
        sortino_ratio = float((np.mean(excess_returns) / downside_std) * np.sqrt(252))
        
        # Maximum Drawdown (MDD)
        peak = np.maximum.accumulate(strat_equity)
        drawdowns = (strat_equity - peak) / peak
        max_drawdown = float(np.min(drawdowns) * 100.0)
        
        # Win Rate & Profit Factor
        trade_returns = strat_daily_returns[signals == 1.0]
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        win_rate = (len(wins) / len(trade_returns) * 100.0) if len(trade_returns) > 0 else 50.0
        gross_profit = np.sum(wins) if len(wins) > 0 else 1e-6
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-6
        profit_factor = float(gross_profit / gross_loss)
        
        df_results = pd.DataFrame({
            "Date": dates if dates is not None else list(range(n)),
            "Actual_Return": returns_actual,
            "Strategy_Return": strat_daily_returns,
            "Strategy_Equity": strat_equity,
            "Benchmark_Equity": benchmark_equity,
            "Position": signals
        })
        
        metrics = {
            "Initial Capital ($)": initial_capital,
            "Final Strategy Equity ($)": round(strat_equity[-1], 2),
            "Strategy Cumulative Return (%)": round(strat_cum_return, 2),
            "Benchmark Cumulative Return (%)": round(bench_cum_return, 2),
            "Excess Return over Market (%)": round(strat_cum_return - bench_cum_return, 2),
            "Annualized Sharpe Ratio": round(sharpe_ratio, 2),
            "Sortino Ratio": round(sortino_ratio, 2),
            "Maximum Drawdown (%)": round(max_drawdown, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2)
        }
        
        return {
            "metrics": metrics,
            "equity_df": df_results
        }


def plot_backtest_performance(
    backtest_data: Dict,
    ticker: str = "Portfolio",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Renders institutional performance chart comparing Model Strategy vs. Buy & Hold.
    """
    df = backtest_data["equity_df"]
    metrics = backtest_data["metrics"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    fig.patch.set_facecolor("#ffffff")
    
    x = range(len(df))
    # Top Plot: Equity Curves
    ax1.plot(x, df["Strategy_Equity"], label=f"Proposed Multi-Modal DGT ({metrics['Strategy Cumulative Return (%)']:+.1f}%)", color="#2563eb", linewidth=2.0)
    ax1.plot(x, df["Benchmark_Equity"], label=f"Buy & Hold Market Benchmark ({metrics['Benchmark Cumulative Return (%)']:+.1f}%)", color="#64748b", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("Portfolio Value ($)", fontweight="bold")
    ax1.set_title(f"Out-of-Sample Backtesting Performance: {ticker} (Sharpe: {metrics['Annualized Sharpe Ratio']})", fontweight="bold", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Bottom Plot: Drawdown
    peak = np.maximum.accumulate(df["Strategy_Equity"])
    dd = ((df["Strategy_Equity"] - peak) / peak) * 100.0
    ax2.fill_between(x, dd, 0, color="#ef4444", alpha=0.3, label=f"Strategy Drawdown (Max: {metrics['Maximum Drawdown (%)']:.1f}%)")
    ax2.set_ylabel("Drawdown %", fontweight="bold")
    ax2.set_xlabel("Trading Days (Out-of-Sample Test Set)", fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
