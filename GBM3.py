'''
The use of Generative AI was used in the production of the code, especially for readabilty and
style improments (e.g. type hints, formatting, and docstrings)
'''

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from typing import Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class GBMConfig:
    """Configuration parameters for GBM simulation"""
    ticker: str = 'SPY'
    start_date: str = '2024-01-01'
    end_date: str = '2025-01-01'
    rolling_window: int = 21
    confidence_level: float = 90.0
    n_simulation_paths: int = 10000
    trading_days_per_year: int = 252
    use_simple_rolling: bool = False
    sigma_scalar: float = 1                #A very quick and dirty fix to sigma over estimation

def fetch_stock_data(config: GBMConfig) -> pd.Series:
    """
    Fetch stock price data from Yahoo Finance
    
    Args:
        config: GBM configuration parameters
        
    Returns:
        Series of closing prices
    """
    data = yf.download(config.ticker, start=config.start_date, end=config.end_date)
    return data['Close']

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series
    
    Args:
        prices: Series of stock prices
        
    Returns:
        Series of log returns
    """
    return np.log(prices / prices.shift(1)).dropna()

def calculate_static_parameters(log_returns: pd.Series, trading_days: int) -> Tuple[float, float]:
    """
    Calculate static drift and volatility parameters
    
    Args:
        log_returns: Series of log returns
        trading_days: Number of trading days per year for annualization
        
    Returns:
        Tuple of (annualized_drift, annualized_volatility)
    """
    mu = float(log_returns.mean()) * trading_days
    sigma = float(log_returns.std()) * np.sqrt(trading_days)
    return mu, sigma

def simulate_gbm_paths(config: GBMConfig, 
                      initial_price: float, 
                      log_returns: pd.Series, 
                      mu_static: float, 
                      sigma_static: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate GBM paths with rolling parameter estimation
    
    Args:
        config: GBM configuration parameters
        initial_price: Starting stock price
        log_returns: Historical log returns
        mu_static: Static drift parameter (fallback)
        sigma_static: Static volatility parameter (fallback)
        
    Returns:
        Tuple of (simulated_paths, mu_parameters_used, sigma_parameters_used)
    """
    N = len(log_returns) + 1  # +1 for initial price
    T = N / config.trading_days_per_year
    dt = T / N
    
    # Initialize arrays
    S_sim = np.zeros((config.n_simulation_paths, N))
    S_sim[:, 0] = initial_price
    mu_used = np.zeros(N)
    sigma_used = np.zeros(N)
    
    for t in range(1, N):
        # Calculate rolling parameters using only past data (no look-ahead bias)
        if t >= config.rolling_window:
            past_returns = log_returns.iloc[t-config.rolling_window:t]

            if config.use_simple_rolling:
                mu_t = float(past_returns.mean() * config.trading_days_per_year)
                sigma_t = float(past_returns.std() * np.sqrt(config.trading_days_per_year))
            else:
                mu_t = float(past_returns.ewm(span=config.rolling_window).mean().iloc[-1] * config.trading_days_per_year)
                sigma_t = float(past_returns.ewm(span=config.rolling_window).std().iloc[-1] * np.sqrt(config.trading_days_per_year))


        else:
            # Use static parameters when insufficient data
            mu_t = mu_static
            sigma_t = sigma_static*config.sigma_scalar
        
        # Handle NaN values
        if np.isnan(mu_t) or np.isnan(sigma_t):
            mu_t = mu_static
            sigma_t = sigma_static

        sigma_t *= config.sigma_scalar

        # Store parameters for analysis
        mu_used[t] = mu_t
        sigma_used[t] = sigma_t
        
        # Generate random shocks
        Z = np.random.standard_normal(config.n_simulation_paths)
        
        # GBM simulation step
        S_sim[:, t] = S_sim[:, t - 1] * np.exp((mu_t - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z)
    
    return S_sim, mu_used, sigma_used

def calculate_confidence_intervals(simulated_paths: np.ndarray, confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals from simulated paths
    
    Args:
        simulated_paths: Array of simulated price paths
        confidence_level: Confidence level (e.g., 90.0 for 90%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = (100 - confidence_level) / 2
    lower = np.percentile(simulated_paths, alpha, axis=0)
    upper = np.percentile(simulated_paths, 100 - alpha, axis=0)
    return lower, upper

def calculate_empirical_coverage(actual_prices: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> Dict[str, float]:
    """
    Calculate empirical coverage statistics
    
    Args:
        actual_prices: Array of actual stock prices
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        
    Returns:
        Dictionary with coverage statistics
    """
    within_bounds = (actual_prices >= lower_bound) & (actual_prices <= upper_bound)
    N = len(within_bounds)
    
    return {
        'overall_coverage': within_bounds.mean() * 100,
        'first_half_coverage': within_bounds[:N//2].mean() * 100,
        'second_half_coverage': within_bounds[N//2:].mean() * 100,
        'within_bounds_array': within_bounds
    }

def calculate_performance_metrics(actual_prices: np.ndarray, mean_simulation: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics
    
    Args:
        actual_prices: Array of actual stock prices
        mean_simulation: Mean of simulated paths
        
    Returns:
        Dictionary with performance metrics
    """
    rmse = root_mean_squared_error(actual_prices, mean_simulation)
    
    # Naive benchmark (random walk)
    price_changes = np.diff(actual_prices)
    naive_rmse = np.sqrt(np.mean(price_changes**2))
    
    
    return {
        'rmse': rmse,
        'naive_rmse': naive_rmse,
        'rmse_vs_mean_price_pct': (rmse / actual_prices.mean()) * 100,
    }

def print_model_diagnostics(config: GBMConfig, performance_metrics: Dict[str, float], coverage_stats: Dict[str, float], mu_static: float,
                          sigma_static: float, mu_used: np.ndarray, sigma_used: np.ndarray) -> None:
    """
    Print comprehensive model diagnostics
    """
    print("="*60)
    print(f"GBM SIMULATION RESULTS FOR {config.ticker}")
    print("="*60)
    
    print(f"\nCONFIGURATION:")
    print(f"  Rolling Window: {config.rolling_window} days")
    print(f"  Confidence Level: {config.confidence_level}%")
    print(f"  Simulation Paths: {config.n_simulation_paths:,}")
    print(f"  Period: {config.start_date} to {config.end_date}")
    
    print(f"\nPARAMETER ESTIMATES:")
    print(f"  Static Drift (μ): {mu_static:.3f} per year")
    print(f"  Static Volatility (σ): {sigma_static:.3f} per year")
    print(f"  Rolling μ range: [{mu_used[mu_used>0].min():.3f}, {mu_used.max():.3f}]")
    print(f"  Rolling σ range: [{sigma_used[sigma_used>0].min():.3f}, {sigma_used.max():.3f}]")
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"  RMSE: ${performance_metrics['rmse']:.2f}")
    print(f"  RMSE vs Mean Price: {performance_metrics['rmse_vs_mean_price_pct']:.1f}%")
    print(f"  Naive Benchmark RMSE: ${performance_metrics['naive_rmse']:.2f}")
    
    print(f"\nCOVERAGE ANALYSIS:")
    print(f"  Volatility Scalar: {config.sigma_scalar}")
    print(f"  Overall Coverage: {coverage_stats['overall_coverage']:.1f}% at CI of {config.confidence_level}")
    print(f"  First Half Coverage: {coverage_stats['first_half_coverage']:.1f}%")
    print(f"  Second Half Coverage: {coverage_stats['second_half_coverage']:.1f}%")

def create_comprehensive_plots(config: GBMConfig, dates: pd.DatetimeIndex, actual_prices: np.ndarray, mean_simulation: np.ndarray,
                             lower_bound: np.ndarray, upper_bound: np.ndarray,  coverage_stats: Dict[str, float],
                             mu_used: np.ndarray, sigma_used: np.ndarray, mu_static: float, sigma_static: float) -> None:
    """
    Create comprehensive visualization plots
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2,2,height_ratios=[1,1])
    #gs = gridspec.GridSpec(1,1)

    # Main price comparison plot
    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(dates, actual_prices, label=f'Actual {config.ticker} Price', 
             color='black', linewidth=2)
    ax1.plot(dates, mean_simulation, linestyle='--', 
             label=f"Mean of {config.n_simulation_paths:,} Paths", alpha=0.8)
    ax1.fill_between(dates, lower_bound, upper_bound, 
                     color='grey', alpha=0.3, label=f"{config.confidence_level}% CI\nEmpirical Coverage: {coverage_stats['overall_coverage']:.1f}%")
    #ax1.annotate(f"Empirical Coverage: {coverage_stats['overall_coverage']:.1f}%", (0.05, 0.95), xycoords='axes fraction')
    ax1.set_title(f"Real vs Simulated GBM Stock Prices for Ticker {config.ticker}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    txt_box = AnchoredText(f"Rolling Window Size: {config.rolling_window}\nNum of Simulations: {config.n_simulation_paths}\nTrading Days per Year: {config.trading_days_per_year}\nSigma Scaled: {config.sigma_scalar}",
                           loc='upper left', prop=dict(size=12), frameon=True, alpha = 0.2)
    txt_box.patch.set_facecolor('white')
    txt_box.patch.set_edgecolor('black')
    ax1.add_artist(txt_box)

    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #Parameter evolution - Drift
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(dates, mu_used, label='Rolling μ', alpha=0.7, color='green')
    ax2.axhline(y=mu_static, color='red', linestyle='--', 
                alpha=0.7, label='Static μ')
    ax2.set_title("Evolution of Drift Parameter")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Annual Drift")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter evolution - Volatility
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(dates, sigma_used, label='Rolling σ', alpha=0.7, color='orange')
    ax3.axhline(y=sigma_static*config.sigma_scalar, color='red', linestyle='--', 
                alpha=0.7, label='Static σ')
    ax3.set_title("Evolution of Volatility Parameter")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Annual Volatility")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_gbm_simulation(config: GBMConfig = None) -> Dict[str, Any]:
    """
    Main function to run complete GBM simulation analysis
    
    Args:
        config: GBM configuration parameters (uses default if None)
        
    Returns:
        Dictionary containing all results and analysis
    """
    if config is None:
        config = GBMConfig()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Fetch and prepare data
    print(f"Fetching {config.ticker} data from {config.start_date} to {config.end_date}...")
    real_prices = fetch_stock_data(config)
    real_prices_1d = real_prices.values.flatten()
    log_returns = calculate_log_returns(real_prices)
    
    # 2. Calculate static parameters
    mu_static, sigma_static = calculate_static_parameters(log_returns, config.trading_days_per_year) 
    
    # 3. Run simulation
    print(f"Running {config.n_simulation_paths:,} GBM simulations...")
    S_sim, mu_used, sigma_used = simulate_gbm_paths(
        config, float(real_prices.iloc[0]), log_returns, mu_static, sigma_static
    )
    
    # 4. Calculate results
    mean_simulation = S_sim.mean(axis=0)
    lower_bound, upper_bound = calculate_confidence_intervals(S_sim, config.confidence_level)
    
    # 5. Analysis
    performance_metrics = calculate_performance_metrics(real_prices_1d, mean_simulation)
    coverage_stats = calculate_empirical_coverage(real_prices_1d, lower_bound, upper_bound)
    
    # 6. Display results
    print_model_diagnostics(config, performance_metrics, coverage_stats, 
                          mu_static, sigma_static, mu_used, sigma_used)
    
    # 7. Create plots
    create_comprehensive_plots(config, real_prices.index, real_prices_1d, 
                             mean_simulation, lower_bound, upper_bound,
                             coverage_stats, mu_used, sigma_used, 
                             mu_static, sigma_static)
    
    # 8. Return all results
    return {
        'config': config,
        'real_prices': real_prices_1d,
        'simulated_paths': S_sim,
        'mean_simulation': mean_simulation,
        'confidence_intervals': (lower_bound, upper_bound),
        'performance_metrics': performance_metrics,
        'coverage_stats': coverage_stats,
        'parameters': {
            'mu_static': mu_static,
            'sigma_static': sigma_static,
            'mu_used': mu_used,
            'sigma_used': sigma_used
        }
    }

# Example usage with different configurations
if __name__ == "__main__":
    # Default configuration
    print("Running with default configuration...")
    results = run_gbm_simulation(GBMConfig(sigma_scalar=0.3))
    
    # Custom configuration example
    print("\n" + "="*60)
    print("Running with custom configuration...")
    custom_config = GBMConfig(
        ticker='SPY',
        rolling_window=21,
        confidence_level=90.0,
        start_date='2010-01-01',
        end_date='2015-12-31'
    )
    results = run_gbm_simulation(custom_config)
