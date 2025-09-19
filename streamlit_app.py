import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime
from jugaad_data.nse import stock_df
import random
import time
import logging
import sys
from io import StringIO
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected


# Configure logging
@st.cache_resource
def setup_logging():
    """Setup logging configuration"""
    # Create a logger
    logger = logging.getLogger('portfolio_simulation')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    file_handler = logging.FileHandler('portfolio_simulation.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()


# Page configuration
st.set_page_config(
    page_title="Portfolio Simulation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4a5568;
        color: white;
    }
    .stMetric {
        background-color: #2d3748 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #4a5568 !important;
        color: white !important;
    }
    .stMetric > div {
        color: white !important;
    }
    .stMetric label {
        color: #e2e8f0 !important;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        color: white !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: white !important;
    }
    .stMetric [data-testid="metric-container"] label {
        color: #cbd5e0 !important;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #a0aec0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_stock_list(csv_file):
    """Load and cache the stock list from CSV"""
    logger.info(f"Loading stock list from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        stock_count = len(df)
        logger.info(f"Successfully loaded {stock_count} stocks from CSV file")
        return df[["Symbol"]].copy()
    except Exception as e:
        logger.error(f"Error loading stock list from {csv_file}: {e}")
        st.error(f"Error loading stock list: {e}")
        return None


def fetch_stock_data_with_retry(symbol, start_date, end_date, max_retries=3, retry_delay=1):
    """
    Fetch stock data with retry logic to handle connection issues.
    
    Args:
        symbol (str): Stock symbol
        start_date (date): Start date
        end_date (date): End date
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    
    Returns:
        DataFrame or None: Stock data or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to fetch {symbol} data (attempt {attempt + 1}/{max_retries})")
            df = stock_df(symbol, start_date, end_date)
            
            if df is not None and len(df) > 0:
                logger.debug(f"Successfully fetched {symbol} data: {len(df)} records")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except (ConnectionError, RemoteDisconnected, ProtocolError) as e:
            logger.warning(f"Connection error fetching {symbol} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1)  # Exponential backoff
                logger.info(f"Retrying {symbol} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All retry attempts failed for {symbol}")
                return None
                
        except Timeout as e:
            logger.warning(f"Timeout error fetching {symbol} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1)
                logger.info(f"Retrying {symbol} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Timeout: All retry attempts failed for {symbol}")
                return None
                
        except RequestException as e:
            logger.warning(f"Request error fetching {symbol} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = retry_delay * (attempt + 1)
                logger.info(f"Retrying {symbol} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Request error: All retry attempts failed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return None
    
    return None


def run_portfolio_simulation(from_date, to_date, num_stocks=5, initial_capital=1000000, 
                           csv_file="ind_niftytotalmarket_list.csv", verbose=False):
    """
    Runs a single portfolio simulation with monthly rebalancing strategy.
    Buys stocks on the first trading day of each month and sells on the last trading day.
    Capital rolls over to the next month.
    """
    import calendar
    from datetime import datetime, timedelta
    
    simulation_start_time = time.time()
    logger.info(f"Starting portfolio simulation: {from_date} to {to_date}, {num_stocks} stocks, ₹{initial_capital:,} capital")
    
    # Load stock list
    ind_niftytotalmarket_list = load_stock_list(csv_file)
    if ind_niftytotalmarket_list is None:
        logger.error("Failed to load stock list")
        return None
    
    # Generate monthly periods
    monthly_periods = get_monthly_periods(from_date, to_date)
    if not monthly_periods:
        logger.error("No monthly periods generated")
        return None
    
    logger.info(f"Generated {len(monthly_periods)} monthly periods for simulation")
    
    current_capital = initial_capital
    monthly_returns = []
    monthly_details = []
    total_trades = 0
    total_api_calls = 0
    failed_api_calls = 0
    
    for period_idx, (month_start, month_end) in enumerate(monthly_periods):
        month_start_time = time.time()
        logger.info(f"Processing month {period_idx + 1}/{len(monthly_periods)}: {month_start} to {month_end}")
        
        if verbose:
            st.write(f"Processing month {period_idx + 1}: {month_start} to {month_end}")
        
        # Select random stocks for this month
        stock_list_copy = ind_niftytotalmarket_list.copy()
        selected_stocks = []
        
        while len(selected_stocks) < num_stocks and len(stock_list_copy) > 0:
            die_roll = random.randint(0, len(stock_list_copy) - 1)
            selected_stocks.append(stock_list_copy.iloc[die_roll]["Symbol"])
            stock_list_copy = stock_list_copy.drop(die_roll).reset_index(drop=True)
        
        logger.info(f"Selected stocks for month {period_idx + 1}: {selected_stocks}")
        
        # Get stock data for this month and filter out problematic stocks
        valid_stocks = {}
        failed_stocks = []
        
        for stock in selected_stocks:
            total_api_calls += 1
            try:
                logger.debug(f"Fetching data for {stock} from {month_start} to {month_end}")
                
                # Get retry settings from session state (with defaults)
                max_retries = getattr(st.session_state, 'max_retries', 3)
                retry_delay = getattr(st.session_state, 'retry_delay', 2)
                
                df = fetch_stock_data_with_retry(stock, month_start, month_end, 
                                               max_retries=max_retries, retry_delay=retry_delay)
                
                if df is not None and len(df) > 0:
                    # Ensure we have both buy and sell prices
                    if not df['OPEN'].isna().all() and not df['CLOSE'].isna().all():
                        valid_stocks[stock] = df
                        logger.debug(f"Successfully fetched {stock} data ({len(df)} records)")
                    else:
                        failed_stocks.append(stock)
                        failed_api_calls += 1
                        logger.warning(f"{stock} has invalid price data (all NaN values)")
                        if verbose:
                            st.warning(f"{stock} has invalid price data")
                else:
                    failed_stocks.append(stock)
                    failed_api_calls += 1
                    logger.warning(f"{stock} has no data for {month_start} to {month_end}")
                    if verbose:
                        st.warning(f"{stock} has no data for {month_start} to {month_end}")
                        
            except Exception as e:
                failed_stocks.append(stock)
                failed_api_calls += 1
                logger.error(f"Unexpected error with {stock}: {str(e)}")
                if verbose:
                    st.warning(f"Error fetching {stock}: {str(e)}")
        
        logger.info(f"Month {period_idx + 1}: {len(valid_stocks)} valid stocks, {len(failed_stocks)} failed stocks")
        
        # If no valid stocks found, carry forward the capital
        if len(valid_stocks) == 0:
            logger.warning(f"No valid stocks for month {period_idx + 1}, carrying forward capital")
            monthly_returns.append(0)
            monthly_details.append({
                'month': f"{month_start.strftime('%Y-%m')}",
                'stocks': [],
                'start_capital': current_capital,
                'end_capital': current_capital,
                'return': 0
            })
            continue
        
        # Calculate allocation per stock
        allocation_per_stock = current_capital / len(valid_stocks)
        logger.info(f"Month {period_idx + 1}: ₹{allocation_per_stock:,.2f} allocated per stock")
        
        month_end_capital = 0
        month_stock_returns = {}
        
        for stock, df in valid_stocks.items():
            # Buy at first available open price
            buy_price = df.iloc[0]['OPEN']
            # Sell at last available close price
            sell_price = df.iloc[-1]['CLOSE']
            
            if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                # If price data is invalid, allocate the money as cash (no return)
                month_end_capital += allocation_per_stock
                month_stock_returns[stock] = 0
                logger.warning(f"Invalid price data for {stock}, treating as cash")
                continue
            
            # Calculate shares and final value
            shares_bought = allocation_per_stock / buy_price
            final_value = shares_bought * sell_price
            month_end_capital += final_value
            
            # Calculate stock return for this month
            stock_return = (sell_price - buy_price) / buy_price * 100
            month_stock_returns[stock] = stock_return
            total_trades += 1
            
            logger.debug(f"{stock}: Buy ₹{buy_price:.2f}, Sell ₹{sell_price:.2f}, Return {stock_return:.2f}%")
        
        # Calculate monthly return
        month_return = (month_end_capital - current_capital) / current_capital * 100 if current_capital > 0 else 0
        monthly_returns.append(month_return)
        
        month_process_time = time.time() - month_start_time
        logger.info(f"Month {period_idx + 1} completed in {month_process_time:.2f}s: Return {month_return:.2f}%, Capital ₹{month_end_capital:,.2f}")
        
        monthly_details.append({
            'month': f"{month_start.strftime('%Y-%m')}",
            'stocks': list(valid_stocks.keys()),
            'start_capital': current_capital,
            'end_capital': month_end_capital,
            'return': month_return,
            'stock_returns': month_stock_returns
        })
        
        # Update capital for next month
        current_capital = month_end_capital
        
        if verbose:
            st.write(f"Month {period_idx + 1} return: {month_return:.2f}%, Capital: ₹{current_capital:,.2f}")
    
    # Calculate overall statistics
    total_return = (current_capital - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
    
    # Calculate additional metrics
    avg_monthly_return = np.mean(monthly_returns) if monthly_returns else 0
    monthly_volatility = np.std(monthly_returns) if len(monthly_returns) > 1 else 0
    winning_months = len([r for r in monthly_returns if r > 0])
    total_months = len(monthly_returns)
    win_rate_monthly = (winning_months / total_months * 100) if total_months > 0 else 0
    
    simulation_time = time.time() - simulation_start_time
    api_success_rate = ((total_api_calls - failed_api_calls) / total_api_calls * 100) if total_api_calls > 0 else 0
    
    logger.info(f"Simulation completed in {simulation_time:.2f}s: Total return {total_return:.2f}%, {total_trades} trades executed")
    logger.info(f"API Statistics: {total_api_calls} calls, {failed_api_calls} failures, {api_success_rate:.1f}% success rate")
    
    return {
        'selected_stocks': list(set([stock for detail in monthly_details for stock in detail['stocks']])),
        'initial_capital': initial_capital,
        'final_capital': current_capital,
        'total_return': total_return,
        'monthly_returns': monthly_returns,
        'monthly_details': monthly_details,
        'from_date': from_date,
        'to_date': to_date,
        'num_stocks': num_stocks,
        'total_months': total_months,
        'winning_months': winning_months,
        'win_rate_monthly': win_rate_monthly,
        'avg_monthly_return': avg_monthly_return,
        'monthly_volatility': monthly_volatility,
        'total_trades': total_trades,
        'simulation_time': simulation_time,
        'total_api_calls': total_api_calls,
        'failed_api_calls': failed_api_calls,
        'api_success_rate': api_success_rate
    }


def get_monthly_periods(start_date, end_date):
    """
    Generate monthly periods between start_date and end_date.
    Returns list of tuples (month_start, month_end).
    """
    import calendar
    from datetime import datetime, timedelta
    
    periods = []
    current_date = start_date.replace(day=1)  # Start from first day of start month
    
    while current_date <= end_date:
        # First day of the month
        month_start = current_date
        
        # Last day of the month
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1)
        
        month_end = next_month - timedelta(days=1)
        
        # Don't go beyond the specified end_date
        if month_end > end_date:
            month_end = end_date
        
        # Only add if the period has at least one day
        if month_start <= end_date:
            periods.append((month_start, month_end))
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return periods


def run_multiple_simulations(from_date, to_date, num_simulations=100, num_stocks=5, 
                           initial_capital=1000000, csv_file="ind_niftytotalmarket_list.csv"):
    """
    Runs multiple portfolio simulations with progress tracking.
    """
    batch_start_time = time.time()
    logger.info(f"Starting batch of {num_simulations} simulations from {from_date} to {to_date}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    valid_simulations = 0
    total_simulation_time = 0
    
    for i in range(num_simulations):
        sim_start_time = time.time()
        progress = (i + 1) / num_simulations
        progress_bar.progress(progress)
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        
        logger.info(f"Starting simulation {i + 1}/{num_simulations}")
        
        result = run_portfolio_simulation(from_date, to_date, num_stocks, initial_capital, csv_file)
        sim_time = time.time() - sim_start_time
        
        if result is not None:
            results.append(result)
            valid_simulations += 1
            total_simulation_time += result.get('simulation_time', sim_time)
            logger.info(f"Simulation {i + 1} completed successfully in {sim_time:.2f}s (Total return: {result['total_return']:.2f}%)")
        else:
            logger.warning(f"Simulation {i + 1} failed")
        
        # Small delay to show progress
        time.sleep(0.01)
    
    batch_total_time = time.time() - batch_start_time
    avg_sim_time = total_simulation_time / valid_simulations if valid_simulations > 0 else 0
    
    progress_bar.empty()
    status_text.empty()
    
    logger.info(f"Batch completed: {valid_simulations}/{num_simulations} successful simulations in {batch_total_time:.2f}s (avg {avg_sim_time:.2f}s per simulation)")
    
    if len(results) == 0:
        logger.error("No valid simulation results obtained")
        return None
    
    # Calculate statistics
    returns = [result['total_return'] for result in results]
    final_capitals = [result['final_capital'] for result in results]
    
    # Monthly statistics
    monthly_returns_all = []
    total_months_all = []
    winning_months_all = []
    monthly_win_rates = []
    avg_monthly_returns = []
    monthly_volatilities = []
    total_trades_all = []
    
    for result in results:
        if 'monthly_returns' in result:
            monthly_returns_all.extend(result['monthly_returns'])
            total_months_all.append(result.get('total_months', 0))
            winning_months_all.append(result.get('winning_months', 0))
            monthly_win_rates.append(result.get('win_rate_monthly', 0))
            avg_monthly_returns.append(result.get('avg_monthly_return', 0))
            monthly_volatilities.append(result.get('monthly_volatility', 0))
            total_trades_all.append(result.get('total_trades', 0))
    
    # Overall statistics
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    median_return = np.median(returns)
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
    percentile_25 = np.percentile(returns, 25)
    percentile_75 = np.percentile(returns, 75)
    
    # Monthly aggregated statistics
    avg_monthly_return_overall = np.mean(monthly_returns_all) if monthly_returns_all else 0
    monthly_volatility_overall = np.std(monthly_returns_all) if len(monthly_returns_all) > 1 else 0
    avg_monthly_win_rate = np.mean(monthly_win_rates) if monthly_win_rates else 0
    avg_total_trades = np.mean(total_trades_all) if total_trades_all else 0
    
    return {
        'num_simulations': valid_simulations,
        'individual_results': results,
        'statistics': {
            'average_return': avg_return,
            'standard_deviation': std_return,
            'minimum_return': min_return,
            'maximum_return': max_return,
            'median_return': median_return,
            'win_rate': win_rate,
            '25th_percentile': percentile_25,
            '75th_percentile': percentile_75,
            'avg_monthly_return': avg_monthly_return_overall,
            'monthly_volatility': monthly_volatility_overall,
            'avg_monthly_win_rate': avg_monthly_win_rate,
            'avg_total_trades': avg_total_trades
        },
        'returns': returns,
        'final_capitals': final_capitals,
        'monthly_returns_all': monthly_returns_all
    }


def create_distribution_plot(returns, stats, monthly_returns=None):
    """Create interactive distribution plot using Plotly"""
    if monthly_returns and len(monthly_returns) > 0:
        # Create three plots: Total Returns, Monthly Returns, and Box Plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Returns Distribution', 'Monthly Returns Distribution', 
                          'Total Returns Box Plot', 'Monthly Returns Box Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total returns histogram
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, name="Total Returns", 
                        marker_color='skyblue', opacity=0.7),
            row=1, col=1
        )
        
        # Monthly returns histogram
        fig.add_trace(
            go.Histogram(x=monthly_returns, nbinsx=50, name="Monthly Returns", 
                        marker_color='lightcoral', opacity=0.7),
            row=1, col=2
        )
        
        # Total returns box plot
        fig.add_trace(
            go.Box(y=returns, name="Total Returns", marker_color='lightblue'),
            row=2, col=1
        )
        
        # Monthly returns box plot
        fig.add_trace(
            go.Box(y=monthly_returns, name="Monthly Returns", marker_color='lightcoral'),
            row=2, col=2
        )
        
        # Add mean lines for total returns
        fig.add_vline(x=stats['average_return'], line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {stats['average_return']:.2f}%", row=1, col=1)
        
        # Add mean lines for monthly returns
        monthly_mean = np.mean(monthly_returns)
        fig.add_vline(x=monthly_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {monthly_mean:.2f}%", row=1, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Return Distribution Analysis")
        
    else:
        # Original two-plot layout for backward compatibility
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution of Returns', 'Box Plot of Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=30, name="Returns", 
                        marker_color='skyblue', opacity=0.7),
            row=1, col=1
        )
        
        # Add mean and median lines
        fig.add_vline(x=stats['average_return'], line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {stats['average_return']:.2f}%", row=1, col=1)
        fig.add_vline(x=stats['median_return'], line_dash="dash", line_color="orange",
                      annotation_text=f"Median: {stats['median_return']:.2f}%", row=1, col=1)
        
        # Box plot
        fig.add_trace(
            go.Box(y=returns, name="Returns", marker_color='lightblue'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Return Distribution Analysis")
    
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    if monthly_returns and len(monthly_returns) > 0:
        fig.update_xaxes(title_text="Monthly Return (%)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
        fig.update_yaxes(title_text="Monthly Return (%)", row=2, col=2)
    else:
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    
    return fig


def create_performance_metrics(stats, num_simulations, initial_capital):
    """Create performance metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Return (Avg)",
            value=f"{stats['average_return']:.2f}%",
            delta=f"σ: {stats['standard_deviation']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Win Rate",
            value=f"{stats['win_rate']:.1f}%",
            delta=f"{int(stats['win_rate'] * num_simulations / 100)} wins"
        )
    
    with col3:
        st.metric(
            label="Best Return",
            value=f"{stats['maximum_return']:.2f}%",
            delta=f"Worst: {stats['minimum_return']:.2f}%"
        )
    
    with col4:
        st.metric(
            label="Median Return",
            value=f"{stats['median_return']:.2f}%",
            delta=f"IQR: {stats['75th_percentile'] - stats['25th_percentile']:.2f}%"
        )
    
    # Additional row for monthly metrics if available
    if 'avg_monthly_return' in stats and stats['avg_monthly_return'] is not None:
        st.markdown("### Monthly Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Avg Monthly Return",
                value=f"{stats['avg_monthly_return']:.2f}%",
                delta=f"Volatility: {stats['monthly_volatility']:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Monthly Win Rate",
                value=f"{stats['avg_monthly_win_rate']:.1f}%",
                delta="On average"
            )
        
        with col3:
            annualized_return = (1 + stats['avg_monthly_return']/100) ** 12 - 1
            st.metric(
                label="Annualized Return",
                value=f"{annualized_return*100:.2f}%",
                delta="(Estimated)"
            )
        
        with col4:
            st.metric(
                label="Avg Trades/Simulation",
                value=f"{stats['avg_total_trades']:.0f}",
                delta="Monthly rebalancing"
            )


def create_detailed_stats_table(stats, num_simulations):
    """Create detailed statistics table"""
    stats_data = {
        'Metric': [
            'Number of Simulations',
            'Total Average Return (%)',
            'Total Standard Deviation (%)',
            'Total Median Return (%)',
            'Total Minimum Return (%)',
            'Total Maximum Return (%)',
            'Total Win Rate (%)',
            '25th Percentile (%)',
            '75th Percentile (%)',
            'Sharpe Ratio (approx.)'
        ],
        'Value': [
            num_simulations,
            f"{stats['average_return']:.2f}",
            f"{stats['standard_deviation']:.2f}",
            f"{stats['median_return']:.2f}",
            f"{stats['minimum_return']:.2f}",
            f"{stats['maximum_return']:.2f}",
            f"{stats['win_rate']:.1f}",
            f"{stats['25th_percentile']:.2f}",
            f"{stats['75th_percentile']:.2f}",
            f"{stats['average_return'] / stats['standard_deviation']:.2f}" if stats['standard_deviation'] != 0 else "N/A"
        ]
    }
    
    # Add monthly statistics if available
    if 'avg_monthly_return' in stats and stats['avg_monthly_return'] is not None:
        monthly_metrics = [
            'Average Monthly Return (%)',
            'Monthly Volatility (%)',
            'Average Monthly Win Rate (%)',
            'Annualized Return (Est.) (%)',
            'Average Trades per Simulation'
        ]
        
        annualized_return = (1 + stats['avg_monthly_return']/100) ** 12 - 1 if stats['avg_monthly_return'] != 0 else 0
        
        monthly_values = [
            f"{stats['avg_monthly_return']:.2f}",
            f"{stats['monthly_volatility']:.2f}",
            f"{stats['avg_monthly_win_rate']:.1f}",
            f"{annualized_return*100:.2f}",
            f"{stats['avg_total_trades']:.0f}"
        ]
        
        stats_data['Metric'].extend(monthly_metrics)
        stats_data['Value'].extend(monthly_values)
    
    return pd.DataFrame(stats_data)


def main():
    st.title("🎲 Portfolio Simulation Dashboard")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Simulation Parameters")
        
        # Number of simulations
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="More simulations provide better statistical accuracy but take longer to run"
        )
        
        # Initial capital
        initial_capital = st.number_input(
            "Initial Capital (₹)",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=50000,
            format="%d",
            help="Starting amount to invest"
        )
        
        # Number of stocks
        num_stocks = st.slider(
            "Number of Stocks per Portfolio",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of stocks to randomly select for each portfolio"
        )
        
        # Date range
        st.subheader("📅 Time Period")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date(2020, 1, 1),
                min_value=date(2015, 1, 1),
                max_value=date.today()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=date(2020, 12, 31),
                min_value=date(2015, 1, 1),
                max_value=date.today()
            )
        
        # Validate date range
        if start_date >= end_date:
            st.error("End date must be after start date!")
            return
        
        # CSV file path
        csv_file = st.text_input(
            "Stock List CSV File",
            value="ind_niftytotalmarket_list.csv",
            help="Path to the CSV file containing stock symbols"
        )
        
        # Logging controls
        st.subheader("🔍 Debug Options")
        show_logs = st.checkbox(
            "Show Live Logs",
            value=False,
            help="Display detailed logging information during simulation"
        )
        
        log_level = st.selectbox(
            "Log Level",
            options=["INFO", "DEBUG", "WARNING", "ERROR"],
            index=0,
            help="Set the verbosity of logging output"
        )
        
        # Update log level
        if log_level == "DEBUG":
            logger.setLevel(logging.DEBUG)
        elif log_level == "WARNING":
            logger.setLevel(logging.WARNING)
        elif log_level == "ERROR":
            logger.setLevel(logging.ERROR)
        else:
            logger.setLevel(logging.INFO)
        
        # Network configuration
        st.subheader("🌐 Network Settings")
        
        max_retries = st.slider(
            "Max Retries per Stock",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of retry attempts for failed API calls"
        )
        
        retry_delay = st.slider(
            "Retry Delay (seconds)",
            min_value=1,
            max_value=10,
            value=2,
            help="Delay between retry attempts (with exponential backoff)"
        )
        
        # Store network settings in session state for use in simulation
        st.session_state.max_retries = max_retries
        st.session_state.retry_delay = retry_delay
        
        # Run simulation button
        run_button = st.button(
            "🚀 Run Simulation",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if run_button:
        # Validation
        try:
            stock_list = load_stock_list(csv_file)
            if stock_list is None or len(stock_list) == 0:
                st.error("Could not load stock list. Please check the CSV file path.")
                return
        except Exception as e:
            st.error(f"Error loading stock list: {e}")
            return
        
        # Display simulation parameters
        st.subheader("🎯 Simulation Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Simulations:** {num_simulations}")
        with col2:
            st.info(f"**Capital:** ₹{initial_capital:,}")
        with col3:
            st.info(f"**Stocks:** {num_stocks}")
        with col4:
            st.info(f"**Period:** {(end_date - start_date).days} days")
        
        # Network configuration info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🌐 **Max Retries:** {max_retries} per stock")
        with col2:
            st.info(f"⏱️ **Retry Delay:** {retry_delay}s (with backoff)")
        
        # Connection tips
        st.info("""
        💡 **Connection Tips**: The NSE API can be unreliable. If you see connection errors:
        - Higher retry counts help with intermittent failures
        - Longer delays reduce server load but increase runtime  
        - DEBUG logs show detailed retry attempts
        """)
        
        # Log display section
        if show_logs:
            st.subheader("📋 Live Simulation Logs")
            log_container = st.container()
            
            # Create a log display area
            with log_container:
                st.info("🔄 Logs will appear here during simulation...")
                log_placeholder = st.empty()
                
                # Function to read recent logs
                def display_recent_logs():
                    try:
                        with open('portfolio_simulation.log', 'r') as f:
                            lines = f.readlines()
                            recent_logs = lines[-20:] if len(lines) > 20 else lines  # Show last 20 lines
                            log_text = ''.join(recent_logs)
                            with log_placeholder.container():
                                st.code(log_text, language='text')
                    except FileNotFoundError:
                        with log_placeholder.container():
                            st.warning("Log file not found yet...")
                    except Exception as e:
                        with log_placeholder.container():
                            st.error(f"Error reading logs: {e}")
        
        # Run simulations
        st.subheader("🚀 Running Simulations...")
        
        # Create a progress information display
        progress_info = st.empty()
        
        with st.spinner("Running simulations..."):
            start_time = time.time()
            
            if show_logs:
                # Update logs periodically during simulation
                progress_info.info("📊 Check the logs above for detailed progress...")
            
            results = run_multiple_simulations(
                start_date, end_date, num_simulations, 
                num_stocks, initial_capital, csv_file
            )
            
            elapsed_time = time.time() - start_time
            
            if show_logs:
                # Final log update
                display_recent_logs()
                
        progress_info.success(f"⚡ Simulations completed in {elapsed_time:.1f} seconds!")
        
        if results is None:
            st.error("Failed to run simulations. Please check your parameters and try again.")
            return
        
        # Display results
        st.success(f"✅ Completed {results['num_simulations']} simulations successfully!")
        
        # Show final logs if enabled
        if show_logs:
            st.subheader("📋 Final Simulation Logs")
            try:
                with open('portfolio_simulation.log', 'r') as f:
                    log_content = f.read()
                    st.text_area("Complete Log Output", value=log_content[-2000:], height=300, help="Showing last 2000 characters")
                    
                    # Download logs button
                    st.download_button(
                        label="📥 Download Complete Logs",
                        data=log_content,
                        file_name=f"portfolio_simulation_logs_{start_date}_{end_date}.log",
                        mime="text/plain"
                    )
            except FileNotFoundError:
                st.warning("Log file not found")
        
        # Performance metrics
        st.subheader("📈 Performance Overview")
        create_performance_metrics(results['statistics'], results['num_simulations'], initial_capital)
        
        # Charts
        st.subheader("📊 Distribution Analysis")
        monthly_returns = results.get('monthly_returns_all', [])
        fig = create_distribution_plot(results['returns'], results['statistics'], monthly_returns)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Detailed Statistics")
            stats_df = create_detailed_stats_table(results['statistics'], results['num_simulations'])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("💰 Capital Distribution")
            final_capitals = results['final_capitals']
            
            st.metric("Average Final Capital", f"₹{np.mean(final_capitals):,.0f}")
            st.metric("Best Case", f"₹{np.max(final_capitals):,.0f}")
            st.metric("Worst Case", f"₹{np.min(final_capitals):,.0f}")
        
        # Sample portfolios
        st.subheader("🎲 Sample Portfolio Results")
        
        # Show first 10 simulation results
        sample_results = results['individual_results'][:10]
        sample_data = []
        
        for i, result in enumerate(sample_results, 1):
            # Get unique stocks across all months
            all_stocks = result.get('selected_stocks', [])
            total_months = result.get('total_months', 0)
            winning_months = result.get('winning_months', 0)
            
            sample_data.append({
                'Simulation': i,
                'Unique Stocks': ', '.join(all_stocks[:3]) + ('...' if len(all_stocks) > 3 else ''),
                'Months': total_months,
                'Winning Months': winning_months,
                'Final Capital': f"₹{result['final_capital']:,.0f}",
                'Total Return': f"{result['total_return']:.2f}%",
                'Avg Monthly Return': f"{result.get('avg_monthly_return', 0):.2f}%"
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        # Show monthly breakdown for first simulation
        if len(results['individual_results']) > 0:
            first_result = results['individual_results'][0]
            if 'monthly_details' in first_result:
                st.subheader("📅 Monthly Breakdown (First Simulation)")
                monthly_data = []
                for detail in first_result['monthly_details']:
                    monthly_data.append({
                        'Month': detail['month'],
                        'Stocks': ', '.join(detail['stocks'][:2]) + ('...' if len(detail['stocks']) > 2 else ''),
                        'Start Capital': f"₹{detail['start_capital']:,.0f}",
                        'End Capital': f"₹{detail['end_capital']:,.0f}",
                        'Monthly Return': f"{detail['return']:.2f}%"
                    })
                
                monthly_df = pd.DataFrame(monthly_data)
                st.dataframe(monthly_df, use_container_width=True, hide_index=True)
        
        # Download results
        st.subheader("📥 Export Results")
        
        # Prepare data for download
        export_data = []
        for i, result in enumerate(results['individual_results'], 1):
            base_data = {
                'Simulation_Number': i,
                'Initial_Capital': result['initial_capital'],
                'Final_Capital': result['final_capital'],
                'Total_Return_Percent': result['total_return'],
                'Profit_Loss': result['final_capital'] - result['initial_capital'],
                'Total_Months': result.get('total_months', 0),
                'Winning_Months': result.get('winning_months', 0),
                'Monthly_Win_Rate': result.get('win_rate_monthly', 0),
                'Avg_Monthly_Return': result.get('avg_monthly_return', 0),
                'Monthly_Volatility': result.get('monthly_volatility', 0),
                'Total_Trades': result.get('total_trades', 0)
            }
            
            # Add monthly returns as separate columns
            if 'monthly_returns' in result:
                for j, monthly_return in enumerate(result['monthly_returns'], 1):
                    base_data[f'Month_{j}_Return'] = monthly_return
            
            export_data.append(base_data)
        
        export_df = pd.DataFrame(export_data)
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Results as CSV",
            data=csv_data,
            file_name=f"portfolio_simulation_monthly_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        # Initial state - show instructions
        st.info("""
        👋 **Welcome to the Portfolio Simulation Dashboard!**
        
        This tool simulates random portfolio selection with **monthly rebalancing** strategy and calculates returns over longer time periods.
        
        **How it works:**
        - 🗓️ **Monthly Rebalancing**: Buys stocks on the 1st trading day of each month, sells on the last day
        - 💰 **Capital Rollover**: Profits/losses carry forward to the next month
        - 🎲 **Random Selection**: New random stocks selected each month for diversification
        - 📈 **Long-term Analysis**: Ideal for backtesting periods of several months to years
        
        **How to use:**
        1. Set your simulation parameters in the sidebar
        2. Choose a longer time period (months/years) for meaningful results
        3. Click "Run Simulation" to start the analysis
        4. View monthly breakdowns, total returns, and download detailed data
        
        **Features:**
        - 📊 Interactive charts for both total and monthly returns
        - 📈 Comprehensive performance statistics (total + monthly metrics)
        - 🎲 Monthly random stock selection simulation
        - 📅 Detailed monthly breakdown and analysis
        - 📥 Export comprehensive results to CSV
        - � **Live Logging**: Monitor simulation progress with detailed logs
        - 📋 **Debug Options**: Adjustable log levels and real-time monitoring
        - ⚡ **Performance Tracking**: See timing information for each step
        - �📱 Responsive web interface
        """)


if __name__ == "__main__":
    main()