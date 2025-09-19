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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_stock_list(csv_file):
    """Load and cache the stock list from CSV"""
    try:
        df = pd.read_csv(csv_file)
        return df[["Symbol"]].copy()
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        return None


def run_portfolio_simulation(from_date, to_date, num_stocks=5, initial_capital=1000000, 
                           csv_file="ind_niftytotalmarket_list.csv", verbose=False):
    """
    Runs a single portfolio simulation with random stock selection.
    """
    # Load stock list
    ind_niftytotalmarket_list = load_stock_list(csv_file)
    if ind_niftytotalmarket_list is None:
        return None
    
    ind_niftytotalmarket_list = ind_niftytotalmarket_list.copy()

    # Select random stocks
    selected_stocks = []
    while len(selected_stocks) < num_stocks:
        die_roll = random.randint(0, len(ind_niftytotalmarket_list) - 1)
        selected_stocks.append(ind_niftytotalmarket_list.iloc[die_roll]["Symbol"])
        ind_niftytotalmarket_list = ind_niftytotalmarket_list.drop(die_roll).reset_index(drop=True)
        
    # Get stock data and handle errors
    portfolio = {}
    offending = []
    for stock in selected_stocks:
        try:
            df = stock_df(stock, from_date, to_date)
            if len(df) > 0:
                portfolio[stock] = df
            else:
                offending.append(stock)
        except:
            offending.append(stock)
            if verbose:
                st.warning(f"{stock} did not trade during this period")

    # Remove offending stocks
    for stock in offending:
        if stock in selected_stocks:
            selected_stocks.remove(stock)

    # Replace offending stocks
    while len(offending) > 0 and len(ind_niftytotalmarket_list) > 0:
        die_roll = random.randint(0, len(ind_niftytotalmarket_list) - 1)
        new_stock = ind_niftytotalmarket_list.iloc[die_roll]["Symbol"]
        if new_stock not in selected_stocks:
            selected_stocks.append(new_stock)
            ind_niftytotalmarket_list = ind_niftytotalmarket_list.drop(die_roll).reset_index(drop=True)
            offending.remove(offending[0])
            try:
                df = stock_df(new_stock, from_date, to_date)
                if len(df) > 0:
                    portfolio[new_stock] = df
                else:
                    offending.append(new_stock)
                    selected_stocks.remove(new_stock)
            except:
                offending.append(new_stock)
                selected_stocks.remove(new_stock)
                if verbose:
                    st.warning(f"{new_stock} did not trade during this period")

    if len(selected_stocks) == 0:
        return None

    # Calculate returns
    allocation_per_stock = initial_capital / len(selected_stocks)
    final_capital = 0
    stock_returns = {}
    
    for stock in selected_stocks:
        df = portfolio[stock]
        buy_price = df.iloc[0]["OPEN"]
        sell_price = df.iloc[-1]["CLOSE"]
        shares_bought = allocation_per_stock / buy_price
        final_value = shares_bought * sell_price
        final_capital += final_value
        
        stock_return = (sell_price - buy_price) / buy_price * 100
        stock_returns[stock] = stock_return
    
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    return {
        'selected_stocks': selected_stocks,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'stock_returns': stock_returns,
        'from_date': from_date,
        'to_date': to_date,
        'num_stocks': len(selected_stocks)
    }


def run_multiple_simulations(from_date, to_date, num_simulations=100, num_stocks=5, 
                           initial_capital=1000000, csv_file="ind_niftytotalmarket_list.csv"):
    """
    Runs multiple portfolio simulations with progress tracking.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    valid_simulations = 0
    
    for i in range(num_simulations):
        progress = (i + 1) / num_simulations
        progress_bar.progress(progress)
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        
        result = run_portfolio_simulation(from_date, to_date, num_stocks, initial_capital, csv_file)
        if result is not None:
            results.append(result)
            valid_simulations += 1
        
        # Small delay to show progress
        time.sleep(0.01)
    
    progress_bar.empty()
    status_text.empty()
    
    if len(results) == 0:
        return None
    
    # Calculate statistics
    returns = [result['total_return'] for result in results]
    final_capitals = [result['final_capital'] for result in results]
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    min_return = np.min(returns)
    max_return = np.max(returns)
    median_return = np.median(returns)
    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
    percentile_25 = np.percentile(returns, 25)
    percentile_75 = np.percentile(returns, 75)
    
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
            '75th_percentile': percentile_75
        },
        'returns': returns,
        'final_capitals': final_capitals
    }


def create_distribution_plot(returns, stats):
    """Create interactive distribution plot using Plotly"""
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
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Return Distribution Analysis"
    )
    
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    
    return fig


def create_performance_metrics(stats, num_simulations, initial_capital):
    """Create performance metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average Return",
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


def create_detailed_stats_table(stats, num_simulations):
    """Create detailed statistics table"""
    stats_data = {
        'Metric': [
            'Number of Simulations',
            'Average Return (%)',
            'Standard Deviation (%)',
            'Median Return (%)',
            'Minimum Return (%)',
            'Maximum Return (%)',
            'Win Rate (%)',
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
                value=date(2020, 1, 31),
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
        
        # Run simulations
        with st.spinner("Running simulations..."):
            results = run_multiple_simulations(
                start_date, end_date, num_simulations, 
                num_stocks, initial_capital, csv_file
            )
        
        if results is None:
            st.error("Failed to run simulations. Please check your parameters and try again.")
            return
        
        # Display results
        st.success(f"✅ Completed {results['num_simulations']} simulations successfully!")
        
        # Performance metrics
        st.subheader("📈 Performance Overview")
        create_performance_metrics(results['statistics'], results['num_simulations'], initial_capital)
        
        # Charts
        st.subheader("📊 Distribution Analysis")
        fig = create_distribution_plot(results['returns'], results['statistics'])
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
            sample_data.append({
                'Simulation': i,
                'Stocks': ', '.join(result['selected_stocks'][:3]) + ('...' if len(result['selected_stocks']) > 3 else ''),
                'Final Capital': f"₹{result['final_capital']:,.0f}",
                'Return': f"{result['total_return']:.2f}%",
                'Profit/Loss': f"₹{result['final_capital'] - result['initial_capital']:,.0f}"
            })
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        # Download results
        st.subheader("📥 Export Results")
        
        # Prepare data for download
        export_data = []
        for i, result in enumerate(results['individual_results'], 1):
            export_data.append({
                'Simulation_Number': i,
                'Selected_Stocks': '|'.join(result['selected_stocks']),
                'Initial_Capital': result['initial_capital'],
                'Final_Capital': result['final_capital'],
                'Total_Return_Percent': result['total_return'],
                'Profit_Loss': result['final_capital'] - result['initial_capital'],
                'Number_of_Stocks': result['num_stocks']
            })
        
        export_df = pd.DataFrame(export_data)
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Results as CSV",
            data=csv_data,
            file_name=f"portfolio_simulation_results_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        # Initial state - show instructions
        st.info("""
        👋 **Welcome to the Portfolio Simulation Dashboard!**
        
        This tool simulates random portfolio selection and calculates returns over a specified time period.
        
        **How to use:**
        1. Set your simulation parameters in the sidebar
        2. Choose the time period for backtesting
        3. Click "Run Simulation" to start the analysis
        4. View the results and download the data
        
        **Features:**
        - 📊 Interactive charts and visualizations
        - 📈 Comprehensive performance statistics
        - 🎲 Random stock selection simulation
        - 📥 Export results to CSV
        - 📱 Responsive web interface
        """)


if __name__ == "__main__":
    main()