# Portfolio Simulation Dashboard

A web-based tool for simulating random portfolio selection and analyzing investment returns using historical stock data from the NSE (National Stock Exchange of India).

## Features

- 🎲 **Random Portfolio Simulation**: Randomly selects stocks from a predefined list and simulates investment returns
- 📊 **Interactive Visualizations**: Uses Plotly for interactive charts and data exploration
- 📈 **Comprehensive Statistics**: Provides detailed performance metrics including win rate, risk measures, and return distributions
- 🚀 **Real-time Progress**: Shows progress bars during long-running simulations
- 📥 **Export Capabilities**: Download simulation results as CSV files
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd trading-strat
   ```
3. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your web browser and go to `http://localhost:8501`
3. Configure your simulation parameters in the sidebar:
   - **Number of Simulations**: How many random portfolios to simulate (10-1000)
   - **Initial Capital**: Starting investment amount in ₹
   - **Number of Stocks**: How many stocks to include in each portfolio (1-20)
   - **Time Period**: Start and end dates for the backtest
   - **Stock List File**: CSV file containing stock symbols (default: `ind_niftytotalmarket_list.csv`)
4. Click "🚀 Run Simulation" to start the analysis
5. View the results including:
   - Performance overview with key metrics
   - Interactive distribution charts
   - Detailed statistics table
   - Sample portfolio results
   - Export options

## Input Parameters

### Simulation Settings
- **Number of Simulations**: Controls statistical accuracy (more = better accuracy, longer runtime)
- **Initial Capital**: The amount of money to invest (in ₹)
- **Number of Stocks per Portfolio**: Diversification level (more stocks = more diversified)

### Time Period
- **Start Date**: Beginning of the investment period
- **End Date**: End of the investment period
- The period should have sufficient trading days for meaningful results

### Data Source
- **Stock List CSV**: Should contain a "Symbol" column with NSE stock symbols
- Default file: `ind_niftytotalmarket_list.csv`

## Output Analysis

### Performance Metrics
- **Average Return**: Mean return across all simulations
- **Win Rate**: Percentage of simulations with positive returns
- **Best/Worst Returns**: Range of possible outcomes
- **Median Return**: Middle value of return distribution
- **Standard Deviation**: Risk measure (volatility)
- **Sharpe Ratio**: Risk-adjusted return measure

### Visualizations
- **Distribution Histogram**: Shows frequency of different return levels
- **Box Plot**: Displays quartiles, median, and outliers
- **Interactive Features**: Zoom, pan, and hover for detailed information

### Export Options
- Download complete results as CSV
- Includes all simulation details for further analysis
- File naming includes date range for organization

## File Structure

```
trading-strat/
├── streamlit_app.py           # Main Streamlit application
├── main.py                    # Original simulation code
├── requirements.txt           # Python dependencies
├── ind_niftytotalmarket_list.csv  # Stock symbols list
├── sim-strat.ipynb           # Jupyter notebook version
├── die-rolling.ipynb         # Additional analysis notebook
└── README.md                 # This file
```

## Dependencies

- **Streamlit**: Web application framework
- **Plotly**: Interactive plotting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Static plotting (backup)
- **jugaad-data**: NSE stock data fetching

## Data Source

This application uses the `jugaad-data` library to fetch historical stock data from the NSE. The data includes:
- Open, High, Low, Close prices
- Volume information
- Daily trading data

## Limitations

- Data availability depends on `jugaad-data` service
- Some stocks may not have data for selected periods
- Simulation assumes equal allocation to all selected stocks
- Transaction costs and slippage are not considered
- Results are for educational/research purposes only

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.

## License

This project is open source. Use at your own discretion.