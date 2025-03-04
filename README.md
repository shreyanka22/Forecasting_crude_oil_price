# Forecasting_crude_oil_price

This Streamlit application predicts stock prices using a Long Short-Term Memory (LSTM) neural network. Users can upload a CSV file containing historical stock data, train the model with customizable parameters, and generate future price predictions.

## Features

-   **Data Upload:** Upload historical stock price data in CSV format.
-   **Model Customization:** Adjust LSTM units, dropout rate, epochs, and sequence length.
-   **Prediction Types:** Predict stock prices for the next N days or a custom date range.
-   **Interactive Visualizations:** Display historical stock prices, training loss, actual vs. predicted prices, and future predictions with confidence intervals.
-   **Accuracy Metrics:** Show detailed model performance metrics, including RMSE, MAE, R², MAPE, and accuracy gauges.
-   **Data Export:** Download future price predictions as a CSV file.
-   **Clear Error Handling:** Provides user-friendly error messages for data issues and invalid inputs.
-   **Dynamic Accuracy Visualization**: Accuracy over time is visualized.
-   **Confidence Interval**: Future predictions contain a confidence interval.

## Dependencies

-   `streamlit`
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `tensorflow`
-   `plotly`

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install streamlit pandas numpy scikit-learn tensorflow plotly
    ```

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run stock_predictor.py
    ```

    (Replace `stock_predictor.py` with the actual name of your Python file.)

2.  **Upload Data:**
    * Click the "Browse files" button in the sidebar and select your CSV file.
    * Ensure your CSV file has "Date" and "Close" columns.

3.  **Configure Model Parameters:**
    * Adjust the LSTM units, dropout rate, epochs, and sequence length in the sidebar.

4.  **Set Prediction Settings:**
    * Choose "Next N Days" or "Custom Date Range" for predictions.
    * If "Next N Days," specify the number of days.
    * If "Custom Date Range," select the start and end dates.

5.  **Train and Predict:**
    * Click the "Train Model and Make Predictions" button.

6.  **View Results:**
    * The app will display historical data, model metrics, and future predictions with interactive plots.
    * You can download the predictions as a CSV file.

## CSV File Format

The CSV file should have the following format:

```csv
Date,Close
2024-01-02,75.70
2024-01-03,77.33
2024-01-04,76.015
...
