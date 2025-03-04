#to b executed
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# App title and description
st.title("Stock Price Prediction")
st.markdown("Upload historical stock price data, train the model, and predict future prices.")

# Sidebar for uploading data and model parameters
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV file with stock data", type=["csv"])
    
    st.subheader("Model Parameters")
    lstm_units = st.slider("LSTM Units", 10, 200, 100, 10)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    epochs = st.slider("Epochs", 10, 100, 50, 5)
    sequence_length = st.slider("Sequence Length (days to look back)", 5, 60, 30, 5)
    
    st.subheader("Prediction Settings")
    prediction_type = st.radio("Prediction Type", ["Next N Days", "Custom Date Range"])
    
    if prediction_type == "Next N Days":
        prediction_days = st.slider("Number of Days to Predict", 1, 30, 7, 1)
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() + timedelta(days=1))
        with col2:
            end_date = st.date_input("End Date", datetime.now() + timedelta(days=30))

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to date without time
    df = df.sort_values('Date')
    # Fill zero values with the previous day's value
    df['Close'] = df['Close'].replace(0, np.nan)
    df['Close'] = df['Close'].fillna(method='ffill')
    return df

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to build and train LSTM model
def build_and_train_model(X_train, y_train, X_val, y_val, lstm_units, dropout_rate, epochs):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Using Streamlit's progress bar for training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Training Progress: {int(progress * 100)}%")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[TrainingCallback()],
        verbose=0
    )
    
    return model, history

# Function to evaluate model
def evaluate_model(model, X_test, y_test, scaler):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Calculate prediction accuracy percentage (100 - MAPE, bounded at 0)
    accuracy = max(0, 100 - mape)
    
    # Calculate additional accuracy metrics
    # Directional Accuracy (% of times the model correctly predicts price movement direction)
    actual_direction = np.sign(np.diff(y_test_inv.flatten()))
    pred_direction = np.sign(np.diff(y_pred_inv.flatten()))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Calculate Normalized RMSE (RMSE / range of actual values)
    norm_rmse = rmse / (np.max(y_test_inv) - np.min(y_test_inv)) * 100
    forecast_quality = max(0, 100 - norm_rmse)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy,
        'directional_accuracy': directional_accuracy,
        'forecast_quality': forecast_quality
    }

# Function to make predictions
def predict_future(model, last_sequence, scaler, num_days):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(num_days):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict the next day
        next_day_scaled = model.predict(current_sequence_reshaped, verbose=0)[0]
        
        # Add to predictions
        future_predictions.append(next_day_scaled[0])
        
        # Update sequence for the next prediction
        current_sequence = np.append(current_sequence[1:], [[next_day_scaled[0]]], axis=0)
    
    # Inverse transform to get actual prices
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Function to predict for specific dates
def predict_for_date_range(model, last_sequence, scaler, start_date, end_date, last_date):
    # Calculate number of days between the last data point and the end prediction date
    days_diff = (end_date - last_date).days
    
    # Make prediction for all days up to the end date
    all_predictions = predict_future(model, last_sequence, scaler, days_diff)
    
    # Calculate the start index based on the difference between start_date and last_date
    start_idx = max(0, (start_date - last_date).days - 1)
    
    # Return only the predictions for the requested date range
    return all_predictions[start_idx:]

# Function to create accuracy gauge chart
def create_accuracy_gauge(accuracy, title="Prediction Accuracy"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'firebrick'},
                {'range': [50, 75], 'color': 'gold'},
                {'range': [75, 100], 'color': 'forestgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main application
if uploaded_file is not None:
    # Load and display the data
    df = load_data(uploaded_file)
    
    # Display basic statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
    
    # Plot the stock price history
    st.subheader("Stock Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Historical Stock Prices', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
    
    # Train the model button
    if st.button("Train Model and Make Predictions"):
        with st.spinner("Preprocessing data..."):
            # Prepare the data
            data = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            X, y = create_sequences(scaled_data, sequence_length)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Split the data into training, validation, and testing sets
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.15)
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        st.subheader("Training LSTM Model")
        model, history = build_and_train_model(X_train, y_train, X_val, y_val, lstm_units, dropout_rate, epochs)
        
        # Display training results
        st.success("Training complete!")
        
        # Plot training loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
        fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig, use_container_width=True)
        
        # Evaluate the model on test data
        metrics = evaluate_model(model, X_test, y_test, scaler)
        
        # PROMINENT ACCURACY DASHBOARD
        st.subheader("📊 Model Accuracy Dashboard")
        st.markdown("### Key Performance Indicators for Vendors")
        
        # Create three gauge charts for different accuracy metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = create_accuracy_gauge(metrics['accuracy'], "Overall Prediction Accuracy")
            st.plotly_chart(fig1, use_container_width=True)
            st.info(f"Overall price prediction accuracy: **{metrics['accuracy']:.2f}%**")
            
        with col2:
            fig2 = create_accuracy_gauge(metrics['directional_accuracy'], "Directional Accuracy")
            st.plotly_chart(fig2, use_container_width=True)
            st.info(f"Correctly predicts price direction: **{metrics['directional_accuracy']:.2f}%** of the time")
            
        with col3:
            fig3 = create_accuracy_gauge(metrics['forecast_quality'], "Forecast Quality")
            st.plotly_chart(fig3, use_container_width=True)
            st.info(f"Forecast quality score: **{metrics['forecast_quality']:.2f}%**")
        
        # Detailed metrics in expandable section
        with st.expander("Detailed Model Metrics"):
            # Display metrics in columns
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            with detail_col1:
                st.metric("RMSE (Root Mean Squared Error)", f"{metrics['rmse']:.4f}")
                st.metric("MAE (Mean Absolute Error)", f"{metrics['mae']:.4f}")
            with detail_col2:
                st.metric("R² Score (1 is perfect)", f"{metrics['r2']:.4f}")
                st.metric("MSE (Mean Squared Error)", f"{metrics['mse']:.4f}")
            with detail_col3:
                st.metric("MAPE (Mean Absolute % Error)", f"{metrics['mape']:.2f}%")
        
        # Plot actual vs predicted for test data
        y_pred = model.predict(X_test, verbose=0)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Create test dates (these will be offset by sequence_length from the original dates)
        test_dates = df['Date'].iloc[train_size+val_size+sequence_length:].reset_index(drop=True)
        
        # Create test comparison dataframe
        test_df = pd.DataFrame({
            'Date': test_dates[:len(y_test_inv)], 
            'Actual': y_test_inv.flatten(),
            'Predicted': y_pred_inv.flatten()
        })
        
        # Calculate accuracy for each prediction
        test_df['Error'] = np.abs(test_df['Actual'] - test_df['Predicted'])
        test_df['Error_Percentage'] = (test_df['Error'] / test_df['Actual']) * 100
        test_df['Accuracy_Percentage'] = 100 - test_df['Error_Percentage']
        
        st.subheader("Model Performance (Test Data)")
        st.dataframe(test_df.head(10))
        
        # Create a visual showing accuracy over time - FIXED SECTION
        st.subheader("Prediction Accuracy Over Time")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add actual vs predicted lines
        fig.add_trace(
            go.Scatter(x=test_df['Date'], y=test_df['Actual'], mode='lines', name='Actual Price'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=test_df['Date'], y=test_df['Predicted'], mode='lines', name='Predicted Price'),
            secondary_y=False
        )
        
        # Add accuracy as a line on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=test_df['Date'], 
                y=test_df['Accuracy_Percentage'], 
                mode='lines', 
                name='Prediction Accuracy (%)',
                line=dict(color='green', width=1, dash='dot')
            ),
            secondary_y=True
        )
        
        # Update the layout
        fig.update_layout(
            title='Actual vs Predicted with Accuracy Percentage',
            xaxis_title='Date',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(
            title_text="Accuracy (%)", 
            secondary_y=True, 
            range=[0, 100],
            title_font=dict(color='green'),
            tickfont=dict(color='green')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Make future predictions
        with st.spinner("Making predictions..."):
            # Get the last sequence from the data to predict future values
            last_sequence = scaled_data[-sequence_length:]
            last_date = df['Date'].iloc[-1]
            
            if prediction_type == "Next N Days":
                # Predict the next N days
                future_predictions = predict_future(model, last_sequence, scaler, prediction_days)
                
                # Create future dates
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            else:
                # Validate date range
                if start_date <= last_date:
                    st.error(f"Start date must be after the last date in your data ({last_date})")
                    st.stop()
                if end_date < start_date:
                    st.error("End date must be after start date")
                    st.stop()
                
                # Predict for custom date range
                future_predictions = predict_for_date_range(model, last_sequence, scaler, start_date, end_date, last_date)
                
                # Create date range
                date_range = pd.date_range(start=start_date, end=end_date)
                future_dates = [d.date() for d in date_range]
                
                # Ensure correct length
                future_dates = future_dates[:len(future_predictions)]
            
            # Create DataFrame for future predictions
            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': future_predictions.flatten(),
                'Confidence Level': np.clip(metrics['accuracy'] - np.arange(len(future_predictions)) * 0.5, 50, metrics['accuracy'])
            })
            
            # # Show predictions with confidence
            # st.subheader("Future Price Predictions with Confidence Levels")
            
            # # Display the forecast with confidence
            # fig = go.Figure()
            
            # Add the predictions
            fig.add_trace(go.Scatter(
                x=future_df['Date'], 
                y=future_df['Predicted Close'], 
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red')
            ))
            
            # Add confidence information
            hover_text = [f"Price: ${price:.2f}<br>Confidence: {conf:.1f}%" 
                         for price, conf in zip(future_df['Predicted Close'], future_df['Confidence Level'])]
            
            # Add confidence band
            for i, row in future_df.iterrows():
                confidence = row['Confidence Level'] / 100
                price = row['Predicted Close']
                upper_bound = price * (1 + (1-confidence)/2)
                lower_bound = price * (1 - (1-confidence)/2)
                
                fig.add_trace(go.Scatter(
                    x=[row['Date'], row['Date']],
                    y=[lower_bound, upper_bound],
                    mode='lines',
                    line=dict(width=0),
                    fill=None,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Update hover info for main prediction line
            fig.update_traces(
                hovertemplate='%{text}',
                text=hover_text,
                selector=dict(name='Predicted Price')
            )
            
            fig.update_layout(
                title='Future Price Predictions with Confidence',
                xaxis_title='Date',
                yaxis_title='Predicted Price',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(future_df)
            
            # Plot historical data and predictions
            st.subheader("Historical Prices and Predictions")
            
            # Create a figure with make_subplots
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Add historical prices
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical Close Price'),
                secondary_y=False,
            )
            
            # Add test predictions
            fig.add_trace(
                go.Scatter(x=test_df['Date'], y=test_df['Predicted'], mode='lines', 
                          name='Test Predictions', line=dict(color='orange')),
                secondary_y=False,
            )
            
            # Add future predictions
            fig.add_trace(
                go.Scatter(x=future_df['Date'], y=future_df['Predicted Close'], mode='lines+markers', 
                          name='Future Predictions', line=dict(color='red')),
                secondary_y=False,
            )
            
            # Add range slider
            fig.update_layout(
                title='Stock Price Prediction',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type='date'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export predictions
            st.download_button(
                label="Download Predictions as CSV",
                data=future_df.to_csv(index=False).encode('utf-8'),
                file_name='stock_predictions.csv',
                mime='text/csv',
            )
            
            # # Custom date prediction input
            # st.subheader("Predict for a Specific Date")
            # custom_date = st.date_input("Select a date", datetime.now() + timedelta(days=1))
            
            # if st.button("Get Prediction"):
            #     if custom_date <= last_date:
            #         st.error(f"The date must be after the last date in your data ({last_date})")
            #     else:
            #         days_to_predict = (custom_date - last_date).days
            #         prediction = predict_future(model, last_sequence, scaler, days_to_predict)[-1][0]
            #         confidence = max(50, metrics['accuracy'] - days_to_predict * 0.5)
                    
            #         st.success(f"Predicted price for {custom_date}: ${prediction:.2f} (Confidence: {confidence:.1f}%)")
                    
            #         # Create a simple gauge for this specific prediction
            #         fig = create_accuracy_gauge(confidence, f"Prediction Confidence for {custom_date}")
            #         st.plotly_chart(fig, use_container_width=True)
else:
    # If no file is uploaded, show instructions
    st.info("Please upload a CSV file with stock price data to get started. The CSV should have 'Date' and 'Close' columns.")
    st.markdown("""
    ### Sample CSV Format:
    ```
    Date,Close
    1/2/24,75.7
    1/3/24,77.33
    1/4/24,76.015
    ...
    ```
    """)
    
    # Create sample visualization with dummy data
    dates = pd.date_range(start='2024-01-01', periods=100)
    prices = np.random.normal(loc=80, scale=5, size=100).cumsum() + 80
    dummy_df = pd.DataFrame({'Date': dates, 'Close': prices})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dummy_df['Date'], y=dummy_df['Close'], mode='lines', name='Sample Data'))
    fig.update_layout(title='Sample Stock Price Visualization', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)