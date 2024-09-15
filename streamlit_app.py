import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from gspread import authorize
from google.oauth2.service_account import Credentials
import requests
from fpdf import FPDF  # Add this import

# Hardcoded credentials (for demonstration purposes, you may adjust as necessary)
USERNAME = "admin"
PASSWORD = "password@123"

st.set_page_config(
    page_title="Stock Prediction by Xcodefix Global",  # Set the title of the app
    page_icon=":chart_with_upwards_trend:",  # Use an emoji as the favicon

)
# Initialize session state for authentication and search history
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Function to check authentication
def check_authentication():
    if not st.session_state.authenticated:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.success("Authenticated successfully!")
            else:
                st.error("Invalid credentials. Please try again.")

# Function to fetch stock data
def fetch_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end=datetime.date.today().strftime('%Y-%m-%d'))
    if data.empty:
        st.error(f"No data found for ticker {ticker}. Please check the ticker symbol.")
    return data

# Function to train the LSTM model
def train_model(data):
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data, 60)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_test, Y_test = create_dataset(test_data, 60)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1)

    Y_predicted = model.predict(X_test)
    Y_test = Y_test.reshape(-1, 1)
    Y_predicted = scaler.inverse_transform(Y_predicted)
    Y_test = scaler.inverse_transform(Y_test)

    return model, scaler, Y_test, Y_predicted

# Function to predict the next 30 days
def predict_next_30_days(model, scaler, data):
    last_60_days = data['Close'][-60:].values.reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    next_30_days = []

    input_data = last_60_days_scaled.reshape(1, last_60_days_scaled.shape[0], 1)

    for _ in range(30):
        next_price_scaled = model.predict(input_data)
        next_price = scaler.inverse_transform(next_price_scaled)[0][0]
        next_30_days.append(next_price)

        new_input = np.append(input_data[0][1:], next_price_scaled)
        input_data = new_input.reshape(1, 60, 1)

    return next_30_days

# Function to fetch latest news based on ticker
def fetch_latest_news(ticker):
    # Extract the part before the dot, if any
    base_ticker = ticker.split('.')[0]

    # Corrected URL with proper space and encoding
    news_url = f"https://newsapi.org/v2/everything?q={base_ticker}%20India&apiKey={st.secrets['newsapi']['api_key']}"

    print(news_url)
    response = requests.get(news_url)
    if response.status_code == 200:
        news = response.json()
        return news['articles'][:5]  # Fetch top 5 news articles
    else:
        return []

# Function to save data to Google Sheets
# Function to save data to Google Sheets
def save_to_google_sheets(ticker, predictions, news):
    # Convert predictions to a serializable format (floats)
    predictions_serializable = [float(p) for p in predictions]

    # Convert news to a serializable format (just saving the titles of news articles)
    news_serializable = [article['title'] for article in news if 'title' in article]  # Extract titles

    # Concatenate news titles into a single string to avoid list issues
    news_titles_str = '; '.join(news_serializable) if news_serializable else 'No news available'

    # Load Google Sheets credentials from secrets
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = authorize(credentials)

    # Open Google Sheet by URL
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/10nvN2HgnNS0bvsjQh9eTEyPLi_YruVomoHNFmICcpbs/edit?usp=sharing").sheet1

    # Append row with ticker, timestamp, predictions, and news
    sheet.append_row([ticker, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + predictions_serializable + [news_titles_str])

# Function to generate PDF report
def generate_pdf_report(ticker, future_data, news_articles):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Stock Price Prediction Report", ln=True, align='C')

    # Ticker info
    pdf.cell(200, 10, txt=f"Stock Ticker: {ticker}", ln=True, align='L')

    # Predicted Prices
    pdf.cell(200, 10, txt="Predicted Prices for the Next 30 Days:", ln=True, align='L')
    for index, row in future_data.iterrows():
        pdf.cell(200, 10, txt=f"Date: {row['Date'].strftime('%Y-%m-%d')}, Price: {row['Predicted Price']:.2f}", ln=True, align='L')

    # News Articles
    pdf.cell(200, 10, txt="Latest News Articles:", ln=True, align='L')
    for article in news_articles:
        pdf.cell(200, 10, txt=f"{article['title']} - {article['source']['name']}", ln=True, align='L')

    # Save the PDF to a file
    pdf_file = f"{ticker}_stock_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# Start of Streamlit application
if not st.session_state.authenticated:
    check_authentication()

if st.session_state.authenticated:
    st.title("Stock Price Prediction App")
    st.markdown("This app uses an LSTM model to predict the future stock price.")

    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, GOOGL):")

    # Search history sidebar
    st.sidebar.title("Search History")
    if st.session_state.search_history:
        st.sidebar.write(st.session_state.search_history)
    else:
        st.sidebar.write("No search history yet.")

    if ticker:
        # Add to search history
        st.session_state.search_history.append(ticker)

        st.write("Fetching data...")
        data = fetch_data(ticker)

        if not data.empty:
            st.write("Displaying historical data...")
            st.line_chart(data['Close'])

            st.write("Training the model...")
            model, scaler, Y_test, Y_predicted = train_model(data)

            st.write("Predicted vs Actual Price")
            plt.figure(figsize=(10, 6))
            plt.plot(Y_test, label='Actual Price (INR)', color='cyan')
            plt.plot(Y_predicted, label='Predicted Price (INR)', color='red')
            plt.title('Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            accuracy = 100 - np.mean(np.abs((Y_test - Y_predicted) / Y_test)) * 100
            st.write(f"Model Accuracy: {accuracy:.2f}%")

            st.write("Predicting the next 30 days...")
            future_prices = predict_next_30_days(model, scaler, data)

            future_dates = pd.date_range(start=datetime.date.today(), periods=30).to_pydatetime().tolist()
            future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})
            st.write(future_data)

            plt.figure(figsize=(10, 6))
            plt.plot(future_dates, future_prices, label='Predicted Price (INR)', color='blue')
            plt.title('Next 30 Days Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)

            # Fetch and display latest news
            st.write("Fetching latest news...")
            news_articles = fetch_latest_news(ticker)
            for article in news_articles:
                st.write(f"**{article['title']}** - {article['source']['name']}")
                st.write(article['description'])

            # Save to Google Sheets
            save_to_google_sheets(ticker, future_prices, news_articles)

            # Generate Report Button
            if st.button("Generate Report"):
                pdf_file = generate_pdf_report(ticker, future_data, news_articles)
                with open(pdf_file, "rb") as file:
                    st.download_button(label="Download Report", data=file, file_name=pdf_file)

    # Add footer
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent; /* Adjust this if you need a background color */
        color: white;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        box-sizing: border-box; /* Ensures padding doesn't affect width */
    }
    </style>
    <div class="footer">
        <p>Powered by: Xcodefix Global IT Solutions</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
    st.title("Stock Price Prediction App")
