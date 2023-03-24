import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import datetime as dt

# Set the page title
st.set_page_config(page_title='Sales Prediction')

# Add a title to the app
st.title('Sales Prediction')

# Add a file input widget to the app
file = st.file_uploader('Upload a CSV file', type=['csv'])

if file is not None:
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(file)
    # Convert the date column to a numerical value
    df['date'] = pd.to_datetime(df['date']).map(dt.datetime.toordinal)
    # Train the linear regression model on the entire dataset
    lr = LinearRegression()
    lr.fit(df[['date']], df['value'])

    # Add a date input widget to the app
    input_date_str = st.text_input('Enter a date in the format YYYY-MM-DD')
    input_date = dt.datetime.strptime(input_date_str, '%Y-%m-%d')
    input_date_ordinal = dt.datetime.toordinal(input_date)

    # Predict a value for the input date using the trained model
    predicted_value = lr.predict([[input_date_ordinal]])

    # Output the predicted value to the app
    st.write('Predicted value for {}: {}'.format(input_date_str, predicted_value[0]))
