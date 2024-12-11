import streamlit as st
import pandas as pd
from prophet import Prophet
import altair as alt

# Function to forecast demand for a single DataFrame
def forecast_demand(df):
    df_prophet = df.copy()
    df_prophet = df_prophet.rename(columns={'DATE': 'ds', 'UNITS': 'y'})  # Rename columns for Prophet
    m = Prophet()  # Initialize Prophet model
    m.fit(df_prophet)  # Fit the model
    future = m.make_future_dataframe(periods=4, freq='W')  # Create future dataframe for 4 weeks
    forecast = m.predict(future)  # Generate predictions
    return forecast

# Function to forecast demand for all items
@st.cache_data  # Updated caching method (replacing st.experimental_memo)
def Prophet_Model_loop(df):
    all_demand = pd.DataFrame()
    progress = 0
    item_count = df['ITEM'].nunique()

    # Display progress bar
    if item_count > 0:
        my_bar = st.progress(0)
    else:
        st.error("No unique items found in the dataset!")
        return pd.DataFrame()

    # Loop through unique items
    for i in df['ITEM'].unique():
        try:
            temp = df[df['ITEM'] == i]  # Filter data for the current item
            demand = forecast_demand(temp)  # Forecast demand
            demand['ITEM'] = i  # Add item column to the forecast
            demand = demand[['ITEM', 'ds', 'yhat']].rename(columns={'ds': 'DATE', 'yhat': 'UNIT_FORECAST'})
            all_demand = pd.concat([all_demand, demand])  # Append to the result
        except Exception as e:
            st.error(f"Error processing ITEM: {i} - {e}")

        progress += 1
        my_bar.progress(progress / item_count)  # Update progress bar

    my_bar.empty()  # Clear the progress bar
    return all_demand

# Initialize session state variables
if 'forecast' not in st.session_state:
    st.session_state['forecast'] = pd.DataFrame()

if 'btn' not in st.session_state:
    st.session_state['btn'] = False

# Callback for button click
def callback1():
    st.session_state['btn'] = True

# Main application logic
if 'df' in st.session_state and not st.session_state['df'].empty:
    st.title("Demand Forecast")
    st.caption("Leverage weekly historical sales data to create a 4-week demand forecast, taking into account factors such as seasonality and trends.")

    df = st.session_state.df

    # Validate input DataFrame
    if not all(col in df.columns for col in ['DATE', 'UNITS', 'ITEM']):
        st.error("Uploaded file must include 'DATE', 'UNITS', and 'ITEM' columns.")
    else:
        if st.button("Forecast Demand", on_click=callback1):
            df['DATE'] = pd.to_datetime(df['DATE'])  # Ensure DATE column is in datetime format
            item_demand = Prophet_Model_loop(df)  # Run the forecasting loop
            st.session_state.forecast = item_demand

        if st.session_state.btn:
            # Plotting demand for a selected item
            plot_item = st.selectbox("Select an Item to Plot:", st.session_state.forecast['ITEM'].unique())
            title = alt.TitleParams(
                f"DEMAND FORECAST: {plot_item}",
                anchor='middle',
                subtitle='Orange: Forecast     Blue: Actual'
            )

            # Actual data chart
            chart1 = alt.Chart(st.session_state.df[st.session_state.df['ITEM'] == plot_item], title=title).mark_circle().encode(
                x='DATE',
                y=alt.Y('UNITS', title='Actual Units')
            )

            # Forecasted data chart
            chart2 = alt.Chart(st.session_state.forecast[st.session_state.forecast['ITEM'] == plot_item]).mark_line().encode(
                x='DATE',
                y=alt.Y('UNIT_FORECAST', title="Forecasted Units"),
                color=alt.value("#f35b04")
            )

            # Display the chart
            st.altair_chart(chart1 + chart2, theme="streamlit", use_container_width=True)

            # Display forecast table
            st.dataframe(st.session_state.forecast[st.session_state.forecast['ITEM'] == plot_item], use_container_width=True)

            # Download forecast data
            st.download_button(
                label="Download Forecast",
                data=st.session_state.forecast.to_csv(index=False).encode('utf-8'),
                file_name='demand_forecast.csv',
                mime='text/csv'
            )
else:
    st.title(":orange[Upload a File under the Home Tab!]")
