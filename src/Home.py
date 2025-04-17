import pandas as pd
import streamlit as st
import numpy as np

# option = st.sidebar.radio("Pick Option", ('Upload File', 'Simulate Data'))

@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    return df
    

st.set_page_config(
    page_title="Tarriff Pricing Optimizer",
    menu_items={
        'About': "# Tarriff Price Optimizer "
    }
)

st.title("Tarriff Pricing Optimizer")
st.caption("App is designed to allow users to optimize pricing based on tarrif increases" )
st.markdown("### Getting Started:")
st.markdown(":orange[Upload file having weekly time series data containing item sold quantity, selling price and unit cost. Any relevant feature can also be included to improve price elasticities and demand forecast prediction]")
st.warning("The CSV file must have ITEM, DATE, UNITS, PRICE, Unit Cost as column names")
st.markdown("* ***Demand Forecast Tab***: Uses uploaded weekly historical sales data to create a 4 weeks of demand forecast, taking into account factors such as seasonality and trends")
st.markdown("* ***Price Elasticities Tab***: Estimates how changes in price will affect demand for a given item")
st.markdown("* ***Price Simulator Tab***: Simulate 'What-If' price change scenarios impact on demand and budget requirement")
st.markdown("* ***Tarriff Simulator Tab***: Leverage Demand Forecast, Tarrif input, margin constraints, and maximum price increase constraints ")
st.markdown("* ***Price Increase Scenario Player Tab***: Users can compare different scenarios using same constraints as Tarriff Simulator ")

def callback_upl():
    st.session_state['upl'] = True

if 'df' not in st.session_state:
    st.session_state['df'] = ''

if 'upl' not in st.session_state:
    st.session_state['upl'] = False

uploaded_file = st.sidebar.file_uploader("Upload file", type = ['csv'], on_change=callback_upl)
if uploaded_file is not None:
    df = get_data(uploaded_file)
    st.session_state["df"] = df

if st.session_state.upl:
    st.markdown("Sample of Uploaded Data:")
    st.dataframe(st.session_state.df.head(), use_container_width = True)
    st.write(f"Number of Unique Items: {st.session_state.df.ITEM.nunique()}")




    

