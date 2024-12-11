import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds
import altair as alt

@st.cache_data
def objective_func(x, e, bp, bq):
    perc_qty_change = np.multiply(e, x)
    new_price = bp + np.multiply(bp, x)
    new_qty = bq + np.multiply(perc_qty_change, bq)
    revenue = np.dot(new_price, new_qty)
    return -revenue

@st.cache_data
def investment(x, bp, bq):
    new_price = bp + np.multiply(bp, x)
    lm = bp - new_price
    investment = np.dot(lm, bq)
    return investment

# Initialize session state variables
for key in ['btn3', 'opt', 'opt_budget', 'slider_budget', 'opt_price_p']:
    if key not in st.session_state:
        st.session_state[key] = '' if key != 'btn3' else False

def callback1():
    st.session_state['btn3'] = True

if (
    'df' in st.session_state and isinstance(st.session_state.df, pd.DataFrame) and not st.session_state.df.empty and
    'elastic' in st.session_state and isinstance(st.session_state.elastic, pd.DataFrame) and not st.session_state.elastic.empty and
    'forecast' in st.session_state and isinstance(st.session_state.forecast, pd.DataFrame) and not st.session_state.forecast.empty
):
    st.title("Optimization Results")

    df = st.session_state.df
    e = st.session_state.elastic['Elasticities'].to_numpy()
    bp = df.loc[df.groupby(["ITEM"])["DATE"].idxmax()].PRICE.to_numpy()
    bq = st.session_state.forecast.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()
    st.session_state.slider_budget = round(int(np.dot(bp, bq)))
    budget = round(int(np.dot(bp, bq)))

    if st.session_state.opt_budget == '':
        max_budget = st.sidebar.slider("Budget:", 0, budget, int(0.3 * budget), step=10, format="$%d")
    else:
        max_budget = st.sidebar.slider("Budget:", 0, st.session_state.slider_budget, st.session_state.opt_budget, step=10, format="$%d")

    if st.session_state.opt_price_p == '':
        max_price = st.sidebar.slider("Maximum Price Reduction:", 0, 50, 20, step=5, format="%d%%")
    else:
        max_price = st.sidebar.slider("Maximum Price Reduction:", 0, 50, st.session_state.opt_price_p, step=5, format="%d%%")

    num_items = e.size

    if st.sidebar.button("Optimize", on_click=callback1):
        with st.spinner("Optimizing..."):
            st.session_state.opt_price_p = max_price
            st.session_state.opt_budget = max_budget
            st.session_state.opt = differential_evolution(
                objective_func,
                x0=-(max_price / 100) * np.ones(num_items) * 0.5,
                args=(e, bp, bq),
                bounds=Bounds(lb=-(max_price / 100) * np.ones(num_items), ub=np.zeros(num_items)),
                constraints=NonlinearConstraint(lambda x: investment(x, bp, bq), lb=0, ub=max_budget),
                seed=1234,
                maxiter=200,
                popsize=10,
                workers=-1
            )

    if st.session_state.btn3:
        if st.session_state.opt.success:
            # Print Results
            pass
else:
    st.title(":orange[Finish Previous Tabs!]")

