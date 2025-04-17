import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.optimize import minimize


@st.cache_data
def simulate_price_increase(x, e, bp, bq):
    perc_qty_change = np.multiply(e, x)  # elasticity * % price change
    new_price = bp + np.multiply(bp, x)  # increased price
    new_qty = bq + np.multiply(perc_qty_change, bq)  # expected demand drop
    sim_revenue = np.dot(new_price, new_qty)
    baseline_revenue = np.dot(bp, bq)
    baseline_qty = sum(bq)
    sim_qty = sum(new_qty)
    margin_gain = np.dot(new_price - bp, new_qty)  # revenue gain from price increase
    return [baseline_revenue, sim_revenue, baseline_qty, sim_qty, margin_gain, new_price]

@st.cache_data
def optimize_margin(x, e, bp, bq):
    # Convert x from percentage to decimal
    x = x / 100
    perc_qty_change = np.multiply(e, x)
    new_price = bp + np.multiply(bp, x)
    new_qty = bq + np.multiply(perc_qty_change, bq)
    margin = np.dot(new_price - bp, new_qty)
    # We negate margin because we want to maximize it
    return -margin

#test

# Session state variables
if 'btn2' not in st.session_state:
    st.session_state['btn2'] = False

if 'sim' not in st.session_state:
    st.session_state.sim = ''

if 'user_p' not in st.session_state:
    st.session_state.user_p = ''

def callback1():
    st.session_state['btn2'] = True

if st.session_state.df is not '' and st.session_state.elastic is not '' and st.session_state.forecast is not '':
    st.title("Price Optimization Results")

    df = st.session_state.df

    e = st.session_state.elastic['Elasticities'].to_numpy()
    bp = df.loc[df.groupby(["ITEM"])["DATE"].idxmax()].PRICE.to_numpy()
    bq = st.session_state.forecast.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()
    num_items = e.size

    # Price increase constraints
    st.sidebar.markdown("### Optimization Constraints")
    max_price_increase = st.sidebar.slider("Maximum Price Increase:", 0, 50, 20, step=5, help="Maximum allowed price increase per item", format="%d%%")
    min_price_increase = st.sidebar.slider("Minimum Price Increase:", 0, 50, 0, step=5, help="Minimum required price increase per item", format="%d%%")

    if st.sidebar.button("Optimize Prices", on_click=callback1):
        with st.spinner("Optimizing prices for maximum margin..."):
            # Initial guess: equal price increases for all items
            x0 = np.ones(num_items) * ((max_price_increase + min_price_increase) / 2)
            
            # Bounds for each item's price increase
            bounds = [(min_price_increase, max_price_increase) for _ in range(num_items)]
            
            # Optimize
            result = minimize(
                optimize_margin,
                x0,
                args=(e, bp, bq),
                bounds=bounds,
                method='SLSQP'
            )
            
            if result.success:
                optimal_price_increases = result.x
                st.session_state.sim = simulate_price_increase(optimal_price_increases/100, e, bp, bq)
                st.session_state.user_p = optimal_price_increases
            else:
                st.error("Optimization failed. Please try different constraints.")

    if st.session_state.btn2:
        panel1 = st.container()
        panel2 = st.container()

        baseline_revenue = st.session_state.sim[0]
        sim_revenue = st.session_state.sim[1]
        baseline_qty = st.session_state.sim[2]
        new_qty = st.session_state.sim[3]
        margin_gain = st.session_state.sim[4]
        new_price = st.session_state.sim[5]

        with panel1:
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Baseline Revenue", value=f"${round(baseline_revenue)}")
            col2.metric(label="Optimized Revenue", value=f"${round(sim_revenue)}")
            col3.metric(label="Revenue Change", value=f"${round(sim_revenue - baseline_revenue)}", delta=f"{round(((sim_revenue / baseline_revenue) - 1) * 100, 1)}%")

        with panel2:
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Baseline Qty", value=f"{round(baseline_qty)}")
            col2.metric(label="Optimized Qty", value=f"{round(new_qty)}")
            col3.metric(label="% Qty Change", value=f"{round(new_qty - baseline_qty)}", delta=f"{round(((new_qty / baseline_qty) - 1) * 100, 1)}%")

        st.subheader(f"Margin Gain from Optimized Prices: ${round(margin_gain)}")

        st.markdown("#### Item Price Changes")
        chart_data_2 = pd.DataFrame({
            'Item': st.session_state.elastic['ITEM'],
            'Base Price': np.around(bp, 2),
            'Optimized Price': np.around(new_price, 2),
            'Price Increase %': np.around(st.session_state.user_p, 1)
        })

        chart2 = alt.Chart(chart_data_2.melt('Item', ['Base Price', 'Optimized Price'])).mark_bar().encode(
            alt.Y('variable:N', axis=alt.Axis(title='')),
            alt.X('value:Q', axis=alt.Axis(title='Price', grid=False, format='$.2f')),
            color=alt.Color('variable:N'),
            row=alt.Row('Item:O', header=alt.Header(labelAngle=0, labelAlign='left'))
        ).configure_view(stroke='transparent')

        st.altair_chart(chart2, theme="streamlit", use_container_width=True)

        # Display price increase percentages
        st.markdown("#### Price Increase Percentages")
        st.dataframe(chart_data_2[['Item', 'Price Increase %']].style.format({'Price Increase %': '{:.1f}%'}))

        st.download_button(
            label="Download Results",
            data=chart_data_2.to_csv(index=False).encode('utf-8'),
            file_name='optimized_price_changes.csv',
            mime='text/csv',
        )
else:
    st.title(":orange[Finish Previous Tabs!]")
