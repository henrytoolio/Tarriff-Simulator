import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

@st.cache_data
def simulate_price_increase(x, e, bp, bq, bc, tariff_pct):
    perc_qty_change = np.multiply(e, x)
    new_price = bp + np.multiply(bp, x)
    new_qty = bq + np.multiply(perc_qty_change, bq)

    # Apply tariff to cost
    new_cost = bc * (1 + tariff_pct / 100)

    # Revenue and baseline revenue
    sim_revenue = np.dot(new_price, new_qty)
    baseline_revenue = np.dot(bp, bq)

    # Margin calculations
    sim_margin = np.dot(new_price - new_cost, new_qty)
    baseline_margin = np.dot(bp - bc, bq)

    baseline_qty = np.sum(bq)
    sim_qty = np.sum(new_qty)

    return [baseline_revenue, sim_revenue, baseline_qty, sim_qty, sim_margin, baseline_margin, new_price, new_cost]

# Initialize session state variables
if 'btn2' not in st.session_state:
    st.session_state['btn2'] = False

if 'sim' not in st.session_state:
    st.session_state.sim = ''

if 'user_p' not in st.session_state:
    st.session_state.user_p = ''

def callback1():
    st.session_state['btn2'] = True

if 'df' in st.session_state and 'elastic' in st.session_state and 'forecast' in st.session_state:
    st.title("Simulation Results")

    df = st.session_state.df

    e = st.session_state.elastic['Elasticities'].to_numpy()
    bp = df.loc[df.groupby(["ITEM"])["DATE"].idxmax()].PRICE.to_numpy()
    bc = df.loc[df.groupby(["ITEM"])["DATE"].idxmax()]["unit_cost"].to_numpy()
    bq = st.session_state.forecast.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()
    num_items = e.size

    if st.session_state.user_p == '':
        max_price = st.sidebar.slider("Price Increase for Tariff:", 0, 50, 20, step=5, help="Price increase per item", format="%d%%")
    else:
        max_price = st.sidebar.slider("Price Increase for Tariff:", 0, 50, st.session_state.user_p, step=5, help="Price increase per item", format="%d%%")

    current_tariff = st.sidebar.number_input(
        "Current Tariff (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
    )

    max_margin_impact_pct = st.sidebar.number_input(
        "Maximum Margin Impact (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
        help="Maximum allowed gain/loss in total margin as a percent of baseline margin"
    )

    if st.sidebar.button("Calculate", on_click=callback1):
        with st.spinner("Please Wait..."):
            user_price = np.ones(num_items) * (max_price / 100)
            sim_result = simulate_price_increase(user_price, e, bp, bq, bc, current_tariff)

            sim_margin = sim_result[4]
            baseline_margin = sim_result[5]
            margin_impact_pct = ((sim_margin - baseline_margin) / baseline_margin) * 100

            if abs(margin_impact_pct) > max_margin_impact_pct:
                st.warning(f"Scenario exceeds the allowed margin impact of {max_margin_impact_pct:.1f}%. "
                           f"Actual impact: {margin_impact_pct:.1f}%. Try reducing the price increase.")
                st.stop()
            else:
                st.session_state.sim = sim_result
                st.session_state.user_p = max_price

    if st.session_state.btn2 and st.session_state.sim != '':
        panel1 = st.container()
        panel2 = st.container()

        baseline_revenue, sim_revenue, baseline_qty, new_qty, sim_margin, baseline_margin, new_price, new_cost = st.session_state.sim

        with panel1:
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Baseline Revenue", value=f"${round(baseline_revenue)}")
            col2.metric(label="Simulated Revenue", value=f"${round(sim_revenue)}")
            col3.metric(
                label="Revenue Change",
                value=f"${round(sim_revenue - baseline_revenue)}",
                delta=f"{round(((sim_revenue / baseline_revenue) - 1) * 100, 1)}%"
            )

        with panel2:
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Baseline Qty", value=f"{round(baseline_qty)}")
            col2.metric(label="Simulated Qty", value=f"{round(new_qty)}")
            col3.metric(
                label="% Qty Change",
                value=f"{round(new_qty - baseline_qty)}",
                delta=f"{round(((new_qty / baseline_qty) - 1) * 100, 1)}%"
            )

        st.subheader(f"Margin Change from {st.session_state.user_p}% Price Increase: ${round(sim_margin - baseline_margin)}")
        st.markdown(f"**Current Tariff:** {current_tariff:.1f}%")
        st.markdown(f"**Margin Impact:** {round(margin_impact_pct, 2)}%")

        st.markdown("#### Item Price Change")
        chart_data_2 = pd.DataFrame({
            'Item': st.session_state.elastic['ITEM'],
            'Base Price': np.around(bp, 2),
            'New Price': np.around(new_price, 2)
        })

        chart2 = alt.Chart(chart_data_2.melt('Item')).mark_bar().encode(
            alt.Y('variable:N', axis=alt.Axis(title='')),
            alt.X('value:Q', axis=alt.Axis(title='Price', grid=False, format='$.2f')),
            color=alt.Color('variable:N'),
            row=alt.Row('Item:O', header=alt.Header(labelAngle=0, labelAlign='left'))
        ).configure_view(stroke='transparent')

        st.altair_chart(chart2, theme="streamlit", use_container_width=True)

        st.download_button(
            label="Download",
            data=chart_data_2.to_csv(index=False).encode('utf-8'),
            file_name='scenario_price_change.csv',
            mime='text/csv',
        )
else:
    st.title(":orange[Finish Previous Tabs!]")
