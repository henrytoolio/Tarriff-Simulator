import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Simulate weekly demand after price change
def simulate_weekly_demand(df_forecast, elasticity_dict, price_increase_pct):
    df = df_forecast.copy()
    df['Elasticity'] = df['ITEM'].map(elasticity_dict)
    df['Pct_Change'] = price_increase_pct / 100  # scalar or dict of ITEM → pct
    df['Adj_Factor'] = 1 + (df['Elasticity'] * df['Pct_Change'])
    df['Adj_Units'] = df['UNIT_FORECAST'] * df['Adj_Factor']
    return df

def optimize_price_for_revenue(e, bp, bq, bc, tariff_pct, max_margin_loss_pct, step=0.5):
    best_revenue = -np.inf
    best_increase = 0
    base_margin = np.dot(bp - bc, bq)

    for pct in np.arange(0, 100 + step, step):
        x = np.ones_like(bp) * (pct / 100)
        new_price = bp + bp * x
        new_qty = bq + bq * e * x
        new_cost = bc * (1 + tariff_pct / 100)
        
        sim_margin = np.dot(new_price - new_cost, new_qty)
        margin_delta_pct = ((sim_margin - base_margin) / base_margin) * 100

        if margin_delta_pct >= -max_margin_loss_pct:
            revenue = np.dot(new_price, new_qty)
            if revenue > best_revenue:
                best_revenue = revenue
                best_increase = pct

    return best_increase
# Streamlit app
st.title("📊 Price Recommendation with Weekly Forecast Simulation")

if 'df' in st.session_state and 'elastic' in st.session_state and 'forecast' in st.session_state:
    df = st.session_state.df
    forecast_df = st.session_state.forecast
    elasticity = st.session_state.elastic

    latest_df = df.loc[df.groupby("ITEM")["DATE"].idxmax()]
    bp = latest_df["PRICE"].to_numpy()
    bc = latest_df["Unit_cost"].to_numpy()
    e = elasticity['Elasticities'].to_numpy()
    items = elasticity['ITEM'].to_numpy()
    bq = forecast_df.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()

    elasticity_dict = dict(zip(elasticity['ITEM'], elasticity['Elasticities']))

    tariff_pct = st.sidebar.number_input("Tariff Increase (%)", 0.0, 100.0, 5.0, 0.5)
    max_margin_loss_pct = st.sidebar.number_input("Max Margin Loss (%)", 0.0, 100.0, 5.0, 0.5)

    if st.sidebar.button("Recommend Price Increase"):
        price_increase_pct = optimize_price_for_revenue(e, bp, bq, bc, tariff_pct, max_margin_loss_pct)
        st.success(f"Recommended Price Increase: {price_increase_pct:.2f}%")

        # Simulate weekly demand after applying optimal price increase
        weekly_sim = simulate_weekly_demand(forecast_df, elasticity_dict, price_increase_pct)

        st.subheader("📆 Simulated Weekly Demand After Price Increase")
        st.dataframe(weekly_sim[['ITEM', 'DATE', 'UNIT_FORECAST', 'Adj_Units']], use_container_width=True)

        chart = alt.Chart(weekly_sim).mark_line(point=True).encode(
            x='DATE:T', y='Adj_Units:Q', color='ITEM:N'
        ).properties(title="Adjusted Weekly Forecast After Price Increase")

        st.altair_chart(chart, use_container_width=True)
        st.download_button("Download Adjusted Forecast", data=weekly_sim.to_csv(index=False).encode('utf-8'),
                           file_name="adjusted_weekly_forecast.csv", mime="text/csv")
else:
    st.warning("⚠️ Please upload your data and run the forecast and elasticity steps first.")
