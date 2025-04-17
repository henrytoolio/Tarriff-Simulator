import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- Simulate Adjusted Demand ---
def simulate_weekly_demand(df_forecast, elasticity_dict, price_increase_pct):
    df = df_forecast.copy()
    df['Elasticity'] = df['ITEM'].map(elasticity_dict)
    df['Pct_Change'] = price_increase_pct / 100
    df['PRICE'] = df['PRICE'].astype(float)
    df['Adj_Price'] = df['PRICE'] * (1 + df['Pct_Change'])
    df['Adj_Units'] = df['UNIT_FORECAST'] * (df['Adj_Price'] / df['PRICE']) ** df['Elasticity']
    return df

# --- Optimizer: Maximize Profit under Margin Constraint ---
def optimize_price_for_profit(
    e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
    step=0.5, max_price_increase_pct=50
):
    best_profit = -np.inf
    best_increase = None

    bc_tariff = bc * (1 + tariff_pct / 100)
    base_revenue = np.dot(bp, bq)
    base_cost = np.dot(bc, bq)
    base_profit = base_revenue - base_cost

    fallback_profit = -np.inf
    fallback_increase = 0

    for pct in np.arange(0, max_price_increase_pct + step, step):
        x = pct / 100
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        new_revenue = np.dot(new_price, new_qty)
        new_cost = np.dot(bc_tariff, new_qty)
        new_profit = new_revenue - new_cost

        margin_delta_pct = ((new_profit - base_profit) / base_profit) * 100
        is_valid = margin_delta_pct >= -max_margin_loss_pct

        if is_valid:
            if new_profit > best_profit:
                best_profit = new_profit
                best_increase = pct
        else:
            if new_profit > fallback_profit:
                fallback_profit = new_profit
                fallback_increase = pct

    return best_increase if best_increase is not None else fallback_increase

# --- Streamlit App ---
st.title("\ud83d\udcc8 Price Optimization: Maximize Profit Under Margin Constraint")

if 'df' in st.session_state and 'elastic' in st.session_state and 'forecast' in st.session_state:
    df = st.session_state.df
    forecast_df = st.session_state.forecast.copy()
    elasticity = st.session_state.elastic

    latest_df = df.loc[df.groupby("ITEM")["DATE"].idxmax()]
    bp = latest_df["PRICE"].astype(float).to_numpy()
    bc = latest_df["Unit_cost"].astype(float).to_numpy()
    e_raw = elasticity['Elasticities'].to_numpy()

    e = np.where(e_raw > 0, -e_raw, e_raw)
    items = elasticity['ITEM'].to_numpy()
    bq = forecast_df.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()

    elasticity_dict = dict(zip(elasticity['ITEM'], e))
    item_price_map = dict(zip(latest_df['ITEM'], latest_df['PRICE']))
    forecast_df['PRICE'] = forecast_df['ITEM'].map(item_price_map)

    tariff_pct = st.sidebar.number_input("Tariff Increase (%)", 0.0, 100.0, 5.0, 0.5)
    max_margin_loss_pct = st.sidebar.number_input("Max Margin Loss (%)", 0.0, 100.0, 5.0, 0.5)
    max_price_increase_pct = st.sidebar.number_input("Max Price Increase Cap (%)", 0.0, 100.0, 50.0, 1.0)

    if st.sidebar.button("Recommend Price Increase"):
        price_increase_pct = optimize_price_for_profit(
            e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
            step=0.5, max_price_increase_pct=max_price_increase_pct
        )

        st.success(f"\u2705 Recommended Price Increase: **{price_increase_pct:.2f}%**")

        x = price_increase_pct / 100
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        bc_tariff = bc * (1 + tariff_pct / 100)

        base_revenue = np.dot(bp, bq)
        base_cost = np.dot(bc, bq)
        base_profit = base_revenue - base_cost

        new_revenue = np.dot(new_price, new_qty)
        new_cost = np.dot(bc_tariff, new_qty)
        new_profit = new_revenue - new_cost

        profit_delta = new_profit - base_profit

        st.markdown("### \ud83d\udcb0 Financial Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue (Before)", f"${base_revenue:,.2f}")
        col1.metric("Revenue (After)", f"${new_revenue:,.2f}", f"{((new_revenue - base_revenue)/base_revenue)*100:.2f}%")

        col2.metric("Profit (Before)", f"${base_profit:,.2f}")
        col2.metric("Profit (After)", f"${new_profit:,.2f}", f"{profit_delta/base_profit*100:.2f}%")

        col3.metric("Margin \u0394 (%)", f"{((new_profit - base_profit)/base_profit)*100:.2f}%")
        col3.metric("Unit Cost (w/ Tariff)", f"${bc_tariff.mean():.2f}")

        # Simulated Weekly Demand
        weekly_sim = simulate_weekly_demand(forecast_df, elasticity_dict, price_increase_pct)
        st.subheader("\ud83d\uddd3\ufe0f Simulated Weekly Demand After Price Increase")
        st.dataframe(weekly_sim[['ITEM', 'DATE', 'UNIT_FORECAST', 'Adj_Units']], use_container_width=True)

        chart = alt.Chart(weekly_sim).mark_line(point=True).encode(
            x='DATE:T', y='Adj_Units:Q', color='ITEM:N'
        ).properties(title="Adjusted Weekly Forecast After Price Increase")

        st.altair_chart(chart, use_container_width=True)
        st.download_button("Download Adjusted Forecast", data=weekly_sim.to_csv(index=False).encode('utf-8'),
                           file_name="adjusted_weekly_forecast.csv", mime="text/csv")

else:
    st.warning("\u26a0\ufe0f Please upload your data and run the forecast and elasticity steps first.")
