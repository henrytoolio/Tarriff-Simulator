import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- Demand Simulation ---
def simulate_weekly_demand(df_forecast, elasticity_dict, price_increase_pct):
    df = df_forecast.copy()
    df['Elasticity'] = df['ITEM'].map(elasticity_dict)
    df['Pct_Change'] = price_increase_pct / 100
    df['PRICE'] = df['PRICE'].astype(float)
    df['Adj_Price'] = df['PRICE'] * (1 + df['Pct_Change'])
    df['Adj_Units'] = df['UNIT_FORECAST'] * (df['Adj_Price'] / df['PRICE']) ** df['Elasticity']
    return df

# --- Optimizer: Maximize revenue under margin constraint ---
def optimize_price_for_revenue(e, bp, bq, bc, tariff_pct, max_margin_loss_pct, step=0.5, max_price_increase_pct=50):
    best_revenue = -np.inf
    best_increase = 0
    base_margin = np.dot(bp - bc, bq)
    bc_new = bc * (1 + tariff_pct / 100)

    for pct in np.arange(0, max_price_increase_pct + step, step):
        x = pct / 100
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        new_margin = np.dot(new_price - bc_new, new_qty)
        margin_delta_pct = ((new_margin - base_margin) / base_margin) * 100

        if margin_delta_pct >= -max_margin_loss_pct:
            revenue = np.dot(new_price, new_qty)
            if revenue > best_revenue:
                best_revenue = revenue
                best_increase = pct

    return best_increase

# --- Streamlit App ---
st.title("📊 Price Optimization: Maximize Revenue with Margin Constraints")

if 'df' in st.session_state and 'elastic' in st.session_state and 'forecast' in st.session_state:
    df = st.session_state.df
    forecast_df = st.session_state.forecast.copy()
    elasticity = st.session_state.elastic

    latest_df = df.loc[df.groupby("ITEM")["DATE"].idxmax()]
    bp = latest_df["PRICE"].astype(float).to_numpy()
    bc = latest_df["Unit_cost"].astype(float).to_numpy()
    e_raw = elasticity['Elasticities'].to_numpy()

    # Ensure elasticities are negative
    e = np.where(e_raw > 0, -e_raw, e_raw)

    items = elasticity['ITEM'].to_numpy()
    bq = forecast_df.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()

    elasticity_dict = dict(zip(elasticity['ITEM'], e))
    item_price_map = dict(zip(latest_df['ITEM'], latest_df['PRICE']))
    forecast_df['PRICE'] = forecast_df['ITEM'].map(item_price_map)

    # --- Sidebar Controls ---
    tariff_pct = st.sidebar.number_input("Tariff Increase (%)", 0.0, 100.0, 5.0, 0.5)
    max_margin_loss_pct = st.sidebar.number_input("Max Margin Loss (%)", 0.0, 100.0, 5.0, 0.5)
    max_price_increase_pct = st.sidebar.number_input("Max Price Increase Cap (%)", 0.0, 100.0, 50.0, 1.0)

    if st.sidebar.button("Recommend Price Increase"):
        price_increase_pct = optimize_price_for_revenue(
            e, bp, bq, bc, tariff_pct, max_margin_loss_pct, step=0.5, max_price_increase_pct=max_price_increase_pct
        )
        st.success(f"✅ Recommended Price Increase: **{price_increase_pct:.2f}%**")

        # --- Recompute Metrics ---
        x = price_increase_pct / 100
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        new_cost = bc * (1 + tariff_pct / 100)

        base_margin_total = np.dot(bp - bc, bq)
        new_margin_total = np.dot(new_price - new_cost, new_qty)
        base_revenue = np.dot(bp, bq)
        new_revenue = np.dot(new_price, new_qty)

        # --- Display Metrics ---
        st.markdown("### 💰 Margin Summary")
        st.metric("Original Total Margin", f"${base_margin_total:,.2f}")
        st.metric("New Total Margin", f"${new_margin_total:,.2f}")
        st.metric("Margin Δ (%)", f"{((new_margin_total - base_margin_total) / base_margin_total) * 100:.2f}%")

        st.markdown("### 📈 Revenue Summary")
        st.metric("Original Total Revenue", f"${base_revenue:,.2f}")
        st.metric("New Total Revenue", f"${new_revenue:,.2f}")
        st.metric("Revenue Δ (%)", f"{((new_revenue - base_revenue) / base_revenue) * 100:.2f}%")

        # --- Simulate Weekly Forecast ---
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

