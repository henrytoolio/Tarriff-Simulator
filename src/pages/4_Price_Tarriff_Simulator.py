import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.optimize import minimize

# --- Simulate Adjusted Demand ---
def simulate_weekly_demand(df_forecast, elasticity_dict, price_increase_dict):
    df = df_forecast.copy()
    df['Elasticity'] = df['ITEM'].map(elasticity_dict)
    df['Pct_Change'] = df['ITEM'].map(price_increase_dict) / 100
    df['PRICE'] = df['PRICE'].astype(float)
    df['Adj_Price'] = df['PRICE'] * (1 + df['Pct_Change'])
    df['Adj_Units'] = df['UNIT_FORECAST'] * (df['Adj_Price'] / df['PRICE']) ** df['Elasticity']
    return df

# --- Optimizer: Maximize Profit considering tariff and elasticity ---
def optimize_price_for_profit(
    e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
    step=0.5, max_price_increase_pct=50
):
    # Convert inputs to numpy arrays if they aren't already
    e = np.array(e)
    bp = np.array(bp)
    bq = np.array(bq)
    bc = np.array(bc)
    
    # Calculate base values with tariff
    bc_tariff = bc * (1 + tariff_pct / 100)
    base_revenue = np.dot(bp, bq)
    base_cost = np.dot(bc_tariff, bq)
    base_profit = base_revenue - base_cost
    
    def objective(x):
        # x is the price increase percentage for each item
        x = x / 100  # Convert to decimal
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        new_revenue = np.dot(new_price, new_qty)
        new_cost = np.dot(bc_tariff, new_qty)
        new_profit = new_revenue - new_cost
        
        # We want to maximize profit, so minimize negative profit
        return -new_profit
    
    def constraint(x):
        # Constraint to ensure margin doesn't drop below max_margin_loss_pct
        x = x / 100
        new_price = bp * (1 + x)
        new_qty = bq * (new_price / bp) ** e
        new_revenue = np.dot(new_price, new_qty)
        new_cost = np.dot(bc_tariff, new_qty)
        new_profit = new_revenue - new_cost
        
        margin_delta_pct = ((new_profit - base_profit) / base_profit) * 100
        return margin_delta_pct + max_margin_loss_pct  # Must be >= 0
    
    # Initial guess: equal price increases for all items
    x0 = np.ones(len(e)) * (max_price_increase_pct / 2)
    
    # Bounds for each item's price increase
    bounds = [(0, max_price_increase_pct) for _ in range(len(e))]
    
    # Constraints
    cons = [{'type': 'ineq', 'fun': constraint}]
    
    # Optimize
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=cons,
        method='SLSQP'
    )
    
    if result.success:
        return result.x
    else:
        st.error("Optimization failed. Please try different constraints.")
        return np.zeros(len(e))

# --- Streamlit App ---
st.set_page_config(page_title="Price Optimization", layout="wide")
st.title("📈 Price Optimization: Maximize Profit Under Margin Constraint")

if st.session_state.get('df') is not None and st.session_state.get('elastic') is not None and st.session_state.get('forecast') is not None:
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

    if st.sidebar.checkbox("Show Elasticity Curves"):
        st.subheader("🔄 Elasticity Curves by Product")

        elasticity_df = pd.DataFrame({"ITEM": items, "Elasticity": e})
        elasticity_df = elasticity_df.sort_values(by="Elasticity")

        selected_items = st.multiselect(
            "Select products to visualize:",
            options=elasticity_df['ITEM'].tolist(),
            default=elasticity_df['ITEM'].head(5).tolist()
        )

        price_range = np.linspace(0.5, 2.0, 50)
        curves = []
        for item in selected_items:
            idx = np.where(items == item)[0][0]
            elasticity_val = e[idx]
            base_qty = bq[idx]
            base_price = bp[idx]

            for pr in price_range:
                curves.append({
                    "ITEM": item,
                    "Price Multiplier": pr,
                    "Price": base_price * pr,
                    "Demand": base_qty * pr ** elasticity_val
                })

        curves_df = pd.DataFrame(curves)

        elasticity_chart = alt.Chart(curves_df).mark_line().encode(
            x="Price",
            y="Demand",
            color="ITEM"
        ).properties(title="Price Elasticity Curves")

        st.altair_chart(elasticity_chart, use_container_width=True)

    if st.sidebar.button("Recommend Price Increases"):
        price_increases = optimize_price_for_profit(
            e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
            step=0.5, max_price_increase_pct=max_price_increase_pct
        )
        
        price_increase_dict = dict(zip(items, price_increases))
        
        # Calculate new prices and quantities
        new_prices = bp * (1 + price_increases / 100)
        new_quantities = bq * (new_prices / bp) ** e
        bc_tariff = bc * (1 + tariff_pct / 100)
        
        # Calculate financial metrics
        base_revenue = np.dot(bp, bq)
        base_cost = np.dot(bc_tariff, bq)
        base_profit = base_revenue - base_cost
        
        new_revenue = np.dot(new_prices, new_quantities)
        new_cost = np.dot(bc_tariff, new_quantities)
        new_profit = new_revenue - new_cost
        
        profit_delta = new_profit - base_profit
        
        # Display results
        st.success("✅ Price Increase Recommendations Generated")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Item': items,
            'Current Price': bp,
            'Recommended Price': new_prices,
            'Price Increase %': price_increases,
            'Current Quantity': bq,
            'Projected Quantity': new_quantities,
            'Unit Cost (w/ Tariff)': bc_tariff
        })
        
        st.markdown("### 💰 Financial Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue (Before)", f"${base_revenue:,.2f}")
        col1.metric("Revenue (After)", f"${new_revenue:,.2f}", f"{((new_revenue - base_revenue)/base_revenue)*100:.2f}%")
        
        col2.metric("Profit (Before)", f"${base_profit:,.2f}")
        col2.metric("Profit (After)", f"${new_profit:,.2f}", f"{profit_delta/base_profit*100:.2f}%")
        
        col3.metric("Margin Δ (%)", f"{((new_profit - base_profit)/base_profit)*100:.2f}%")
        col3.metric("Average Unit Cost (w/ Tariff)", f"${bc_tariff.mean():.2f}")
        
        st.markdown("### 📊 Price Increase Recommendations")
        st.dataframe(results_df.style.format({
            'Current Price': '${:.2f}',
            'Recommended Price': '${:.2f}',
            'Price Increase %': '{:.2f}%',
            'Current Quantity': '{:.0f}',
            'Projected Quantity': '{:.0f}',
            'Unit Cost (w/ Tariff)': '${:.2f}'
        }))
        
        # Simulate weekly demand
        weekly_sim = simulate_weekly_demand(forecast_df, elasticity_dict, price_increase_dict)
        
        st.subheader("📅 Simulated Weekly Demand After Price Increase")
        st.dataframe(weekly_sim[['ITEM', 'DATE', 'UNIT_FORECAST', 'Adj_Units']], use_container_width=True)
        
        chart = alt.Chart(weekly_sim).mark_line(point=True).encode(
            x='DATE:T', y='Adj_Units:Q', color='ITEM:N'
        ).properties(title="Adjusted Weekly Forecast After Price Increase")
        
        st.altair_chart(chart, use_container_width=True)
        
        st.download_button(
            "Download Results",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name="price_optimization_results.csv",
            mime="text/csv"
        )
else:
    st.warning("⚠️ Please upload your data and run the forecast and elasticity steps first.")
