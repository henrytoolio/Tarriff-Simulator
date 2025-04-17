import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy.optimize import minimize

# --- Simulate Adjusted Demand ---
def simulate_weekly_demand(df_forecast, elasticity_dict, price_increase_dict):
    try:
        df = df_forecast.copy()
        df['Elasticity'] = df['ITEM'].map(elasticity_dict)
        df['Pct_Change'] = df['ITEM'].map(price_increase_dict) / 100
        df['PRICE'] = df['PRICE'].astype(float)
        df['Adj_Price'] = df['PRICE'] * (1 + df['Pct_Change'])
        df['Adj_Units'] = df['UNIT_FORECAST'] * (df['Adj_Price'] / df['PRICE']) ** df['Elasticity']
        return df
    except Exception as e:
        st.error(f"Error in demand simulation: {str(e)}")
        return None

# --- Optimizer: Maximize Profit considering tariff and elasticity ---
def optimize_price_for_profit(
    e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
    step=0.5, max_price_increase_pct=50
):
    try:
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
    except Exception as e:
        st.error(f"Error in optimization: {str(e)}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Price Optimization", layout="wide")
st.title("📈 Price Optimization: Maximize Profit Under Margin Constraint")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'elastic' not in st.session_state:
    st.session_state.elastic = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# Check if required data exists
if st.session_state.df is None or st.session_state.elastic is None or st.session_state.forecast is None:
    st.warning("⚠️ Please upload your data and run the forecast and elasticity steps first.")
    st.stop()

try:
    df = st.session_state.df
    forecast_df = st.session_state.forecast.copy()
    elasticity = st.session_state.elastic

    # Validate required columns
    required_columns = ['ITEM', 'DATE', 'PRICE', 'Unit_cost']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns in data. Required: {required_columns}")
        st.stop()

    # Get latest prices and costs
    latest_df = df.loc[df.groupby("ITEM")["DATE"].idxmax()]
    bp = latest_df["PRICE"].astype(float).to_numpy()
    bc = latest_df["Unit_cost"].astype(float).to_numpy()
    
    # Handle elasticity data
    if 'Elasticities' not in elasticity.columns:
        st.error("Missing 'Elasticities' column in elasticity data")
        st.stop()
        
    e_raw = elasticity['Elasticities'].to_numpy()
    e = np.where(e_raw > 0, -e_raw, e_raw)
    items = elasticity['ITEM'].to_numpy()
    
    # Get forecast quantities
    if 'UNIT_FORECAST' not in forecast_df.columns:
        st.error("Missing 'UNIT_FORECAST' column in forecast data")
        st.stop()
        
    bq = forecast_df.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()

    # Create mappings
    elasticity_dict = dict(zip(elasticity['ITEM'], e))
    item_price_map = dict(zip(latest_df['ITEM'], latest_df['PRICE']))
    forecast_df['PRICE'] = forecast_df['ITEM'].map(item_price_map)

    # Sidebar controls
    st.sidebar.markdown("### 🎯 Optimization Parameters")
    tariff_pct = st.sidebar.number_input("Tariff Increase (%)", 0.0, 100.0, 5.0, 0.5)
    max_margin_loss_pct = st.sidebar.number_input("Max Margin Loss (%)", 0.0, 100.0, 5.0, 0.5)
    max_price_increase_pct = st.sidebar.number_input("Max Price Increase Cap (%)", 0.0, 100.0, 50.0, 1.0)

    # Scenario management
    st.sidebar.markdown("### 📊 Scenario Management")
    scenario_name = st.sidebar.text_input("Scenario Name", "Scenario 1")
    
    if st.sidebar.button("Save Current Scenario"):
        price_increases = optimize_price_for_profit(
            e, bp, bq, bc, tariff_pct, max_margin_loss_pct,
            step=0.5, max_price_increase_pct=max_price_increase_pct
        )
        
        if price_increases is not None:
            # Calculate scenario results
            new_prices = bp * (1 + price_increases / 100)
            new_quantities = bq * (new_prices / bp) ** e
            bc_tariff = bc * (1 + tariff_pct / 100)
            
            base_revenue = np.dot(bp, bq)
            base_cost = np.dot(bc_tariff, bq)
            base_profit = base_revenue - base_cost
            
            new_revenue = np.dot(new_prices, new_quantities)
            new_cost = np.dot(bc_tariff, new_quantities)
            new_profit = new_revenue - new_cost
            
            # Save scenario
            scenario = {
                'name': scenario_name,
                'tariff_pct': tariff_pct,
                'max_margin_loss_pct': max_margin_loss_pct,
                'max_price_increase_pct': max_price_increase_pct,
                'price_increases': price_increases,
                'new_prices': new_prices,
                'new_quantities': new_quantities,
                'base_revenue': base_revenue,
                'new_revenue': new_revenue,
                'base_profit': base_profit,
                'new_profit': new_profit,
                'bc_tariff': bc_tariff
            }
            
            st.session_state.scenarios.append(scenario)
            st.sidebar.success(f"✅ Scenario '{scenario_name}' saved!")

    # Display saved scenarios
    if st.session_state.scenarios:
        st.markdown("### 📊 Saved Scenarios")
        
        # Create comparison table
        comparison_data = []
        for scenario in st.session_state.scenarios:
            comparison_data.append({
                'Scenario': scenario['name'],
                'Tariff (%)': scenario['tariff_pct'],
                'Max Margin Loss (%)': scenario['max_margin_loss_pct'],
                'Max Price Increase (%)': scenario['max_price_increase_pct'],
                'Revenue ($)': scenario['new_revenue'],
                'Profit ($)': scenario['new_profit'],
                'Revenue Change (%)': ((scenario['new_revenue'] - scenario['base_revenue']) / scenario['base_revenue']) * 100,
                'Profit Change (%)': ((scenario['new_profit'] - scenario['base_profit']) / scenario['base_profit']) * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.format({
            'Revenue ($)': '${:,.2f}',
            'Profit ($)': '${:,.2f}',
            'Revenue Change (%)': '{:.2f}%',
            'Profit Change (%)': '{:.2f}%',
            'Tariff (%)': '{:.1f}%',
            'Max Margin Loss (%)': '{:.1f}%',
            'Max Price Increase (%)': '{:.1f}%'
        }))
        
        # Visualize scenarios
        st.markdown("### 📈 Scenario Comparison")
        
        # Revenue comparison
        revenue_chart = alt.Chart(pd.DataFrame(comparison_data)).mark_bar().encode(
            x='Scenario:N',
            y='Revenue ($):Q',
            color='Scenario:N'
        ).properties(title="Revenue by Scenario")
        
        # Profit comparison
        profit_chart = alt.Chart(pd.DataFrame(comparison_data)).mark_bar().encode(
            x='Scenario:N',
            y='Profit ($):Q',
            color='Scenario:N'
        ).properties(title="Profit by Scenario")
        
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(revenue_chart, use_container_width=True)
        with col2:
            st.altair_chart(profit_chart, use_container_width=True)
        
        # Allow user to select a scenario to view details
        selected_scenario = st.selectbox(
            "Select a scenario to view details:",
            options=[s['name'] for s in st.session_state.scenarios]
        )
        
        if selected_scenario:
            scenario = next(s for s in st.session_state.scenarios if s['name'] == selected_scenario)
            
            st.markdown(f"### 📊 {selected_scenario} Details")
            
            # Create detailed results DataFrame
            results_df = pd.DataFrame({
                'Item': items,
                'Current Price': bp,
                'Recommended Price': scenario['new_prices'],
                'Price Increase %': scenario['price_increases'],
                'Current Quantity': bq,
                'Projected Quantity': scenario['new_quantities'],
                'Unit Cost (w/ Tariff)': scenario['bc_tariff']
            })
            
            st.dataframe(results_df.style.format({
                'Current Price': '${:.2f}',
                'Recommended Price': '${:.2f}',
                'Price Increase %': '{:.2f}%',
                'Current Quantity': '{:.0f}',
                'Projected Quantity': '{:.0f}',
                'Unit Cost (w/ Tariff)': '${:.2f}'
            }))
            
            # Show weekly demand simulation
            price_increase_dict = dict(zip(items, scenario['price_increases']))
            weekly_sim = simulate_weekly_demand(forecast_df, elasticity_dict, price_increase_dict)
            
            if weekly_sim is not None:
                st.subheader("📅 Simulated Weekly Demand")
                st.dataframe(weekly_sim[['ITEM', 'DATE', 'UNIT_FORECAST', 'Adj_Units']], use_container_width=True)
                
                chart = alt.Chart(weekly_sim).mark_line(point=True).encode(
                    x='DATE:T', y='Adj_Units:Q', color='ITEM:N'
                ).properties(title="Adjusted Weekly Forecast")
                
                st.altair_chart(chart, use_container_width=True)
            
            # Download button for selected scenario
            st.download_button(
                f"Download {selected_scenario} Results",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_scenario.lower().replace(' ', '_')}_results.csv",
                mime="text/csv"
            )

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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()
