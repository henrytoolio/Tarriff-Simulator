import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Binary search optimizer
def find_min_price_increase(e, bp, bq, bc, tariff_pct, max_margin_loss_pct, precision=0.001):
    low, high = 0, 1  # search between 0% and 100% price increase
    new_price = bp.copy()
    while high - low > precision:
        mid = (low + high) / 2
        x = np.ones_like(bp) * mid

        # New price and quantity
        new_price = bp + (bp * x)
        new_qty = bq + (bq * e * x)

        # New cost with tariff
        new_cost = bc * (1 + tariff_pct / 100)

        sim_margin = np.dot(new_price - new_cost, new_qty)
        baseline_margin = np.dot(bp - bc, bq)

        margin_change_pct = ((sim_margin - baseline_margin) / baseline_margin) * 100

        if margin_change_pct >= -max_margin_loss_pct:
            high = mid
        else:
            low = mid

    return high * 100, new_price, new_cost, sim_margin, baseline_margin  # return % price increase

# Streamlit UI
st.title("📈 Price Recommendation Tool")

if 'df' in st.session_state and 'elastic' in st.session_state and 'forecast' in st.session_state:
    df = st.session_state.df

    # Extract data
    latest_df = df.loc[df.groupby(["ITEM"])["DATE"].idxmax()]
    bp = latest_df["PRICE"].to_numpy()
    bc = latest_df["Unit_cost"].to_numpy()
    e = st.session_state.elastic['Elasticities'].to_numpy()
    bq = st.session_state.forecast.groupby("ITEM").tail(4).groupby("ITEM")["UNIT_FORECAST"].sum().to_numpy()
    items = st.session_state.elastic['ITEM']

    # Inputs
    st.sidebar.markdown("### 💡 Inputs")
    tariff_pct = st.sidebar.number_input("Tariff Increase (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    max_margin_loss_pct = st.sidebar.number_input("Maximum Allowed Margin Loss (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    if st.sidebar.button("Recommend Prices"):
        with st.spinner("Optimizing price increase..."):
            price_increase_pct, new_price, new_cost, sim_margin, baseline_margin = find_min_price_increase(
                e, bp, bq, bc, tariff_pct, max_margin_loss_pct
            )

            sim_revenue = np.dot(new_price, bq)
            baseline_revenue = np.dot(bp, bq)
            revenue_change_pct = ((sim_revenue - baseline_revenue) / baseline_revenue) * 100
            margin_impact_pct = ((sim_margin - baseline_margin) / baseline_margin) * 100

            st.success(f"📊 Recommended Price Increase: **{price_increase_pct:.2f}%**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline Margin", f"${round(baseline_margin)}")
            col2.metric("Simulated Margin", f"${round(sim_margin)}")
            col3.metric("Margin Change", f"${round(sim_margin - baseline_margin)}", delta=f"{margin_impact_pct:.2f}%")

            col4, col5, col6 = st.columns(3)
            col4.metric("Baseline Revenue", f"${round(baseline_revenue)}")
            col5.metric("Simulated Revenue", f"${round(sim_revenue)}")
            col6.metric("Revenue Change", f"${round(sim_revenue - baseline_revenue)}", delta=f"{revenue_change_pct:.2f}%")

            st.subheader("📋 Item-Level Summary")
            results_df = pd.DataFrame({
                "Item": items,
                "Base Price": np.round(bp, 2),
                "New Price": np.round(new_price, 2),
                "Cost w/ Tariff": np.round(new_cost, 2)
            })

            st.dataframe(results_df, use_container_width=True)

            chart = alt.Chart(results_df.melt('Item')).mark_bar().encode(
                y=alt.Y('variable:N', title=''),
                x=alt.X('value:Q', title='USD', axis=alt.Axis(format='$')),
                color='variable:N',
                row=alt.Row('Item:N', header=alt.Header(labelAngle=0, labelAlign='left'))
            ).configure_view(stroke='transparent')

            st.altair_chart(chart, use_container_width=True)

            st.download_button("Download Results", data=results_df.to_csv(index=False).encode("utf-8"),
                               file_name="price_recommendations.csv", mime="text/csv")

else:
    st.warning("🚨 Please complete the earlier steps first (upload data, run elasticity & forecast tabs).")
