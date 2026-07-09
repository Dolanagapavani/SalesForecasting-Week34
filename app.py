import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📊")

# ── load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    return df

df = load_data()

monthly = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
monthly.columns = ['ds', 'y']

weekly = df.groupby(df['Order Date'].dt.to_period('W'))['Sales'].sum().reset_index()
weekly['Order Date'] = weekly['Order Date'].dt.to_timestamp()
weekly.columns = ['ds', 'y']

# ── sidebar nav ────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", [
    "Page 1 — Sales Overview",
    "Page 2 — Forecast Explorer",
    "Page 3 — Anomaly Report",
    "Page 4 — Product Segments"
])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Page 1 — Sales Overview":
    st.title("📈 Sales Overview Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df.shape[0]:,}")
    col3.metric("Avg Order Value", f"${df['Sales'].mean():.2f}")

    st.markdown("---")

    # yearly sales bar chart
    st.subheader("Total Sales by Year")
    yearly = df.groupby('Year')['Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly['Year'].astype(str), yearly['Sales'], color='steelblue', edgecolor='black')
    ax.set_title('Total Sales by Year')
    ax.set_ylabel('Sales ($)')
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'${bar.get_height():,.0f}', ha='center', fontsize=9)
    st.pyplot(fig)

    st.markdown("---")

    # monthly trend
    st.subheader("Monthly Sales Trend")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(monthly['ds'], monthly['y'], color='steelblue', linewidth=2, marker='o', markersize=3)
    ax2.set_title('Monthly Sales — 4 Years')
    ax2.set_ylabel('Sales ($)')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.markdown("---")

    # filters
    st.subheader("Sales by Region & Category")
    col_r, col_c = st.columns(2)
    selected_region = col_r.multiselect("Region", df['Region'].unique(), default=list(df['Region'].unique()))
    selected_cat = col_c.multiselect("Category", df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(selected_region) & df['Category'].isin(selected_cat)]

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    r_sales = filtered.groupby('Region')['Sales'].sum()
    r_sales.plot(kind='bar', ax=axes3[0], color='coral', edgecolor='black')
    axes3[0].set_title('Sales by Region'); axes3[0].set_ylabel('Sales ($)')

    c_sales = filtered.groupby('Category')['Sales'].sum()
    c_sales.plot(kind='bar', ax=axes3[1], color='mediumseagreen', edgecolor='black')
    axes3[1].set_title('Sales by Category')

    plt.tight_layout()
    st.pyplot(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Page 2 — Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("---")

    from prophet import Prophet
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    col1, col2 = st.columns(2)
    segment_type = col1.selectbox("Select Segment Type", ["Overall", "Category", "Region"])
    horizon = col2.slider("Forecast Horizon (months)", 1, 3, 3)

    if segment_type == "Overall":
        seg_data = monthly.copy()
        label = "Overall Sales"
    elif segment_type == "Category":
        cat = st.selectbox("Select Category", df['Category'].unique())
        seg = df[df['Category'] == cat]
        seg_monthly = seg.groupby(seg['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        seg_monthly['Order Date'] = seg_monthly['Order Date'].dt.to_timestamp()
        seg_monthly.columns = ['ds', 'y']
        seg_data = seg_monthly
        label = f"{cat} Sales"
    else:
        reg = st.selectbox("Select Region", df['Region'].unique())
        seg = df[df['Region'] == reg]
        seg_monthly = seg.groupby(seg['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        seg_monthly['Order Date'] = seg_monthly['Order Date'].dt.to_timestamp()
        seg_monthly.columns = ['ds', 'y']
        seg_data = seg_monthly
        label = f"{reg} Region Sales"

    if st.button("Run Forecast"):
        with st.spinner("Training Prophet model..."):
            train_data = seg_data[:-3]
            test_data = seg_data[-3:]

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(train_data)
            future = m.make_future_dataframe(periods=horizon, freq='MS')
            fc = m.predict(future)

            preds = fc[fc['ds'].isin(test_data['ds'])]['yhat'].values[:horizon]
            actuals = test_data['y'].values[:horizon]

            mae = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(seg_data['ds'], seg_data['y'], label='Actual', color='steelblue')
        ax.plot(fc['ds'].tail(horizon), fc['yhat'].tail(horizon), label='Forecast', color='red', linestyle='--', linewidth=2, marker='o')
        ax.fill_between(fc['ds'].tail(horizon), fc['yhat_lower'].tail(horizon), fc['yhat_upper'].tail(horizon), alpha=0.2, color='red')
        ax.set_title(f'{label} — {horizon}-Month Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("MAE", f"${mae:,.2f}")
        col_m2.metric("RMSE", f"${rmse:,.2f}")

        st.subheader("Forecasted Values")
        fc_table = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).round(2)
        fc_table.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        st.dataframe(fc_table)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Page 3 — Anomaly Report":
    st.title("⚠️ Anomaly Report")
    st.markdown("---")

    from sklearn.ensemble import IsolationForest

    weekly_data = weekly.copy().sort_values('ds').reset_index(drop=True)

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_data['iso_anomaly'] = iso.fit_predict(weekly_data[['y']])
    weekly_data['iso_anomaly'] = weekly_data['iso_anomaly'].map({1: 0, -1: 1})

    weekly_data['rolling_mean'] = weekly_data['y'].rolling(4, center=True).mean()
    weekly_data['rolling_std'] = weekly_data['y'].rolling(4, center=True).std()
    weekly_data['zscore'] = (weekly_data['y'] - weekly_data['rolling_mean']) / weekly_data['rolling_std']
    weekly_data['zscore_anomaly'] = (weekly_data['zscore'].abs() > 2).astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].plot(weekly_data['ds'], weekly_data['y'], color='steelblue', label='Weekly Sales', linewidth=1)
    iso_pts = weekly_data[weekly_data['iso_anomaly'] == 1]
    axes[0].scatter(iso_pts['ds'], iso_pts['y'], color='red', s=80, zorder=5, label='Anomaly')
    axes[0].set_title('Isolation Forest Anomalies'); axes[0].legend()

    axes[1].plot(weekly_data['ds'], weekly_data['y'], color='steelblue', label='Weekly Sales', linewidth=1)
    z_pts = weekly_data[weekly_data['zscore_anomaly'] == 1]
    axes[1].scatter(z_pts['ds'], z_pts['y'], color='darkorange', s=80, zorder=5, label='Anomaly')
    axes[1].set_title('Z-Score Anomalies'); axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Detected Anomaly Dates — Isolation Forest")
    st.dataframe(iso_pts[['ds', 'y']].rename(columns={'ds': 'Week', 'y': 'Sales'}).sort_values('Sales', ascending=False).reset_index(drop=True))

    col1, col2 = st.columns(2)
    col1.metric("Isolation Forest Anomalies", int(weekly_data['iso_anomaly'].sum()))
    col2.metric("Z-Score Anomalies", int(weekly_data['zscore_anomaly'].sum()))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRODUCT SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Page 4 — Product Demand Segments":
    st.title("🗂️ Product Demand Segments")
    st.markdown("---")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    subcat = df.groupby('Sub-Category').agg(
        total_sales=('Sales', 'sum'),
        avg_order=('Sales', 'mean'),
        volatility=('Sales', 'std')
    ).reset_index()

    yoy = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack()
    yoy['growth'] = (yoy[yoy.columns[-1]] - yoy[yoy.columns[0]]) / yoy[yoy.columns[0]] * 100
    subcat = subcat.merge(yoy[['growth']], on='Sub-Category').fillna(0)

    scaler = StandardScaler()
    X_clust = scaler.fit_transform(subcat[['total_sales', 'avg_order', 'volatility', 'growth']])

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    subcat['cluster'] = km.fit_predict(X_clust)

    labels = {0: 'High Volume, Stable', 1: 'Low Volume, High Volatility', 2: 'Growing Demand', 3: 'Declining/Niche'}
    subcat['Demand Segment'] = subcat['cluster'].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_clust)
    subcat['pca1'] = coords[:, 0]
    subcat['pca2'] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_c = ['steelblue', 'coral', 'green', 'purple']
    for i, label in labels.items():
        pts = subcat[subcat['cluster'] == i]
        ax.scatter(pts['pca1'], pts['pca2'], label=label, color=colors_c[i], s=100)
        for _, row in pts.iterrows():
            ax.annotate(row['Sub-Category'], (row['pca1']+0.05, row['pca2']+0.05), fontsize=7)
    ax.set_title('Product Demand Segmentation')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Category Cluster Assignments")
    st.dataframe(
        subcat[['Sub-Category', 'Demand Segment', 'total_sales', 'growth']]\
        .rename(columns={'total_sales': 'Total Sales', 'growth': 'YoY Growth %'})\
        .sort_values('Demand Segment').reset_index(drop=True)
    )
