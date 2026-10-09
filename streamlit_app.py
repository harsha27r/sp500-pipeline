import streamlit as st
import boto3
import pandas as pd
import io
import altair as alt

st.set_page_config(page_title="S&P 500 Sector Analytics", page_icon="📈", layout="wide")

S3_BUCKET = "harsha-job-market-pipeline"
REGION    = "us-east-1"

@st.cache_data(ttl=3600)
def load_parquet_from_s3(prefix):
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id     = st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
    )
    paginator = s3.get_paginator("list_objects_v2")
    frames = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            buf = io.BytesIO()
            s3.download_fileobj(S3_BUCKET, key, buf)
            buf.seek(0)
            frames.append(pd.read_parquet(buf))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

with st.spinner("Loading data from S3…"):
    daily_df  = load_parquet_from_s3("processed/sp500_daily/")
    sector_df = load_parquet_from_s3("processed/sector_summary/")

if daily_df.empty:
    st.error("No data found in S3. Run the pipeline first.")
    st.stop()

daily_df["trade_date"]    = pd.to_datetime(daily_df["trade_date"])
sector_df["summary_date"] = pd.to_datetime(sector_df["summary_date"])

latest_date   = daily_df["trade_date"].max()
latest_daily  = daily_df[daily_df["trade_date"] == latest_date].copy()
latest_sector = sector_df[sector_df["summary_date"] == sector_df["summary_date"].max()].copy()

st.title("📈 S&P 500 Sector Analytics Pipeline")
st.caption(f"Data as of **{latest_date.strftime('%B %d, %Y')}** · Pipeline: GitHub Actions → S3 (Parquet) → Athena · Built by Harsha Raja Neathala")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickers Tracked",  f"{latest_daily['ticker'].nunique()}")
col2.metric("Sectors",          f"{latest_daily['sector'].nunique()}")
col3.metric("Avg Daily Return", f"{latest_daily['daily_return_pct'].mean():.2f}%")
col4.metric("Volume Spikes",    f"{latest_daily['volume_spike'].sum()}")

st.divider()
st.subheader("Sector Performance — Latest Trading Day")

chart_data = latest_sector[["sector","avg_return_pct"]].sort_values("avg_return_pct", ascending=False).copy()
chart_data["color"] = ["green" if v >= 0 else "red" for v in chart_data["avg_return_pct"]]

bar = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        x=alt.X("avg_return_pct:Q", title="Avg Return (%)"),
        y=alt.Y("sector:N", sort="-x", title="Sector"),
        color=alt.Color("color:N", scale=alt.Scale(domain=["green","red"], range=["#22c55e","#ef4444"]), legend=None),
        tooltip=["sector", alt.Tooltip("avg_return_pct:Q", format=".2f")],
    )
    .properties(height=350)
)
st.altair_chart(bar, use_container_width=True)

st.divider()
col_g, col_l = st.columns(2)

with col_g:
    st.subheader("🟢 Top 5 Gainers")
    gainers = latest_daily.nlargest(5, "daily_return_pct")[["ticker","sector","daily_return_pct","close","volume_spike"]].reset_index(drop=True)
    gainers.index += 1
    st.dataframe(gainers, use_container_width=True)

with col_l:
    st.subheader("🔴 Top 5 Losers")
    losers = latest_daily.nsmallest(5, "daily_return_pct")[["ticker","sector","daily_return_pct","close","volume_spike"]].reset_index(drop=True)
    losers.index += 1
    st.dataframe(losers, use_container_width=True)

st.divider()
st.subheader("⚡ Volume Anomalies — Tickers with >2× Avg Volume")
spikes = latest_daily[latest_daily["volume_spike"] == True][["ticker","sector","volume","avg_volume_30d","daily_return_pct"]].copy()
if spikes.empty:
    st.info("No volume spikes detected today.")
else:
    spikes["volume_ratio"] = (spikes["volume"] / spikes["avg_volume_30d"]).round(2)
    st.dataframe(spikes.sort_values("volume_ratio", ascending=False).reset_index(drop=True), use_container_width=True)

st.divider()
st.subheader("📊 Historical Close Price")
tickers  = sorted(daily_df["ticker"].unique().tolist())
selected = st.selectbox("Select ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
ticker_df = daily_df[daily_df["ticker"] == selected].sort_values("trade_date")

line = (
    alt.Chart(ticker_df)
    .mark_line(color="#3b82f6", strokeWidth=2)
    .encode(
        x=alt.X("trade_date:T", title="Date"),
        y=alt.Y("close:Q", title="Close Price ($)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("trade_date:T", format="%b %d, %Y"), alt.Tooltip("close:Q", format="$.2f"), alt.Tooltip("daily_return_pct:Q", format=".2f")],
    )
    .properties(height=300)
)
st.altair_chart(line, use_container_width=True)

st.divider()
st.subheader("📋 Sector Summary Table")
display_sector = latest_sector[["sector","avg_return_pct","gainers","losers","volume_spikes","total_volume","ticker_count"]].sort_values("avg_return_pct", ascending=False).reset_index(drop=True)
display_sector.index += 1
st.dataframe(display_sector, use_container_width=True)

st.caption("Pipeline runs Mon–Fri at 5pm ET via GitHub Actions. Data stored in AWS S3 as Parquet, queried via Amazon Athena.")
