import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from datetime import date
import io
import os

S3_BUCKET = os.environ['S3_BUCKET']

SECTORS = {
    'Technology':             ['AAPL','MSFT','NVDA','META','GOOGL','AVGO','ORCL','AMD','INTC','CRM'],
    'Healthcare':             ['UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','DHR','PFE','BMY'],
    'Financials':             ['JPM','V','MA','BAC','WFC','GS','MS','BLK','C','AXP'],
    'Consumer_Discretionary': ['AMZN','TSLA','HD','MCD','NKE','SBUX','TGT','BKNG','LOW','TJX'],
    'Consumer_Staples':       ['PG','KO','PEP','COST','WMT','PM','MO','CL','KHC','MDLZ'],
    'Energy':                 ['XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','HAL'],
    'Industrials':            ['GE','CAT','RTX','HON','UNP','BA','MMM','LMT','DE','UPS'],
    'Materials':              ['LIN','APD','SHW','FCX','NEM','ECL','NUE','VMC','MLM','CF'],
    'Utilities':              ['NEE','DUK','SO','D','AEP','EXC','XEL','PEG','ED','WEC'],
    'Real_Estate':            ['AMT','PLD','CCI','EQIX','PSA','SPG','WELL','DLR','O','VICI'],
    'Communication':          ['GOOG','META','VZ','T','CMCSA','NFLX','DIS','TMUS','WBD','EA'],
}

ALL_TICKERS  = [t for tickers in SECTORS.values() for t in tickers]
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}


def fetch_market_data():
    raw = yf.download(
        tickers=ALL_TICKERS,
        period='35d',
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
    )

    rows = []
    today = date.today()

    for ticker in ALL_TICKERS:
        try:
            df = raw[ticker].dropna(subset=['Close', 'Volume'])
        except KeyError:
            continue

        df = df.copy()
        df.index = pd.to_datetime(df.index).date
        hist = df[df.index < today]
        avg_vol_30d = hist['Volume'].tail(30).mean() if len(hist) >= 5 else None

        for day, row in df.iterrows():
            close  = round(float(row['Close']), 4)
            open_  = round(float(row['Open']),  4)
            volume = int(row['Volume'])
            daily_return = round((close - open_) / open_ * 100, 4) if open_ else None
            volume_spike = bool(volume > 2 * avg_vol_30d) if avg_vol_30d and avg_vol_30d > 0 else False

            rows.append({
                'trade_date':       day.isoformat(),
                'ticker':           ticker,
                'sector':           TICKER_SECTOR.get(ticker, 'Unknown'),
                'open':             open_,
                'high':             round(float(row['High']), 4),
                'low':              round(float(row['Low']),  4),
                'close':            close,
                'volume':           volume,
                'daily_return_pct': daily_return,
                'avg_volume_30d':   round(avg_vol_30d, 0) if avg_vol_30d else None,
                'volume_spike':     volume_spike,
                'pulled_date':      today.isoformat(),
            })

    return pd.DataFrame(rows)


def compute_sector_summary(df):
    today_str = date.today().isoformat()
    today_df  = df[df['trade_date'] == today_str]
    if today_df.empty:
        today_df = df[df['trade_date'] == df['trade_date'].max()]

    summary = (
        today_df.groupby('sector')
        .agg(
            avg_return_pct = ('daily_return_pct', 'mean'),
            total_volume   = ('volume',           'sum'),
            gainers        = ('daily_return_pct', lambda x: (x > 0).sum()),
            losers         = ('daily_return_pct', lambda x: (x < 0).sum()),
            volume_spikes  = ('volume_spike',     'sum'),
            ticker_count   = ('ticker',           'count'),
        )
        .reset_index()
    )
    summary['avg_return_pct'] = summary['avg_return_pct'].round(4)
    summary['summary_date']   = today_str
    summary['pulled_date']    = today_str
    return summary


def write_parquet_to_s3(df, dataset_name):
    today  = date.today()
    s3_key = (
        f"processed/{dataset_name}/"
        f"year={today.year}/month={today.month:02d}/day={today.day:02d}/data.parquet"
    )
    buf   = io.BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf, compression='snappy')
    buf.seek(0)

    boto3.client('s3', region_name='us-east-1').put_object(
        Bucket=S3_BUCKET, Key=s3_key, Body=buf.getvalue()
    )
    print(f"✓ {dataset_name}: {len(df):,} rows → s3://{S3_BUCKET}/{s3_key}")


if __name__ == '__main__':
    print("Fetching market data...")
    market_df = fetch_market_data()
    print(f"Got {len(market_df):,} rows across {market_df['ticker'].nunique()} tickers")

    write_parquet_to_s3(market_df, 'sp500_daily')

    sector_df = compute_sector_summary(market_df)
    write_parquet_to_s3(sector_df, 'sector_summary')

    today_str  = date.today().isoformat()
    today_data = market_df[market_df['trade_date'] == today_str]
    if not today_data.empty:
        print("\nTop 5 gainers:")
        print(today_data.nlargest(5, 'daily_return_pct')[['ticker','sector','daily_return_pct','volume_spike']].to_string(index=False))
        print("\nTop 5 losers:")
        print(today_data.nsmallest(5, 'daily_return_pct')[['ticker','sector','daily_return_pct','volume_spike']].to_string(index=False))

    print("\nDone.")
