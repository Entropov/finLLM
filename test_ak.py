import akshare as ak
try:
    df = ak.stock_zh_a_hist(symbol="002625", period="daily", start_date="20230307", end_date="20260306", adjust="qfq")
    print(df.head())
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
