"""Download TLC High Volume FHV (Uber/Lyft) trip records month by month, reduce each to aggregate tables, and drop
the raw month (re-downloadable from the TLC CDN) to fit the disk.

Outputs in OUT (one parquet per month and table):
  od_day_<ym>.parquet    date x company x PU zone x DO zone: trips, miles, trip time, fare, driver pay,
                         CBD fee sum, trips charged the CBD fee, request-to-pickup wait, shared requests
  zone_hour_<ym>.parquet date x hour x company x PU zone: same measures
  crz_zones.csv          congestion relief zone taxi zones, derived from January 2025 pickups (>90% charged)
Usage: fetch_hvfhv.py 2024-01 2026-08 [workers]
"""
import sys, os, time, subprocess, concurrent.futures as cf
import numpy as np, pandas as pd, pyarrow.parquet as pq

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{}.parquet"
OUT = os.path.expanduser("~/Desktop/CGT/CGT-DB/Traffic/NYC_pricing/hvfhv_agg")
TMP = os.environ.get("HV_TMP", "/tmp/hvfhv_raw")
COLS = ["hvfhs_license_num", "request_datetime", "pickup_datetime", "PULocationID", "DOLocationID", "trip_miles",
        "trip_time", "base_passenger_fare", "driver_pay", "congestion_surcharge", "shared_request_flag"]


def months(a, b):
    return [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]


def remote_size(ym):
    h = subprocess.run(["curl", "-sfI", "--retry", "5", URL.format(ym)], capture_output=True, text=True).stdout
    return int(next(l.split(":")[1] for l in h.lower().splitlines() if l.startswith("content-length")))


def download(ym):
    """Resume (-C -) until the local size equals the remote size; the connection drops mid-file often."""
    dst = os.path.join(TMP, f"{ym}.parquet")
    if os.path.exists(dst) or os.path.exists(os.path.join(OUT, f"od_day_{ym}.parquet")):
        return ym
    size, part = remote_size(ym), dst + ".part"
    for attempt in range(200):
        subprocess.run(["curl", "-s", "-C", "-", "--speed-limit", "20000", "--speed-time", "60", "-o", part, URL.format(ym)])
        if os.path.exists(part) and os.path.getsize(part) == size:
            os.rename(part, dst)
            return ym
        if os.path.exists(part) and os.path.getsize(part) > size:
            os.remove(part)
        time.sleep(5)
    raise RuntimeError(f"{ym}: incomplete after 200 attempts")


def reduce(ym):
    path = os.path.join(TMP, f"{ym}.parquet")
    names = pq.ParquetFile(path).schema_arrow.names
    t = pq.read_table(path, columns=COLS + (["cbd_congestion_fee"] if "cbd_congestion_fee" in names else [])).to_pandas()
    if "cbd_congestion_fee" not in t:
        t["cbd_congestion_fee"] = 0.0
    t["date"] = t.pickup_datetime.dt.normalize()
    t["hour"] = t.pickup_datetime.dt.hour.astype("int8")
    t["company"] = t.hvfhs_license_num.map({"HV0003": "uber", "HV0005": "lyft"}).fillna(t.hvfhs_license_num)
    t["wait_s"] = (t.pickup_datetime - t.request_datetime).dt.total_seconds().clip(0, 3 * 3600)
    t["fee_n"] = (t.cbd_congestion_fee > 0).astype("int32")
    t["shared_req"] = (t.shared_request_flag == "Y").astype("int32")
    agg = dict(trips=("trip_miles", "size"), miles=("trip_miles", "sum"), trip_time_s=("trip_time", "sum"),
               fare=("base_passenger_fare", "sum"), driver_pay=("driver_pay", "sum"),
               cbd_fee=("cbd_congestion_fee", "sum"), cbd_fee_n=("fee_n", "sum"),
               cong_surcharge=("congestion_surcharge", "sum"), wait_s=("wait_s", "sum"), shared_req=("shared_req", "sum"))
    od = t.groupby(["date", "company", "PULocationID", "DOLocationID"], observed=True).agg(**agg).reset_index()
    zh = t.groupby(["date", "hour", "company", "PULocationID"], observed=True).agg(**agg).reset_index()
    od.to_parquet(os.path.join(OUT, f"od_day_{ym}.parquet"), index=False)
    zh.to_parquet(os.path.join(OUT, f"zone_hour_{ym}.parquet"), index=False)
    if ym == "2025-01" and not os.path.exists(os.path.join(OUT, "crz_zones.csv")):
        f = t[t.pickup_datetime >= "2025-01-06"]
        share = f.groupby("PULocationID").fee_n.mean()
        share[share > 0.9].rename("share_charged").to_csv(os.path.join(OUT, "crz_zones.csv"))
    os.remove(path)
    return ym, len(t)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    todo = [m for m in months(a, b) if not os.path.exists(os.path.join(OUT, f"od_day_{m}.parquet"))]
    # months nearest the policy start (2025-01) first, so a usable before/after window exists early
    todo.sort(key=lambda m: abs((pd.Period(m, "M") - pd.Period("2025-01", "M")).n))
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(download, m): m for m in todo}
        for fu in cf.as_completed(futs):
            try:
                ym = fu.result()
                print("reduced", *reduce(ym), time.strftime("%H:%M"), flush=True)
            except Exception as e:
                print("FAILED", futs[fu], repr(e), flush=True)
