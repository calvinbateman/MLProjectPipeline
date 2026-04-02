"""
run_inference.py
────────────────
Fraud scoring inference script for the IS 455 deployment pipeline.

Called by:
  - The Next.js API route (/api/scoring/run) via child_process.spawn
  - The Railway FastAPI service (/score endpoint) via subprocess
  - Manually: python jobs/run_inference.py

Reads unscored orders from Supabase, applies identical feature engineering
as training, loads fraud_model.sav, and writes predictions back to Supabase.

Required environment variables:
  SUPABASE_URL  — https://tqvaebgxkymimisiahfc.supabase.co
  SUPABASE_KEY  — your secret key (sb_secret_...)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ── Supabase client ──────────────────────────────────────────────────────────
try:
    from supabase import create_client
except ImportError:
    print("ERROR: supabase-py not installed. Run: pip install supabase")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://tqvaebgxkymimisiahfc.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    print("ERROR: SUPABASE_KEY environment variable not set.")
    sys.exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"✓ Connected to Supabase: {SUPABASE_URL}")

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.sav")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("  Run the CRISP-DM notebook to generate fraud_model.sav first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded — steps: {[s[0] for s in model.steps]}")

# ── Helper: fetch full table from Supabase ───────────────────────────────────
def fetch_table(table: str) -> pd.DataFrame:
    """Fetch all rows from a Supabase table, handling pagination."""
    all_rows = []
    page = 0
    page_size = 1000
    while True:
        res = (sb.table(table)
               .select("*")
               .range(page * page_size, (page + 1) * page_size - 1)
               .execute())
        rows = res.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        page += 1
    return pd.DataFrame(all_rows)

# ── Load all source tables ───────────────────────────────────────────────────
print("\nLoading tables from Supabase...")
orders      = fetch_table("orders")
customers   = fetch_table("customers")
order_items = fetch_table("order_items")
products    = fetch_table("products")
shipments   = fetch_table("shipments")

print(f"  orders:      {len(orders):,}")
print(f"  customers:   {len(customers):,}")
print(f"  order_items: {len(order_items):,}")
print(f"  products:    {len(products):,}")
print(f"  shipments:   {len(shipments):,}")

# ── Find unscored orders ─────────────────────────────────────────────────────
# Ensure order_predictions table exists
# (your teammate should have this in the Supabase migration SQL)
try:
    scored_res = sb.table("order_predictions").select("order_id").execute()
    scored_ids = set(str(r["order_id"]) for r in scored_res.data)
except Exception:
    # Table doesn't exist yet — score everything
    scored_ids = set()

unscored = orders[~orders["order_id"].astype(str).isin(scored_ids)].copy()
print(f"\n✓ {len(unscored)} unscored orders to process (of {len(orders)} total)")

if len(unscored) == 0:
    print("All orders already scored. Nothing to do.")
    sys.exit(0)

# ── ETL: aggregate order_items to order level ────────────────────────────────
order_agg = (
    order_items
    .groupby("order_id")
    .agg(
        num_items             = ("quantity",   "sum"),
        num_unique_products   = ("product_id", "nunique"),
        avg_unit_price        = ("unit_price",  "mean"),
        max_unit_price        = ("unit_price",  "max"),
    )
    .reset_index()
)

# Dominant product category per order
order_cats = (
    order_items
    .merge(products[["product_id", "category"]], on="product_id", how="left")
    .groupby("order_id")["category"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reset_index()
    .rename(columns={"category": "top_category"})
)

# Pre-delivery shipment features only (post-delivery = leakage)
ship_feats = shipments[["order_id", "carrier", "shipping_method",
                         "distance_band", "promised_days"]]

# Full join
df = (
    unscored
    .merge(customers[["customer_id", "full_name", "gender", "birthdate",
                       "customer_segment", "loyalty_tier"]],
           on="customer_id", how="left")
    .merge(order_agg,  on="order_id", how="left")
    .merge(order_cats, on="order_id", how="left")
    .merge(ship_feats, on="order_id", how="left")
)

# Keep order_id aside before dropping
order_ids = df["order_id"].copy()

# ── Feature engineering (IDENTICAL to training — Ch. 17 principle) ───────────
df["order_datetime"] = pd.to_datetime(df["order_datetime"])
df["birthdate"]      = pd.to_datetime(df["birthdate"], errors="coerce")

df["order_dow"]   = df["order_datetime"].dt.dayofweek
df["order_month"] = df["order_datetime"].dt.month
df["order_hour"]  = df["order_datetime"].dt.hour

df["customer_age"] = (
    (datetime.now().year - df["birthdate"].dt.year)
    .clip(0, 120)
)

# customer_order_count — use full orders table for correct historical counts
order_counts = (
    orders.groupby("customer_id")["order_id"]
    .count()
    .reset_index()
    .rename(columns={"order_id": "customer_order_count"})
)
df = df.merge(order_counts, on="customer_id", how="left")

df["same_zip"]   = (df["billing_zip"] == df["shipping_zip"]).astype(int)
df["foreign_ip"] = (df["ip_country"] != "US").astype(int)

# ── Drop leakage columns, IDs, and raw dates ─────────────────────────────────
DROP_COLS = [
    "order_id", "customer_id",
    "order_datetime", "birthdate",
    "billing_zip", "shipping_zip",
    "full_name", "promo_code",
    "risk_score", "is_fraud",          # targets / co-targets
    "fulfilled",                        # post-placement status
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# ── Rare category binning (5% rule — same threshold as training) ─────────────
for col in df.select_dtypes(include=["object"]).columns:
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < 0.05].index.tolist()
    if rare:
        df[col] = df[col].replace(rare, "Other")

print(f"✓ Feature engineering complete — X shape: {df.shape}")

# ── Generate predictions ─────────────────────────────────────────────────────
fraud_proba = model.predict_proba(df)[:, 1]
fraud_pred  = model.predict(df)

print(f"✓ Predictions generated")
print(f"  Fraud rate (predicted): {fraud_pred.mean():.3%}")
print(f"  High-risk (prob > 0.5): {(fraud_proba > 0.5).sum()}")

# ── Write predictions to Supabase ────────────────────────────────────────────
timestamp = datetime.utcnow().isoformat()

records = [
    {
        "order_id":            int(oid),
        "fraud_probability":   round(float(prob), 6),
        "predicted_fraud":     int(pred),
        "prediction_timestamp": timestamp,
    }
    for oid, prob, pred in zip(order_ids, fraud_proba, fraud_pred)
]

# Upsert in batches of 500
batch_size = 500
for i in range(0, len(records), batch_size):
    batch = records[i : i + batch_size]
    sb.table("order_predictions").upsert(batch).execute()

print(f"\n✓ Wrote {len(records)} predictions to order_predictions table")
print("Done.")
