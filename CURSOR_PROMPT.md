# Cursor Prompt — ML Pipeline Frontend Integration

## Context: What This Project Is

This is an IS 455 class project building a full-stack ML deployment system. The system allows a retail company to place orders through a web app, score those orders for fraud risk using a trained ML model, and display a fraud priority queue for review.

There are three components:
1. **Supabase** — shared PostgreSQL database (cloud)
2. **Vercel** — hosts the Next.js frontend
3. **Railway** — hosts a Python FastAPI service that runs the ML inference

---

## The ML Service (Already Built and Deployed)

A Python FastAPI service is live on Railway at:
```
https://mlprojectpipeline-production.up.railway.app
```

### Available Endpoints

**Health check:**
```
GET https://mlprojectpipeline-production.up.railway.app/
```
Returns:
```json
{"status": "ok", "service": "fraud-inference"}
```

**Trigger fraud scoring:**
```
POST https://mlprojectpipeline-production.up.railway.app/score
```
- No request body required
- No special headers required
- Content-Type does not need to be set
- Returns JSON:
```json
{
  "success": true,
  "stdout": "✓ Connected to Supabase\n✓ 5000 predictions written...",
  "stderr": "",
  "duration_ms": 12400
}
```
- `success: true` means predictions were written to Supabase successfully
- `success: false` means something failed — check `stderr` for details
- This call can take 30–90 seconds on the first run (scoring 5,000 orders)
- Do NOT set a short timeout on this call — use at least 120 seconds

### What the scoring job does
When `/score` is called, the Railway service:
1. Connects to Supabase
2. Pulls all orders from the `orders` table that do not yet have a row in `order_predictions`
3. Joins with `customers`, `order_items`, `products`, and `shipments` tables
4. Applies ML feature engineering
5. Loads the trained fraud detection model (`fraud_model.sav`)
6. Generates `fraud_probability` (0.0–1.0) and `predicted_fraud` (0 or 1) for each order
7. Writes results to the `order_predictions` table in Supabase

---

## The Supabase Database

**Project URL:** `https://tqvaebgxkymimisiahfc.supabase.co`

### Tables that exist (all populated with data):
- `customers` — 250 rows
- `products` — 100 rows
- `orders` — 5,000 rows
- `order_items` — ~15,000 rows
- `shipments` — 5,000 rows
- `product_reviews` — 3,000 rows
- `order_predictions` — populated after scoring runs

### The `order_predictions` table schema:
```sql
CREATE TABLE order_predictions (
  order_id              integer PRIMARY KEY,
  fraud_probability     numeric,      -- 0.0 to 1.0
  predicted_fraud       integer,      -- 0 or 1
  prediction_timestamp  timestamptz DEFAULT now()
);
```

### The fraud priority queue query:
This is the SQL that should power the fraud review queue page. It joins orders with customer info and predictions, returning high-risk unfulfilled orders first:

```sql
SELECT
  o.order_id,
  o.order_datetime,
  o.order_total,
  o.fulfilled,
  o.payment_method,
  o.ip_country,
  c.customer_id,
  c.full_name AS customer_name,
  c.customer_segment,
  p.fraud_probability,
  p.predicted_fraud,
  p.prediction_timestamp
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN order_predictions p ON p.order_id = o.order_id
WHERE o.fulfilled = 0
ORDER BY p.fraud_probability DESC, o.order_datetime ASC
LIMIT 50;
```

To call this via Supabase JS client:
```typescript
const { data, error } = await supabase
  .from('orders')
  .select(`
    order_id,
    order_datetime,
    order_total,
    fulfilled,
    payment_method,
    ip_country,
    customers!inner(customer_id, full_name, customer_segment),
    order_predictions!inner(fraud_probability, predicted_fraud, prediction_timestamp)
  `)
  .eq('fulfilled', 0)
  .order('fraud_probability', { ascending: false, foreignTable: 'order_predictions' })
  .limit(50);
```

Or alternatively, create a Supabase RPC function called `get_fraud_queue` using the SQL above and call it with:
```typescript
const { data, error } = await supabase.rpc('get_fraud_queue');
```

---

## What the Frontend Needs to Do

### 1. "Run Scoring" page
There should be a page with a button labeled "Run Scoring" or "Score Orders". When clicked:

```typescript
const res = await fetch(
  "https://mlprojectpipeline-production.up.railway.app/score",
  {
    method: "POST",
    // No body, no special headers needed
  }
);
const data = await res.json();
// data.success === true means it worked
// data.stdout has the log output to display
// data.stderr has any error output
// data.duration_ms is how long it took
```

The button should show a loading state while waiting (this can take 60+ seconds). After completion, show success or error message and link to the fraud queue page.

### 2. Fraud Priority Queue page
Displays the results from `order_predictions` joined with `orders` and `customers`. Rows should be color-coded:
- `fraud_probability > 0.7` → red (high risk)
- `fraud_probability > 0.4` → yellow (moderate risk)
- `fraud_probability <= 0.4` → green (low risk)

Columns to display:
- Order ID
- Customer Name (from `customers.full_name`)
- Order Date (`order_datetime`)
- Order Total (`order_total`)
- Payment Method (`payment_method`)
- Fraud Probability (formatted as percentage, e.g. "73.2%")
- Predicted Fraud (Yes/No badge)
- Prediction Timestamp

### 3. Other required pages (per assignment)
- **Select Customer** — dropdown of customers from `customers` table, no login required
- **Place Order** — selected customer places an order, saved to `orders` table
- **Order History** — admin view of all orders in `orders` table

---

## Key Things to Know

### CORS is handled
The Railway FastAPI service has `allow_origins=["*"]` so any frontend origin can call it without CORS errors.

### The fetch call to Railway is simple
Do NOT add `Content-Type: application/json` or a request body to the `/score` POST — it doesn't need one and adding a body may cause issues. Just:
```typescript
fetch("https://mlprojectpipeline-production.up.railway.app/score", { method: "POST" })
```

### Timeout handling
The scoring job takes 30–90 seconds. The frontend should:
- Show a spinner/loading state immediately
- Not time out before 120 seconds
- Display `data.stdout` after completion so the user can see what happened

### After scoring runs
The `order_predictions` table in Supabase will have new rows. The fraud queue page should re-fetch from Supabase after scoring completes to show updated results.

### The `order_predictions` table uses `order_id` as primary key with upsert
Re-running scoring will update existing predictions rather than creating duplicates.

---

## Supabase RPC Function to Create (Optional but Recommended)

Run this in the Supabase SQL editor to create a clean RPC function for the fraud queue:

```sql
CREATE OR REPLACE FUNCTION get_fraud_queue()
RETURNS TABLE(
  order_id integer,
  order_datetime text,
  order_total numeric,
  fulfilled integer,
  payment_method text,
  ip_country text,
  customer_id integer,
  customer_name text,
  customer_segment text,
  fraud_probability numeric,
  predicted_fraud integer,
  prediction_timestamp timestamptz
)
LANGUAGE sql SECURITY DEFINER AS $$
  SELECT
    o.order_id,
    o.order_datetime,
    o.order_total,
    o.fulfilled,
    o.payment_method,
    o.ip_country,
    c.customer_id,
    c.full_name AS customer_name,
    c.customer_segment,
    p.fraud_probability,
    p.predicted_fraud,
    p.prediction_timestamp
  FROM orders o
  JOIN customers c ON c.customer_id = o.customer_id
  JOIN order_predictions p ON p.order_id = o.order_id
  WHERE o.fulfilled = 0
  ORDER BY p.fraud_probability DESC, o.order_datetime ASC
  LIMIT 50;
$$;
```

Then call it from the frontend with:
```typescript
const { data, error } = await supabase.rpc('get_fraud_queue');
```

---

## Summary of Action Items for Cursor

1. **Fix or create the "Run Scoring" page** — button calls `POST https://mlprojectpipeline-production.up.railway.app/score` with no body, shows loading state, displays result
2. **Fix or create the Fraud Priority Queue page** — reads from `order_predictions` joined with `orders` and `customers`, color-coded by fraud probability
3. **Create the `get_fraud_queue` RPC function** in Supabase SQL editor (SQL provided above)
4. **Ensure the Select Customer, Place Order, and Order History pages exist** per assignment requirements
5. **Do not add any authentication** — the assignment specifies no signup/login required
