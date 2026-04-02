# IS 455 Chapter 17 Assignment: ML Pipeline Master Plan

## Your Role

Calvin is building the **ML pipeline only** (Part 2: CRISP-DM Jupyter Notebook predicting `is_fraud`). Teammates are building the web app (Part 1) using React + Next.js, deployed to Vercel with Supabase (PostgreSQL) as the database. Calvin develops locally against `shop.db` (SQLite) and the pipeline output must integrate with the Supabase-backed app at deployment.

---

## Master Prompt (for AI-Assisted Development)

Paste this into Claude, Cursor, or ChatGPT when you're ready to start building:

```
You are helping me build a CRISP-DM Jupyter notebook that predicts the `is_fraud`
column in the `orders` table of a SQLite database called `shop.db`. This is for a
university ML class (IS 455) using the textbook "Machine Learning in Python: From
Data Collection to Model Deployment" by Mark Keith.

IMPORTANT CONSTRAINTS:
- All code must follow the patterns and techniques from the textbook (Chapters 2-4,
  6-8, 13-16, 17).
- The database is `shop.db` (SQLite) with tables: orders, customers, order_items,
  products, and potentially order_predictions.
- The target variable is `is_fraud` (binary: 1 = fraud, 0 = legitimate) which
  already exists in the orders table.
- This notebook is the analytical/modeling work. Separately, my teammates are
  building a web app (Next.js + Supabase) that will consume predictions.
- At the end, I must serialize the model and show how it integrates into a
  deployment pipeline (Ch. 17 pattern: ETL -> Train -> Inference -> write
  predictions back to DB).

THE NOTEBOOK MUST DEMONSTRATE THESE CRISP-DM PHASES:

1. BUSINESS UNDERSTANDING
   - Define the fraud detection problem and success criteria
   - Explain why fraud detection matters operationally
   - Define what metrics matter (precision vs recall tradeoff for fraud)

2. DATA UNDERSTANDING
   - Load all tables from shop.db using sqlite3 + pandas
   - Explore each table: shape, dtypes, head(), describe()
   - Feature-level exploration using automated `unistats()` function (Ch. 6 pattern):
     automated univariate statistics with type-checking via
     pd.api.types.is_numeric_dtype()
   - Relationship discovery using automated `bivariate()` function (Ch. 8 pattern):
     N2N (Pearson r, Kendall tau, Spearman rho via scipy.stats.linregress,
     kendalltau, spearmanr), C2N/N2C (ANOVA via f_oneway, pairwise t-tests with
     Bonferroni), C2C (chi-square via chi2_contingency)
   - Visualizations: scatterplots with embedded stats (sns.regplot), bar charts
     (sns.barplot), heatmaps (sns.heatmap with pd.crosstab)

3. DATA PREPARATION
   - ETL: Join/denormalize tables into one-row-per-order modeling table (Ch. 17
     pattern: orders + customers + aggregated order_items + products)
   - Basic wrangling (Ch. 7): remove empty/constant columns, standardize types
   - Date/time feature engineering: extract year, month, day_of_week from
     order_timestamp
   - Numeric features: customer_age from birthdate, customer_order_count,
     num_items, total_value, avg_weight
   - Categorical binning: group rare categories using 5% rule (Ch. 7)
   - Skewness correction: test log1p, sqrt, cbrt, Yeo-Johnson (Ch. 7 pattern)
   - Missing data handling: test MCAR vs MAR, then impute or drop (Ch. 7)
   - Outlier detection: Z-score/IQR per feature, optionally DBSCAN (Ch. 7)
   - Build reusable cleaning functions that can be imported by the deployment
     pipeline

4. MODELING
   - Stratified train/test split (stratify=y because fraud is imbalanced)
   - Build sklearn Pipeline objects combining preprocessing + model (Ch. 17 pattern):
     ColumnTransformer with numeric pipe (SimpleImputer + StandardScaler) and
     categorical pipe (SimpleImputer + OneHotEncoder)
   - Train baseline: LogisticRegression (Ch. 13)
   - Train ensemble models: RandomForestClassifier, GradientBoostingClassifier
     (Ch. 14)
   - Compare using log_loss (primary for probability quality), accuracy, F1,
     ROC AUC, precision, recall

5. EVALUATION
   - Cross-validation with StratifiedKFold(n_splits=5) and multi-metric scoring
     (Ch. 15)
   - Learning curves to diagnose bias/variance (Ch. 15)
   - Hyperparameter tuning with GridSearchCV or RandomizedSearchCV (Ch. 15)
   - Feature selection: VarianceThreshold, SelectKBest(f_classif), RFECV,
     permutation importance (Ch. 16)
   - Classification report with precision, recall, F1 per class
   - ROC curve and PR curve visualization
   - Final model selection with justification

6. DEPLOYMENT
   - Serialize best model with joblib (Ch. 17 pattern)
   - Save model_metadata.json (version, timestamp, features, row counts)
   - Save metrics.json (all evaluation metrics)
   - Demonstrate inference: load model, generate predictions on new/unfulfilled
     orders, write predictions to order_predictions table in shop.db
   - Show the exact SQL query the web app will use to display the fraud queue
   - Explain how this integrates with the team's Next.js + Supabase deployment

STYLE REQUIREMENTS:
- Use markdown headers and explanatory text between code cells (this is a
  report, not just code)
- Every code cell should have a preceding markdown cell explaining what it does
  and why
- Print/display intermediate results so the notebook tells a story
- Use f-strings for clean output formatting
- Follow scikit-learn Pipeline conventions throughout
```

---

## Detailed Plan Outline

### Phase 0: Setup and Data Loading

**Goal**: Get the data loaded, understand the schema, confirm `is_fraud` exists.

Cells to create:

- **0.1** Imports cell: sqlite3, pandas, numpy, matplotlib, seaborn, sklearn modules, scipy.stats, joblib, json, datetime, warnings
- **0.2** Connect to shop.db, list all tables with `SELECT name FROM sqlite_master WHERE type='table'`
- **0.3** Load each table into its own DataFrame: `orders`, `customers`, `order_items`, `products`
- **0.4** Print shape, dtypes, head(3) for each table
- **0.5** Check `is_fraud` value counts and class balance: `orders["is_fraud"].value_counts(normalize=True)`

**Key question to answer**: How imbalanced is the fraud class? This drives metric choices later.

---

### Phase 1: Business Understanding

**Goal**: Frame the problem in business terms. (2-3 markdown cells, no code needed.)

Content to write:

- What is the business problem? (Fraudulent orders cost money, damage trust)
- What does a correct prediction enable? (Flag for manual review, delay fulfillment, trigger verification)
- Success criteria: prioritize **recall** (catch as many frauds as possible) while keeping precision reasonable to avoid overwhelming the review team
- Define the prediction task: binary classification, `is_fraud` = 1 (fraud) vs 0 (legitimate)

---

### Phase 2: Data Understanding

**Goal**: Explore features at the univariate and bivariate level following Ch. 6 and Ch. 8 patterns.

Cells to create:

- **2.1** ETL denormalization (following Ch. 17 Sec 17.4 exactly):
  - Aggregate `order_items` joined with `products` by `order_id` to get: `num_items`, `avg_price`, `total_value`, `avg_weight`
  - Merge: `orders` + `customers` + `order_item_features`
  - This produces the single modeling DataFrame `df`

- **2.2** Automated univariate stats function `unistats(df)` (Ch. 6 pattern):
  - Loop through columns, check `is_numeric_dtype()`
  - For numeric: count, unique, min, max, mean, median, std, skew, kurtosis
  - For categorical: count, unique, mode, frequency of mode
  - Return summary DataFrame

- **2.3** Call `unistats(df)` and display results

- **2.4** Visualize distributions:
  - Histograms for numeric features (matplotlib/seaborn)
  - Value count bar charts for categorical features
  - Special attention to class balance of `is_fraud`

- **2.5** Automated bivariate analysis function `bivariate(df, label="is_fraud")` (Ch. 8 pattern):
  - N2N: `linregress`, `kendalltau`, `spearmanr`, `sns.regplot`
  - C2N/N2C: `f_oneway`, pairwise t-tests with Bonferroni, `sns.barplot`
  - C2C: `chi2_contingency`, `sns.heatmap` with `pd.crosstab`

- **2.6** Call `bivariate()` and display ranked results by p-value

- **2.7** Key findings markdown cell: which features appear most related to fraud?

---

### Phase 3: Data Preparation

**Goal**: Clean, engineer features, and build a repeatable preparation pipeline following Ch. 7 and Ch. 17.

Cells to create:

- **3.1** Date/time feature engineering:
  ```python
  df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
  df["birthdate"] = pd.to_datetime(df["birthdate"])
  df["customer_age"] = (datetime.now().year - df["birthdate"].dt.year)
  df["order_dow"] = df["order_timestamp"].dt.dayofweek
  df["order_month"] = df["order_timestamp"].dt.month
  df["order_hour"] = df["order_timestamp"].dt.hour  # if timestamp has time
  ```

- **3.2** Customer-level aggregates:
  ```python
  df["customer_order_count"] = df.groupby("customer_id")["order_id"].transform("count")
  ```

- **3.3** Basic wrangling (Ch. 7 pattern):
  - Drop columns with >95% missing
  - Drop columns with >95% unique values (likely IDs)
  - Drop leakage columns: `delivery_days`, `ship_date`, `fulfilled` (if they leak future info), and any ID columns not needed for modeling

- **3.4** Missing data analysis:
  - Calculate missing percentage per column
  - Test MCAR vs MAR for columns with missing values (t-test for numeric labels, proportions z-test for categorical)
  - Decision: drop rows, impute with median/mode, or use KNNImputer

- **3.5** Skewness correction (Ch. 7 pattern):
  - Check skewness of numeric features
  - For features with |skew| > 1: test log1p, sqrt, cbrt, Yeo-Johnson
  - Select transformation that brings skewness closest to 0
  - Create `*_skewfix` columns

- **3.6** Outlier detection (Ch. 7 pattern):
  - For skewed features: Tukey IQR (1.5 * IQR)
  - For normal features: Empirical Rule (mean +/- 3*std)
  - Decision: cap, remove, or flag

- **3.7** Categorical binning (Ch. 7 pattern):
  - Group rare categories (<5% frequency) into "Other"

- **3.8** Define final feature list and target:
  ```python
  label_col = "is_fraud"
  feature_cols = [...]  # explicitly listed
  X = df[feature_cols]
  y = df[label_col].astype(int)
  ```

- **3.9** Wrap all preparation into a reusable function `prepare_data(df)` that can be imported by the deployment scripts later

---

### Phase 4: Modeling

**Goal**: Train classification models using sklearn Pipelines following Ch. 13-14 and Ch. 17.

Cells to create:

- **4.1** Stratified train/test split:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  ```

- **4.2** Build preprocessing pipeline (Ch. 17 pattern):
  ```python
  numeric_features = [...]  # list of numeric column names
  categorical_features = [...]  # list of categorical column names

  numeric_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ])
  categorical_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore"))
  ])
  preprocessor = ColumnTransformer([
      ("num", numeric_pipe, numeric_features),
      ("cat", categorical_pipe, categorical_features)
  ], remainder="drop")
  ```

- **4.3** Model 1 — Logistic Regression baseline (Ch. 13):
  ```python
  lr_pipeline = Pipeline([
      ("prep", preprocessor),
      ("clf", LogisticRegression(max_iter=1000))
  ])
  lr_pipeline.fit(X_train, y_train)
  ```

- **4.4** Model 2 — Random Forest (Ch. 14):
  ```python
  rf_pipeline = Pipeline([
      ("prep", preprocessor),
      ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
  ])
  ```

- **4.5** Model 3 — Gradient Boosting (Ch. 14):
  ```python
  gb_pipeline = Pipeline([
      ("prep", preprocessor),
      ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))
  ])
  ```

- **4.6** Train all models, collect predictions and probabilities

- **4.7** Comparison table:
  ```python
  # For each model: accuracy, log_loss, f1, roc_auc, precision, recall
  # Display as a clean DataFrame
  ```

---

### Phase 5: Evaluation and Tuning

**Goal**: Rigorously evaluate, tune, and select the best model following Ch. 15-16.

Cells to create:

- **5.1** Cross-validation with multiple metrics (Ch. 15):
  ```python
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1",
             "precision": "precision", "recall": "recall", "neg_log_loss": "neg_log_loss"}
  cv_results = cross_validate(best_candidate, X_train, y_train, cv=skf, scoring=scoring)
  ```

- **5.2** Learning curve (Ch. 15):
  - Plot training vs validation score across training sizes
  - Diagnose: is the model underfitting or overfitting?

- **5.3** Feature selection (Ch. 16):
  - VarianceThreshold to remove near-constant features
  - SelectKBest(f_classif) for univariate ranking
  - RFECV with the best ensemble model for interaction-aware selection
  - Permutation importance on test set for post-hoc understanding
  - Compare model performance with full features vs selected features

- **5.4** Hyperparameter tuning (Ch. 15):
  ```python
  param_grid = {
      "clf__n_estimators": [100, 200, 300],
      "clf__max_depth": [3, 5, 7, None],
      "clf__min_samples_leaf": [1, 5, 10]
  }
  gs = GridSearchCV(best_pipeline, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
  gs.fit(X_train, y_train)
  ```

- **5.5** Final model evaluation on held-out test set:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix visualization
  - ROC curve with AUC
  - PR curve (especially important for imbalanced fraud data)

- **5.6** Final model selection justification (markdown cell):
  - Which model won and why
  - Tradeoffs considered (recall vs precision for fraud)
  - How it compares across all metrics

---

### Phase 6: Deployment Integration

**Goal**: Serialize the model and demonstrate the full deployment pipeline following Ch. 17 Sec 17.5-17.7.

Cells to create:

- **6.1** Save model artifact:
  ```python
  import joblib
  joblib.dump(final_pipeline, "fraud_model.sav")
  ```

- **6.2** Save metadata:
  ```python
  metadata = {
      "model_name": "fraud_detection_pipeline",
      "model_version": "1.0.0",
      "trained_at_utc": datetime.utcnow().isoformat(),
      "features": feature_cols,
      "label": "is_fraud",
      "num_training_rows": len(X_train),
      "num_test_rows": len(X_test)
  }
  # Save to model_metadata.json
  ```

- **6.3** Save metrics:
  ```python
  metrics = {
      "accuracy": float(accuracy),
      "f1": float(f1),
      "roc_auc": float(roc_auc),
      "precision": float(precision),
      "recall": float(recall),
      "log_loss": float(ll),
      "classification_report": report
  }
  # Save to metrics.json
  ```

- **6.4** Demonstrate inference (Ch. 17 Sec 17.6 pattern):
  ```python
  # Load the saved model
  model = joblib.load("fraud_model.sav")

  # Query unfulfilled orders from shop.db (simulating live data)
  conn = sqlite3.connect("shop.db")
  df_live = pd.read_sql("SELECT ... FROM orders o JOIN customers c ...", conn)

  # Apply same feature engineering as training
  # (use the prepare_data function from Phase 3)

  # Generate predictions
  df_live["fraud_probability"] = model.predict_proba(X_live)[:, 1]
  df_live["predicted_fraud"] = model.predict(X_live)
  ```

- **6.5** Write predictions to database (Ch. 17 Sec 17.6 pattern):
  ```python
  # Create/update order_predictions_fraud table in shop.db
  cursor.execute("""
      CREATE TABLE IF NOT EXISTS order_predictions_fraud (
          order_id INTEGER PRIMARY KEY,
          fraud_probability REAL,
          predicted_fraud INTEGER,
          prediction_timestamp TEXT
      )
  """)
  # Insert predictions
  ```

- **6.6** Show the query the web app will use:
  ```sql
  SELECT o.order_id, o.order_timestamp, o.total_value,
         c.first_name || ' ' || c.last_name AS customer_name,
         p.fraud_probability, p.predicted_fraud, p.prediction_timestamp
  FROM orders o
  JOIN customers c ON c.customer_id = o.customer_id
  JOIN order_predictions_fraud p ON p.order_id = o.order_id
  WHERE p.predicted_fraud = 1
  ORDER BY p.fraud_probability DESC
  LIMIT 50;
  ```

- **6.7** Deployment integration notes (markdown):
  - How the team's Next.js app will call this (trigger Python script from Node)
  - How this translates to Supabase: same pipeline logic, but connect to PostgreSQL via `psycopg2` or Supabase client instead of sqlite3
  - Scheduled retraining concept: cron job runs ETL -> train -> inference nightly

---

## File Deliverables Checklist

When the notebook is complete, you should have:

| File | Purpose |
|------|---------|
| `is_fraud_pipeline.ipynb` | The CRISP-DM Jupyter notebook (Part 2 deliverable) |
| `fraud_model.sav` | Serialized sklearn Pipeline (preprocessing + model) |
| `model_metadata.json` | Version, timestamp, features, row counts |
| `metrics.json` | All evaluation metrics |
| `shop.db` (modified) | Now contains `order_predictions_fraud` table with predictions |

---

## Integration Notes for Teammates

Your teammates building the web app need to know:

1. **The pipeline writes a table called `order_predictions_fraud`** into shop.db (or the Supabase equivalent) with columns: `order_id`, `fraud_probability`, `predicted_fraud`, `prediction_timestamp`.

2. **The "Run Scoring" button** in the app should execute: `python jobs/run_inference.py` (or a Supabase Edge Function equivalent). This loads `fraud_model.sav`, scores unfulfilled orders, and writes results to `order_predictions_fraud`.

3. **For Supabase migration**: Replace `sqlite3.connect("shop.db")` with a PostgreSQL connection string. The SQL queries and Pipeline logic stay the same. The `CREATE TABLE` statement needs minor PostgreSQL syntax adjustments (e.g., `SERIAL` instead of `INTEGER PRIMARY KEY` if auto-incrementing).

4. **The model file (`fraud_model.sav`)** should be stored in the project's `artifacts/` folder and accessible by whatever environment runs inference (Vercel serverless functions won't run sklearn natively -- consider a separate Python inference endpoint or Supabase Edge Function).

---

## Textbook Code Reference Quick-Map

| Notebook Section | Textbook Section | Key Pattern |
|-----------------|-----------------|-------------|
| Data loading + ETL | 17.4 | sqlite3 + pandas, denormalize to one-row-per-order |
| Univariate exploration | Ch. 6 | `unistats()` function with `is_numeric_dtype` branching |
| Bivariate exploration | Ch. 8 | `bivariate()` with linregress/ANOVA/chi2 + visualizations |
| Data wrangling | Ch. 7 (Sec 7.2-7.3) | `basic_wrangling()`, date features, rare category binning |
| Skewness/outliers | Ch. 7 (Sec 7.4-7.6) | `skew_correct()`, IQR/Z-score outlier detection |
| Missing data | Ch. 7 (Sec 7.5) | MCAR/MAR testing, KNNImputer |
| Pipeline construction | 17.5 | `Pipeline([("prep", ColumnTransformer), ("clf", model)])` |
| Logistic Regression | Ch. 13 | `LogisticRegression(max_iter=1000)` inside Pipeline |
| Ensemble models | Ch. 14 | RandomForest, GradientBoosting with log_loss evaluation |
| Cross-validation | Ch. 15 | `StratifiedKFold` + `cross_validate` with multi-metric scoring |
| Hyperparameter tuning | Ch. 15 | `GridSearchCV` with Pipeline parameter syntax (`clf__param`) |
| Feature selection | Ch. 16 | VarianceThreshold, SelectKBest, RFECV, permutation importance |
| Model serialization | 17.5 | `joblib.dump(pipeline, "model.sav")` |
| Inference | 17.6 | Load model, apply same features, write predictions to DB |
| Scheduling | 17.7 | Cron jobs for ETL -> train -> inference |
