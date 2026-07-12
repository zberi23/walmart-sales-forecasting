# Walmart Store Sales Forecasting — ML Final Project

[Kaggle Competition: Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) — 45 მაღაზიის, ~80 დეპარტამენტის ყოველკვირეული გაყიდვების პროგნოზი. მეტრიკა: **Weighted MAE** (სადღესასწაულო კვირებს ×5 წონა).

## გუნდი და როლების განაწილება

| მოდელი | ვინ ატრენინგებდა | ლოგირება |
|--------|-------------------|----------|
| XGBoost | Zaqaria | MLflow / DagsHub |
| N-BEATS | Zaqaria | WandB |
| PatchTST | Zaqaria | WandB |
| ARIMA/SARIMA | Zaqaria | MLflow / DagsHub |
| LightGBM | Giga | MLflow / DagsHub |
| DLinear | Giga | WandB |
| Prophet | Giga | MLflow / DagsHub |
| TimesFM (bonus) | Giga | WandB |

## Tracking Links

- **MLflow (DagsHub) — ყველა tree-based + classical:** https://dagshub.com/zberi23/walmart-forecasting.mlflow
- **WandB — Zaqaria-ს DL მოდელები (N-BEATS, PatchTST):** https://wandb.ai/zberi23_ml/walmart-forecasting
- **WandB — Giga-ს DL მოდელები (DLinear, TimesFM):** https://wandb.ai/gbera23-free-university-of-tbilisi-/walmart-forecasting

## TL;DR — 8 მოდელის შედარება

ვალიდაცია — ბოლო 12 კვირა (chronological split, per-series DL-ისთვის, global 90/10 tree-ისთვის).

| Rank | Model | Category | Val WMAE | Type |
|------|-------|----------|----------|------|
| 🥇 1 | **XGBoost** | Tree-based | **769.75** | Trained |
| 🥈 2 | LightGBM | Tree-based | 1098.93 | Trained |
| 🥉 3 | TimesFM | Foundation Model | 1309.76 | **Zero-shot** |
| 4 | N-BEATS | DL (feed-forward) | 1378.04 | Trained |
| 5 | PatchTST | DL (transformer) | 1420.44 | Trained |
| 6 | DLinear | DL (linear) | 1494.80 | Trained |
| 7 | SARIMA | Classical | 7012.96¹ | Trained |
| 8 | Prophet | Classical | 10689.52¹ | Trained |

¹ *ARIMA/Prophet ტოპ 5 (Store, Dept) სერიაზე მხოლოდ — per-series მოდელი, 3000+ სერიაზე ტრენინგი პრაქტიკული არ იყო.*

**გამარჯვებული:** XGBoost — მოწინავე შედეგი. Custom sklearn Pipeline, რომელიც raw CSV-ს (`test.csv.zip`) იღებს და პროგნოზებს აბრუნებს პირდაპირ.

## Kaggle Submission

XGBoost final submission Kaggle-ის private leaderboard-ზე:

| Score Type | WMAE |
|-----------|------|
| **Private Score** | **2904.49** |
| Public Score | 2806.83 |

*გავითვალისწინოთ: val WMAE (769) და Kaggle private score (2904) განსხვავებულია, რადგან test set არის 2012 წლის ბოლო და 2013 წლის დასაწყისი/შუა პერიოდი (2012-11 → 2013-07), სხვა distribution-ით და holiday-ების ჩართულობით.*

## Repo Structure

```
walmart-forecasting/
│
├── eda_and_feature_engineering.ipynb   # EDA + shared preprocessing
│
├── model_experiment_XGBoost.ipynb      # Zaqaria — winner model
├── model_experiment_NBEATS.ipynb       # Zaqaria
├── model_experiment_PatchTST.ipynb     # Zaqaria
├── model_experiment_ARIMA.ipynb        # Zaqaria — classical
│
├── model_experiment_LightGBM.ipynb     # Giga
├── model_experiment_DLinear.ipynb      # Giga
├── model_experiment_Prophet.ipynb      # Giga — classical
├── model_experiment_TimesFM.ipynb      # Giga — bonus
│
├── model_inference.ipynb               # საბოლოო submission generation
│
└── README.md                           # ეს ფაილი
```

Kaggle-ის data (`train.csv.zip`, `test.csv.zip`, `stores.csv`, `features.csv.zip`) რეპოში არ არის — ჩამოტვირთვის ინსტრუქცია EDA notebook-ის Setup სექციაშია.

## თითო მოდელის მოკლე აღწერა

### 1. XGBoost (Zaqaria) — გამარჯვებული

**5 MLflow runs:** `XGBoost_Cleaning`, `XGBoost_Feature_Selection`, `XGBoost_CrossValidation`, `XGBoost_HyperparameterTuning`, `XGBoost_Final`

- Custom `WalmartPreprocessor` sklearn Transformer Pipeline-ის შიგნით (merge stores + features, fillna, feature engineering)
- Time-based train/val split (ბოლო 10% val)
- 3 feature set-ის შედარება (all / no_markdown / core) — `core` (17 ფიჩერი) გაიმარჯვა
- TimeSeriesSplit 5-fold CV
- Optuna 30 trials → **val WMAE 769.75**
- **Model Registry: `walmart_xgboost` v1 → Production**

**რას ვხედავთ:** `core` feature set-მა (მხოლოდ date + store metadata) აჯობა `all`-ს (Temperature, Fuel_Price, CPI, Unemployment-ის ჩათვლით). ეს ცვლადები პირდაპირი (contemporaneous) სახით noise-ს ქმნიდნენ boosting მოდელისთვის. მათი lagged (წინა კვირების) ვერსიები შესაძლოა დახმარებოდა, მაგრამ ეს ცალკე feature engineering ექსპერიმენტი იქნება.

### 2. LightGBM (Giga)

**5 MLflow runs:** იგივე სტრუქტურა, რაც XGBoost-ს.

- val WMAE **1098.93**
- `core` feature set-მა ისევ გაიმარჯვა (XGBoost-ის მსგავსად)
- Leaf-wise growth-მა XGBoost-ის level-wise-ს ვერ აჯობა Walmart-ისთვის
- **Model Registry: `walmart_lightgbm` v1 → Staging**

### 3. N-BEATS (Zaqaria)

**5 WandB runs:** `NBEATS_Baseline`, `NBEATS_Interpretable`, `NBEATS_LongContext`, `NBEATS_Final`, `NBEATS_ModelArtifact`

- Global model — ერთი მოდელი 2934 (Store, Dept) time series-ისთვის
- საუკეთესო კონფიგი: **Long Context** (52-week input + trend/seasonality stacks) — val WMAE **1378.04**
- Interpretable mode (polynomial trend + Fourier seasonality basis) Walmart-ის yearly cycles-ს კარგად ითვისებს

### 4. PatchTST (Zaqaria)

**5 WandB runs:** `PatchTST_Baseline`, `PatchTST_LargerPatch`, `PatchTST_Deeper`, `PatchTST_Final`, `PatchTST_ModelArtifact`

- Transformer-based (patching + channel independence)
- საუკეთესო: **Deeper** (6 encoder layers, hidden=256, 8 heads) — val WMAE **1420.44**
- PatchTST-ის transformer complexity Walmart-ის სპეციფიკურ patterns-ს შედარებით ცოტა ეხმარება

### 5. DLinear (Giga)

**5 WandB runs:** Baseline / Longer Window / Longer Training / Final / Model Artifact

- უბრალო architecture: მხოლოდ ორი Linear layer (trend decomposition + seasonal decomposition)
- საუკეთესო: Longer Training (1500 steps, LR=5e-4) — val WMAE **1494.80**
- **სიმარტივის თეზისი დადასტურდა**: DLinear PatchTST-ს მხოლოდ 5%-ით ჩამორჩება, მაგრამ 3-4x სწრაფად ტრენინგდება

### 6. TimesFM (Giga) — BONUS

**3 WandB runs:** ZeroShot / LongContext / Final

- Google-ის **pretrained** foundation model (500M params, decoder-only transformer)
- **Zero-shot forecasting** — არავითარი training არ ხდება!
- საუკეთესო: **LongContext** (256-week context) — val WMAE **1309.76**
- **აჯობა 3 trained DL მოდელს** (N-BEATS, PatchTST, DLinear) მიუხედავად იმისა, რომ Walmart-ის data-ს არასოდეს ნახა
- Foundation model paradigm-ის კარგი მაგალითი time-series-ისთვის

### 7. ARIMA/SARIMA (Zaqaria)

**4 MLflow runs:** `ARIMA_Stationarity`, `ARIMA_Baseline`, `SARIMA_Seasonal`, `ARIMA_Final`

- Box-Jenkins მიდგომა — ADF + KPSS ტესტები, ACF/PACF ანალიზი
- Recommended: `d=1, D=1, s=52` (double differencing needed)
- ARIMA(1,1,1) baseline WMAE **12302.90**
- SARIMA(1,1,1)(1,1,1)_52 WMAE **7012.96** — **43% შემცირება**. სეზონურობის დამატება ცხადად ცვლის შედეგს
- ტესტი მხოლოდ ტოპ 5 (Store, Dept) სერიაზე (per-series model, 3000+ სერიაზე ტრენინგი პრაქტიკული არ იყო — SARIMA CPU-ზე თითო სერიისთვის რამდენიმე წამს/წუთს იღებს, 3000+ სერიაზე მთელი დღე დასჭირდებოდა)

### 8. Prophet (Giga)

**4 MLflow runs:** `Prophet_Baseline`, `Prophet_Holidays`, `Prophet_Tuned`, `Prophet_Final`

- Facebook Prophet (additive: trend + seasonality + holidays)
- Walmart-specific holidays დამატებული (Super Bowl, Labor Day, Thanksgiving, Christmas)
- Baseline WMAE 10936.63 → Holidays WMAE **10689.52**
- ინტერპრეტირებადი components (plot_components()), მაგრამ SARIMA-ს კონკრეტულ Walmart data-ზე ვერ აჯობა

## საბოლოო Pipeline (XGBoost)

Best model უნდა იმუშაოს **raw** test set-ზე — preprocessing Pipeline-ის შიგნით ჩაშენდა. აი ჩვენი sklearn Pipeline:

```
raw test.csv (Store, Dept, Date, IsHoliday)
         ↓
WalmartPreprocessor
    ├── merge stores.csv + features.csv
    ├── fillna: MarkDown→0, CPI/Unemployment→ffill by Store
    └── feature engineering:
        ├── Date features (Year, Month, Week, Day_of_Year, Quarter)
        ├── Walmart holidays (Super_Bowl, Labor_Day, Thanksgiving, Christmas)
        ├── Cyclical encoding (Month_Sin/Cos, Week_Sin/Cos)
        └── Type encoding (A/B/C → 0/1/2)
         ↓
FeatureSelector (best 17 ფიჩერი)
         ↓
XGBoost (Optuna-tuned hyperparameters)
         ↓
predictions
```

**inference notebook-ში:**
```python
xgb_pipeline = joblib.load('models/xgboost_pipeline.pkl')
test_predictions = xgb_pipeline.predict(test_raw)  # raw CSV პირდაპირ!
```

## Model Registry Status

DagsHub Model Registry-ში:

| Model Name | Version | Stage |
|-----------|---------|-------|
| `walmart_xgboost` | v1 | **Production** |
| `walmart_lightgbm` | v1 | Staging |

## Setup და Reproduction

### 1. Prerequisites

```
Colab (Free/Pro), Google Drive, DagsHub account, WandB account, Kaggle account
```

### 2. Drive Structure

```
/content/drive/MyDrive/walmart/
├── data/
│   ├── train.csv.zip
│   ├── test.csv.zip
│   ├── stores.csv
│   ├── features.csv.zip
│   └── sampleSubmission.csv.zip
├── models/       (თვითონ იქმნება)
└── submissions/  (თვითონ იქმნება)
```

### 3. Secrets (Colab)

- `KAGGLE_USERNAME` + `KAGGLE_KEY` — Kaggle API (მონაცემების ჩამოტვირთვისთვის)
- `WANDB_API_KEY` — WandB tracking-ისთვის
- DagsHub — `dagshub.init()` browser-ის auth-ს ითხოვს

### 4. თანმიმდევრობა

1. `eda_and_feature_engineering.ipynb` — EDA + preprocessed CSV-ების შენახვა Drive-ზე
2. თითო `model_experiment_*.ipynb` — ცალკე ტრენინგდება
3. `model_inference.ipynb` — საბოლოო submission-ის generation

**GPU საჭიროა:** N-BEATS, PatchTST, DLinear, TimesFM — Runtime → T4 GPU  
**CPU საკმარისია:** XGBoost, LightGBM, ARIMA, Prophet

## საერთო დასკვნები

1. **Tree-based მოდელები Walmart-ისთვის საუკეთესოა** — XGBoost 769.75 vs საუკეთესო DL model 1378 (N-BEATS). Structured tabular data-ზე gradient boosting-ს DL-მა ჯერ ვერ აჯობა.

2. **Foundation models-ის მოსვლა** — TimesFM zero-shot-ით (არავითარი training!) აჯობა 3 trained DL მოდელს. ეს არის მიმდინარე time-series ML-ის დამახასიათებელი მიმართულება.

3. **სიმარტივე ხშირად უპირატესია** — DLinear (2 linear layer) PatchTST-ს (transformer) მხოლოდ 5%-ით ჩამორჩება. "Are Transformers Effective for Time Series?" — ხშირად არა.

4. **Domain knowledge > automatic feature discovery** — SARIMA-ს explicit s=52 seasonality-მ Prophet-ის automatic detection-ს აჯობა Walmart-ისთვის.

5. **Core features > all features** — XGBoost და LightGBM ორივემ დაადასტურა, რომ Temperature, Fuel_Price, CPI, Unemployment noise-ს ქმნიდნენ. Date + Store metadata საკმარისია.

6. **Custom Pipeline design ღირს** — Custom sklearn Transformer-ი (`WalmartPreprocessor`) preprocessing-ს Pipeline-ის შიგნით ატარებს, რაც raw test set-ზე inference-ს ამარტივებს (მხოლოდ `predict(test_raw)` — არავითარი ცალკე preprocessing გაშვება).
