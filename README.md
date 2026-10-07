# 🚗 Nigerian Car Price Prediction (Autochek Nigeria)

An end-to-end machine-learning project that **predicts the asking price of cars in Nigeria**.
It covers scraping 9,304 live listings from [Autochek Nigeria](https://autochek.africa/ng/cars-for-sale),
cleaning the data, exploring it in depth, comparing **11 regression algorithms**, and serving the best model
through an interactive **Streamlit** app.

| | |
|---|---|
| **Data** | 9,304 listings scraped 7 Oct 2026 → 9,230 after cleaning |
| **Best model** | XGBoost: **17.1% average error (MAPE)**, **11.6% median error**, R² 0.904 on log-price |
| **Accuracy** | 71% of test-set predictions within ±20% of the listed price |
| **Test set** | 30% hold-out (2,769 cars) + 5-fold cross-validation on the training set |

---

## Project structure

```
├── app.py                         # Streamlit interface
├── train.py                       # Re-trains the final model -> models/
├── carprice/features.py           # Shared cleaning & feature engineering (used by app + training)
├── scraper/autochek_scraper.py    # Scrapes listings -> data/autochek_cars_ng.csv
├── notebooks/
│   └── Nigerian_Car_Price_Prediction_Autochek.ipynb   # Cleaning, EDA, insights, model comparison
├── data/
│   ├── autochek_cars_ng.csv       # Raw scraped data (30 columns)
│   └── autochek_cars_ng_clean.csv # Cleaned data used for modelling
└── models/
    ├── car_price_model.joblib     # Trained pipeline (preprocessing + XGBoost)
    ├── model_meta.json            # Test metrics, prediction-interval width, versions
    └── model_comparison.csv       # Scores of all 11 algorithms
```

## Quick start

```bash
git clone https://github.com/BabajideAlao-knn/Detailed-car-prediction.git
cd Detailed-car-prediction
pip install -r requirements.txt

streamlit run app.py            # launch the app at http://localhost:8501
```

Other commands:

```bash
python train.py                              # retrain the model from data/autochek_cars_ng.csv
python scraper/autochek_scraper.py           # re-scrape fresh listings (~15–20 min for all pages)
pip install -r requirements-notebook.txt     # extra packages to run the notebook
jupyter notebook notebooks/Nigerian_Car_Price_Prediction_Autochek.ipynb
```

## The Streamlit app

- **💰 Price estimator:** choose make, model, year, condition, mileage, body type, engine and location to get:
  - an estimated price, with a likely range covering 80% of similar listings;
  - a comparison with similar cars currently on the market;
  - a chart of how that car's price changes with model year.
- **📊 Market insights:** median price by make, depreciation curves, and price by condition and body type.
- **🧠 Model performance:** the full comparison of the 11 algorithms on the test set.

### Deploy for free on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** and pick this repository, branch `main`, main file `app.py`.
3. Click **Deploy**. Dependencies install from `requirements.txt`. If the saved model can't be loaded on the server's
   library versions, the app retrains it automatically on first start (it takes a few seconds).

## Methodology

**1. Scraping.** Each Autochek listing page embeds its data as JSON (`__NEXT_DATA__`). The scraper walks
`?page_number=1…405` and saves 30 fields per car: price, year, make/model, condition, mileage, engine, body type, location, financing and more.

**2. Cleaning** (99.2% of rows kept)
- **Duplicates:** removed 53 re-posted duplicate listings.
- **Target leakage:** dropped 3 finance columns (monthly instalment, loan value, old price). The site calculates them *from* the price (correlation 1.00), so they would leak the answer.
- **Price errors:** removed 21 typos, e.g. a ₦20 Mercedes and a ₦350 billion Escalade. The rules were: under ₦1M, over ₦2B, or more than 6× away from the median of the same model.
- **Labels:** standardised make, model and state names; decoded the site's body-type IDs.
- **Mileage:** converted miles to km; impossible values (0 km on used cars, > 1M km) set to missing.

**3. Modelling**
- **Features (17):**
  - car age, mileage, mileage per year, cylinders, grade score;
  - financing, warranty and luxury-brand flags, listing month;
  - make, make + model, condition, body type, transmission, fuel, state, inspection status.
- **Target:** `log(price)`. Prices are heavily right-skewed (skewness 7.9 → 0.8 after the log), and predictions are converted back to naira.
- **Encoding:** one-hot encoding, with categories seen fewer than 5 times grouped together. Numeric features are median-imputed and scaled. All preprocessing sits inside the pipeline, so nothing leaks from the test set.

### Model comparison (30% test set)

| Rank | Model | MAPE | Median error | R² (log) | CV R² (log) |
|---|---|---|---|---|---|
| 🥇 | **XGBoost** | **17.1%** | **11.9%** | **0.905** | 0.904 |
| 🥈 | CatBoost | 17.2% | 11.9% | 0.904 | 0.907 |
| 🥉 | Gradient Boosting | 17.7% | 12.5% | 0.899 | 0.902 |
| 4 | LightGBM | 18.4% | 12.3% | 0.886 | 0.885 |
| 5 | Random Forest | 18.4% | 12.3% | 0.890 | 0.883 |
| 6 | Extra Trees | 18.7% | 12.9% | 0.886 | 0.881 |
| 7 | Linear Regression | 19.4% | 14.2% | 0.888 | 0.880 |
| 8 | Ridge Regression | 19.5% | 14.1% | 0.886 | 0.881 |
| 9 | Lasso Regression | 20.4% | 15.2% | 0.873 | 0.873 |
| 10 | Decision Tree | 22.4% | 14.7% | 0.839 | 0.825 |
| 11 | K-Nearest Neighbours | 23.2% | 16.4% | 0.831 | 0.825 |

XGBoost was then tuned (randomised search, 3-fold CV): the median error improved to 11.6%, and the MAPE stayed at 17.1%.

## Key insights

- **Typical car: ₦18M.** Two-thirds of listings are priced ₦10–40M.
- **Toyota (45%), Mercedes-Benz (18%) and Lexus (17%)** make up about 80% of supply. The **Toyota Camry alone is 16%**.
- **Age is the #1 price driver** (correlation −0.79 with log-price, and by far the top feature in the model):
  - Toyota, Honda and Hyundai lose about **12% of their value per year**;
  - Mercedes-Benz loses about **19%**.
- **The "tokunbo premium":** foreign-used cars cost about **25% more** than locally used cars of the same model and year.
- **SUVs cost about 2× sedans.** Pickups and 8-cylinder vehicles are the priciest segments.
- **Mileage and location matter little** once age and model are known. This suggests odometer readings are often unreliable.

## Limitations

- Prices are **asking prices**, not final sale prices, and reflect the market in October 2026. Naira exchange-rate moves shift import prices, so retrain regularly.
- Trim level, colour, accident history and photos are not captured. These explain much of the remaining ~12% error.
- Rare models, brand-new cars and very cheap or very expensive cars have fewer examples and larger errors.

## Disclaimer

The data was collected from publicly available listing pages for educational and research purposes only.
All listing data belongs to Autochek Africa.
