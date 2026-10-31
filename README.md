# NYC Airbnb Room Type Predictor

A machine learning web app that predicts whether an NYC Airbnb listing is an **Entire home/apt**, a **Private room**, or a **Shared room**, based on details such as location, price, reviews and availability.

The project has three parts:

1. **Notebook**: EDA, preprocessing, model comparison, tuning, and export of the trained pipeline.
2. **FastAPI backend**: serves the saved pipeline through a `/predict` endpoint.
3. **Frontend**: a static HTML/CSS/JS page where you enter listing details and see the prediction and class probabilities, visualised as a city skyline.

## Project structure

 House_Type_prediction.ipynb | Model_Pipeline.pkl | main.py | index.html | style.css | script.js | requirements.txt | .gitignore | README.md

## Model

- **Dataset:** [New York City Airbnb Open Data (AB_NYC_2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), downloaded in the notebook with `kagglehub`.
- **Target:** `room_type` (3 classes).
- **Features (10):** `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `neighbourhood_group`, `neighbourhood`.
- **Preprocessing:** median imputation + standard scaling for numeric features; most-frequent imputation + one-hot encoding for categorical features. `price` and `minimum_nights` are capped at the 99th percentile before training.
- **Models compared (3-fold CV):**

| Model               | Accuracy | Macro F1 |
|---------------------|----------|----------|
| Logistic Regression | 0.659    | 0.522    |
| Decision Tree       | 0.782    | 0.647    |
| Random Forest       | 0.851    | 0.715    |
| Gradient Boosting   | 0.850    | 0.705    |

- **Final model:** Random Forest (`class_weight='balanced'`), tuned with `RandomizedSearchCV`. Best parameters: `n_estimators=150`, `min_samples_split=10`, `max_depth=None`.
- **Test set performance:** accuracy **0.856**, macro F1 **0.740**.

## Getting started

### 1. Install dependencies

pip install -r requirements.txt

### 2. (Optional) Retrain the model

Open `House_Type_prediction.ipynb` and run all cells. The last cell writes a fresh `Model_Pipeline.pkl`. Downloading the dataset with `kagglehub` may require Kaggle credentials.

### 3. Start the API

Run this from the project folder, since `main.py` loads `Model_Pipeline.pkl` using a relative path:

uvicorn main:app --reload

The API runs at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Open the frontend

Open `index.html` in your browser (double-click it, or use the VS Code Live Server extension). A status indicator in the Prediction panel shows whether the page can reach the API.

If your API runs somewhere other than `http://127.0.0.1:8000`, change `API_BASE_URL` at the top of `script.js`.

## API reference

### `POST /predict`

**Request body**

| Field                            | Type   | Constraints   |
|----------------------------------|--------|---------------|
| `latitude`                       | float  | -90 to 90     |
| `longitude`                      | float  | -180 to 180   |
| `price`                          | float  | > 0           |
| `minimum_nights`                 | int    | 1 to 365      |
| `number_of_reviews`              | int    | >= 0          |
| `reviews_per_month`              | float  | >= 0          |
| `calculated_host_listings_count` | int    | >= 0          |
| `availability_365`               | int    | 0 to 365      |
| `neighbourhood_group`            | string | non-empty     |
| `neighbourhood`                  | string | non-empty     |

**Example**

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7484,
    "longitude": -73.9857,
    "price": 120,
    "minimum_nights": 2,
    "number_of_reviews": 84,
    "reviews_per_month": 2.3,
    "calculated_host_listings_count": 1,
    "availability_365": 210,
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Midtown"
  }'

**Response**

{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.91, 0.01, 0.08]
}

`Probability` follows the model's `classes_` order, which scikit-learn sorts alphabetically: `Entire home/apt`, `Private room`, `Shared room`.

## Tech stack

Python, FastAPI, Uvicorn, scikit-learn, pandas, joblib, and vanilla HTML/CSS/JavaScript.

## Known limitations

- The dataset is from 2019, so prices and listings are dated.
- Unseen neighbourhood names are ignored by the encoder (`handle_unknown='ignore'`), which can reduce accuracy for them.
- The `Shared room` class is rare in the data, so its predictions are the least reliable (macro F1 is much lower than accuracy).
- CORS is open to all origins (`allow_origins=["*"]`), which is fine for local development but should be restricted in production.
