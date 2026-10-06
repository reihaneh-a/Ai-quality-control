# AI Quality Control – Live Defect Prediction

A small end-to-end machine learning project that predicts whether a manufactured product is **PASS** or **DEFECT** from five sensor readings, and shows the predictions live in a Flask web dashboard.

## Features

- Neural network classifier (Keras) trained on 3,000 historical production records
- Live simulation: the web app generates new products (about 25% abnormal) and classifies them in real time
- Running statistics in the dashboard: pass/defect counts and live accuracy
- Stop / continue / start a new run from the UI
- Every prediction is appended to `production_results.csv` with its run ID

## Input features

| Feature       | Description            |
|---------------|------------------------|
| `temperature` | Machine temperature    |
| `pressure`    | Process pressure       |
| `vibration`   | Vibration level        |
| `speed`       | Line speed             |
| `weight`      | Product weight         |

Target: `actual_result` (`PASS` / `DEFECT`)

## Project structure

```
.
├── app.py                              # Flask app: live simulation + prediction API
├── train_quality_model.ipynb           # Data exploration, training and evaluation
├── data/
│   └── historical_quality_data.csv     # Training data (3,000 rows)
├── templates/
│   └── index.html                      # Live dashboard UI
├── model.keras                         # Trained Keras model
├── scaler.joblib                       # Fitted StandardScaler
└── requirements.txt
```

## Model

A simple fully connected network:

```
Input (5) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

Features are standardized with `StandardScaler`, and the model is trained for 50 epochs with binary cross-entropy. A probability of 0.5 or higher is classified as `DEFECT`.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the web app

```bash
python app.py
```

Then open <http://127.0.0.1:5001> in your browser.

### 4. (Optional) Retrain the model

Open `train_quality_model.ipynb` in Jupyter or VS Code and run all cells. This regenerates `model.keras` and `scaler.joblib`.

## API endpoints

| Method | Endpoint     | Description                                     |
|--------|--------------|-------------------------------------------------|
| GET    | `/`          | Dashboard page                                  |
| GET    | `/live`      | Generate one product, predict it and return JSON |
| GET    | `/status`    | Current run state                               |
| POST   | `/stop`      | Pause the live simulation                       |
| POST   | `/continue`  | Resume the live simulation                      |
| POST   | `/new-run`   | Start a new run (results file is kept)          |

## Tech stack

Python · TensorFlow / Keras · scikit-learn · Flask · pandas · Jupyter

## License

This project is licensed under the MIT License.
