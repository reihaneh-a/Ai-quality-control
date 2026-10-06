from flask import Flask, render_template, jsonify
from tensorflow import keras # type: ignore
import joblib
import random
import csv
import os

app = Flask(__name__)
model = keras.models.load_model("model.keras")
scaler = joblib.load("scaler.joblib")

counter = 0
run_id = 1
running = True
stats = {"pass": 0, "defect": 0, "correct": 0}
CSV_FILE = "production_results.csv"
CSV_FIELDS = [
    "run_id", "product_id", "temperature", "pressure", "vibration", "speed", "weight",
    "prediction", "defect_probability", "actual_result", "match"
]


def make_product():
    global counter
    counter += 1
    abnormal = random.random() < 0.25
    if abnormal:
        values = [random.uniform(74, 82), random.uniform(34, 47), random.uniform(3.2, 5.2), random.uniform(105, 140), random.uniform(482, 516)]
    else:
        values = [random.uniform(68, 73), random.uniform(38, 42), random.uniform(1.5, 2.8), random.uniform(115, 125), random.uniform(495, 505)]
    return {
        "run_id": run_id,
        "product_id": f"LIVE-{counter:03d}",
        "temperature": round(values[0], 2), "pressure": round(values[1], 2),
        "vibration": round(values[2], 2), "speed": round(values[3], 2), "weight": round(values[4], 2),
        "actual_result": "DEFECT" if abnormal else "PASS",
    }


def ensure_csv_schema():
    """Upgrade an older CSV without losing any saved production rows."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return

    with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        old_fields = reader.fieldnames or []
        if old_fields == CSV_FIELDS:
            return
        old_rows = list(reader)

    # Preserve all legacy rows. Rows created before run_id existed belong to Run 1.
    upgraded_rows = []
    for row in old_rows:
        upgraded = {field: row.get(field, "") for field in CSV_FIELDS}
        if not upgraded["run_id"]:
            upgraded["run_id"] = "1"
        upgraded_rows.append(upgraded)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(upgraded_rows)


def save_result(product):
    ensure_csv_schema()
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        writer.writerow({field: product[field] for field in CSV_FIELDS})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify(running=running, run_id=run_id, counter=counter)


@app.route("/stop", methods=["POST"])
def stop():
    global running
    running = False
    return jsonify(running=running)


@app.route("/continue", methods=["POST"])
def continue_run():
    global running
    running = True
    return jsonify(running=running)


@app.route("/new-run", methods=["POST"])
def new_run():
    global counter, run_id, running, stats
    run_id += 1
    counter = 0
    stats = {"pass": 0, "defect": 0, "correct": 0}
    running = True
    # IMPORTANT: production_results.csv is never deleted or overwritten.
    return jsonify(running=running, run_id=run_id)


@app.route("/live")
def live():
    if not running:
        return jsonify(running=False, run_id=run_id)

    product = make_product()
    features = ["temperature", "pressure", "vibration", "speed", "weight"]
    values = [[product[key] for key in features]]
    values_scaled = scaler.transform(values)
    probability = float(model.predict(values_scaled, verbose=0)[0][0])
    prediction = "DEFECT" if probability >= 0.5 else "PASS"
    match = prediction == product["actual_result"]

    stats["defect" if prediction == "DEFECT" else "pass"] += 1
    stats["correct"] += int(match)
    inspected = stats["pass"] + stats["defect"]

    product.update(
        prediction=prediction, defect_probability=round(probability, 6),
        match="YES" if match else "NO", inspected=inspected,
        pass_count=stats["pass"], defect_count=stats["defect"], correct_count=stats["correct"],
        live_accuracy=round(stats["correct"] / inspected * 100, 1), running=True,
    )
    save_result(product)
    return jsonify(product)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
