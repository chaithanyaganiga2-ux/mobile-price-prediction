import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

data = pd.read_csv("mobile_phone_prices.csv")

X = data[[
    "ram_gb",
    "battery_mah",
    "storage_gb",
    "camera_mp",
    "screen_size_inch",
    "five_g"
]]
y = data["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None

    if request.method == "POST":
        ram = float(request.form["ram"])
        battery = float(request.form["battery"])
        storage = float(request.form["storage"])
        camera = float(request.form["camera"])
        screen = float(request.form["screen"])
        five_g = int(request.form["five_g"])

        input_data = pd.DataFrame(
            [[ram, battery, storage, camera, screen, five_g]],
            columns=[
                "ram_gb",
                "battery_mah",
                "storage_gb",
                "camera_mp",
                "screen_size_inch",
                "five_g"
            ]
        )

        predicted_price = round(model.predict(input_data)[0])

    return render_template(
        "index.html",
        predicted_price=predicted_price,
        mae=round(mae, 2),
        r2=round(r2, 3)
    )


if __name__ == "__main__":
    app.run(debug=True)
