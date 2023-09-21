# Step 6: Building the Web Application

from flask import Flask, render_template, request, jsonify
import joblib  # For loading the trained model
import pandas as pd  # For data preprocessing

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("your_trained_model.pkl")  # Replace with your model file

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input data from the web form
        input_data = {
            "gender": request.form["gender"],
            "class": request.form["class"],
            "attended_classes": int(request.form["attended_classes"]),
            "total_classes": int(request.form["total_classes"]),
            "math_score": float(request.form["math_score"]),
            # Add other input fields here
        }

        # Preprocess the input data to match the model's requirements
        input_df = pd.DataFrame([input_data])
        # You may need to perform the same preprocessing steps as in your model building code

        # Use the model to make predictions
        prediction = model.predict(input_df)

        # Convert the prediction to a human-readable label
        result = "Pass" if prediction[0] == 1 else "Fail"

        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
