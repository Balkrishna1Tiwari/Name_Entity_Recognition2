from flask import Flask, request, render_template
from N_E_R.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not text:
        return "Please provide text for prediction."

    try:
        obj = PredictionPipeline()
        prediction = obj.predict(text)
        return render_template('result.html', text=text, prediction=prediction)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
