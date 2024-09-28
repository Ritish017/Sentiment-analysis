from flask import Flask, render_template, request
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define a route for home
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    review = request.form['review']

    # Combine title and review into one string
    full_text = title + ' ' + review

    # Transform the input using the vectorizer
    transformed_text = vectorizer.transform([full_text])

    # Predict sentiment using the model
    prediction = model.predict(transformed_text)[0]

    # Return the prediction result to the front-end
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)
