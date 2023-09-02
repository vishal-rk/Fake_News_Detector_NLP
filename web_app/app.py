# web_app/app.py
import sys

# Append project directory to Python path
sys.path.append(r'C:\Users\visha\Downloads\Fake_News_detector Project')

from flask import Flask, render_template, request, jsonify
from fakenewsdetector.src import model
from fakenewsdetector.src.preprocessing import preprocess_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load your trained model
voc_size = 5000
vector_len = 40
sent_maxlen = 20
trained_model = model.create_model(voc_size, vector_len, sent_maxlen)
trained_model.load_weights(r'C:\Users\visha\Downloads\Fake_News_detector Project\fakenewsdetector\notebooks\trained_model_weights.h5')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data['news_text']
    
    try:
        preprocessed_text = preprocess_text(news_text)
        padded_text = model.preprocess_and_pad([preprocessed_text], voc_size, sent_maxlen)
    
        prediction = trained_model.predict(padded_text)
        prediction_label = "Real" if prediction[0][0] > 0.5 else "Fake"
    except Exception as e:
        print('Error making prediction:', e)
        return jsonify({"error": str(e)})
    
    return jsonify({"prediction": prediction_label})


if __name__ == '__main__':
    app.run(debug=True)


