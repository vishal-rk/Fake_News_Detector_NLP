import sys
sys.path.append(r'C:\Users\visha\Downloads\Fake_News_detector Project')

from tensorflow.keras.models import load_model
from fakenewsdetector.src.model import create_model
from fakenewsdetector.src.preprocessing import preprocess_text, preprocess_and_pad

# Define the variables
voc_size = 5000
vector_len = 40
sent_maxlen = 20

# Load your trained model
model_path = r'C:\Users\visha\Downloads\Fake_News_detector Project\fakenewsdetector\notebooks\trained_model_weights.h5'
trained_model = create_model(voc_size, vector_len, sent_maxlen)
trained_model.load_weights(model_path)

# Test prediction on sample data
sample_text = """ 315 Civilians Killed In Single US Airstrike Have Been Identified,Jessica Purkiss,"Videos 15 Civilians Killed In Single US Airstrike Have Been Identified The rate at which civilians are being killed by American airstrikes in Afghanistan is now higher than it was in 2014 when the US was engaged in active combat operations.   Photo of Hellfire missiles being loaded onto a US military Reaper drone in Afghanistan by Staff Sgt. Brian Ferguson/U.S. Air Force. """
preprocessed_text = preprocess_text(sample_text)
padded_text = preprocess_and_pad([preprocessed_text], voc_size, sent_maxlen)

prediction = trained_model.predict(padded_text)
print("Prediction:", prediction)
