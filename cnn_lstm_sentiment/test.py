from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import json

# load the tokenizer and the model
with open("./keras_tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("./yelp_sentiment_model.hdf5")

# replace with the data you want to classify
print('\n\n')
while True:
    print('*--------------------------------------------*')
    keyword = input('keyword 입력 : ')
    if keyword == 'C':
        break
    newtexts = [keyword]

    # note that we shouldn't call "fit" on the tokenizer again
    sequences = tokenizer.texts_to_sequences(newtexts)
    data = pad_sequences(sequences, maxlen=300)

    # get predictions for each of your new texts
    predictions = model.predict(data)
    print(predictions)
    print('*--------------------------------------------*')
#jsonPredictions = json.dumps(predictions)
#print(jsonPredictions)