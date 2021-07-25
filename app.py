from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chatbot.html")

def clean_up_words(userText):
	sentence_words = nltk.word_tokenize(userText)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	return sentence_words


def bag_of_words(userText, words, show_details=False):
	sentence_words = clean_up_words(userText)
	bag = [0]*len(words)
	for s in sentence_words:
		for i,word in enumerate(words):
			if word == s:
				bag[i] = 1
				if show_details:
					print("found in bag: %s" % word)
	return(np.array(bag))


def predict_class(userText):
	p = bag_of_words(userText, words, show_details=False)
	res = model.predict(np.array([p]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
	return return_list

def getResponse(req, intents_json):
	tag = req[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if(i['tag'] == tag):
			result = random.choice(i['responses'])
			break
	return result


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip()
    # return str(bot.get_response(userText))
    if userText != '':
    	req = predict_class(userText)
    	resp = getResponse(req, intents)
    	# print(resp)
    	return resp



if __name__ == "__main__":
    app.run(debug = True)