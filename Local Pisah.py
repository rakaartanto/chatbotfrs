#2 Importing Relevant Libraries

import json
import string
import random
import time
import datetime
import webbrowser
import requests
import urllib
#  pip install requests_html
from requests_html import HTML
from requests_html import HTMLSession
import re
from os import system, name
from difflib import SequenceMatcher

import nltk
nltk.download('omw-1.4')

import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model

 
nltk.download ("punkt")
nltk.download ("wordnet")


#Loading the Dataset:
data_file = open('intents.json').read()
data = json.loads(data_file)

qa_dict = []

#4 Creating data_X and data_Y

words = [] #For Bow model/ vocabulary for patterns
classes = [] #for Bow model/ vocabulary for tags

today = datetime.datetime.now()
tahunNow = today.year
angkatanx = ["tahun pertama", "tahun kedua", "tahun ketiga", "tahun keempat", "tahun kelima", "tahun kelima", "tahun kelima"]
semesterx = ["satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan"]
math_patterns = (r'calculate\s(.+)', r'hitung\s(.+)', r'berapa\s(.+)', r'evaluate\s(.+)')

data_X = [] #for storing each pattern

data_y = [] #for storing tag corresponding to each pattern in datax
# Tterating over all the intents

def remove_punct(input_str):
	#print (input_str)
	output_str = re.sub('[^A-Za-z0-9]+', ' ', input_str)
	#print (output_str)

	return ([output_str])

def remove_all_punct(words):
	hasil = []
	for word in words:
		hasil.extend(remove_punct(word))

	return (hasil)

def remove_two_letters(words):
	hasil = []
	for word in words:
		word = word.replace(" ", "")
		if (len(word)>2):
			hasil.extend([word])

	return (hasil)

def fuzzy_search(wordcheck, words, strictness, strictoke):
    rtbesar = 0
    wordpilih = wordcheck
    for word in words:
        similarity = SequenceMatcher(None, word, wordcheck)
        if (rtbesar<similarity.ratio()):
            rtbesar = similarity.ratio()
            wordpilih = word

    if (strictoke==1 and rtbesar>=strictness):
        wordhasil = wordpilih
    else:
        wordhasil = wordcheck
                    
    return wordhasil, rtbesar


def get_right_word(tulisan):
    tulisan = re.sub("\s\s+", " ", tulisan)
    words_list = tulisan.split()
    hasil = []
    tulisanx = ''
    for word in words_list:
        wordpilih, rasio = fuzzy_search(word, words, 0.8, 1)
        # print ('asli=', word, ', fuzzy_search=', wordpilih, ', rasio=', rasio)
        hasil.extend([wordpilih])
        tulisanx = tulisanx + ' ' + wordpilih

    tulisanx = re.sub("\s\s+", " ", tulisanx)
    tulisanx = tulisanx.strip()
            
    return tulisanx, hasil 


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern) # tokenize each pattern
        words.extend(tokens) #and append tokens to words
        data_X.append(pattern) #appending pattern to data_X
        data_y.append(intent["tag"]) # appending the associated tag to each pattern

    # adding the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes .append(intent["tag"])

 
# initialzing lenmatizer to get stem of words

lemmatizer = WordNetLemmatizer()

# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
words = remove_all_punct(words)
words = remove_two_letters(words)

classes = sorted(set(classes))


#5 Text to Numbers

training = []

out_empty = [0] * len(classes)

# creating the bag of words model
for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize (doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # mark the index of class that the current pattern is associated
    # to
 
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    # add the one hot encoded BoW and asociated classes to training
    training.append([bow, output_row])

# shuffle the data and convert it to an array
random.shuffle (training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

#6 Keras Functional API to Create the Neural Network

    # Define the input layer
input_layer = Input(shape=(len(train_X[0]),))
    # Hidden layers
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
    # Output layer
output_layer = Dense(len(train_Y[0]), activation='softmax')(x)
    # Create the model
model = Model(inputs=input_layer, outputs=output_layer)
    # Compile the model with RMSprop optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # Fit the model
print (model.summary())
model.fit(x=train_X, y=train_Y, epochs=500, verbose=2)
    # Save file
model.save("accuracy01.h5")


#7 Preprocessing the input
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]), verbose = 0)[0] # extracting probabilities
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res>thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True) # sorting by values of prob in decreasing order
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]]) # contains labels (tags) for highest probability
    return return_list
    
def evaluate(expr):
	if (len(expr)==0):
		return 0

	hasil = eval(expr)
	return hasil


def get_math_exp1(message):
	hasil = ''
	for pattern in math_patterns:
		# hapus excessive whitespace
		message = re.sub("\s\s+", " ", message)
		if re.search(pattern, message):
			hasil=re.search(pattern, message).group(1)
			break

	return hasil


def get_math_exp(message):
	expr = ''
	regexparse = r'(\d+)|[()+*/-]'
	for a in re.finditer(regexparse, message):
		#print (a.group(0))
		expr = expr + a.group(0)
	return expr    

def parse_results(response):
    
    css_identifier_result = ".tF2Cxc"
    css_identifier_title = "h3"
    css_identifier_link = ".yuRUbf a"
    css_identifier_text = ".VwiC3b"
    
    results = response.html.find(css_identifier_result)

    output = []
    
    for result in results:

        item = {
            'link': result.find(css_identifier_link, first=True).attrs['href']
        }
        
        output.append(item)
        
    return output

def get_source(url):
    #Return the source code for the provided URL. 
    # Args: url (string): URL of the page to scrape.
    # Returns: response (object): HTTP response object from requests_html. 

    try:
        session = HTMLSession()
        response = session.get(url)
        return response

    except requests.exceptions.RequestException as e:
        print(e)
    
def get_google_results(query):    
    query = urllib.parse.quote_plus(query)
    response = get_source("https://www.google.com/search?q=" + query)
    
    return response    
    
def google_search(query):
    response = get_google_results(query)
    return parse_results(response)    

def cetak_hasil_googling(alink):
    hasil = ''
    i = 1
    for info in alink:
        hasil = hasil + str(i) + ',' + info['link'] + '<br>'
        i = i + 1
    return hasil 


def tambah_dict(hasil, tambahan):
	hasil.extend(tambahan)
	return (hasil)

def create_satu_dict(tag1, isi1, tag2, isi2):
	satu_dict = {tag1: isi1, tag2: isi2}
	return ([satu_dict])    


google_patterns = (r'google\s(.+)', r'search\s(.+)', r'cari\s(.+)', r'internet\s(.+)')


def get_google_query(message):
	hasil = ''
	for pattern in google_patterns:
		# hapus excessive whitespace
		message = re.sub("\s\s+", " ", message)
		if re.search(pattern, message):
			hasil=re.search(pattern, message).group(1)
			break

	return hasil

def clear():
    qa_dict = []


def get_response(intents_list, message, intents_json):
    lagi = 0
    message_lagi = ''
    if len(intents_list)==0:
        result = "Mohon maaf saya tidak dapat memahami pertanyaan anda, silahkan bertanya kembali"
        verbose = 1
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice (i["responses"])
                verbose = 1

                if tag=='mahasiswa':
                    # jangan lupa untuk mengaktifkan nodejs terlebih dulu !!!!
                    nrp = re.findall(r'\b\d+\b', message)
                    if (len(nrp)==0):
                        verbose = 1
                    else:
                        #print ('len nrp= ', len(nrp))
                        query = nrp[0]
                        mhs = requests.get("http://localhost:3000/students/%s" % query)
                        mhs = mhs.json()
                        #print (mhs['data'])
                        mhs_list = mhs['data']
                        #print('mhs_list= ', type(mhs_list), 'len= ', len(mhs_list))
                        if (len(mhs_list)>0):
                            mhs_dict = mhs_list[0]
                            #print ('mhs_dict= ', type(mhs_dict))
                            result = 'Nama Mahasiswa : ' + mhs_dict['mhs_nama'] + '<br>' + 'NRP : ' + str(mhs_dict['nrp']) + '<br>' + 'Angkatan : ' + str(mhs_dict['mhs_angkatan']) + '<br>' + 'Mata Kuliah yang Perlu diambil mahasiswa semester berikutnya : ' + mhs_dict['mhs_matkul']
                        verbose = 0

                if tag=='google':
                    # query=input('Enter query: ')
                    query = get_google_query(message)
                    hasil = google_search(query)
                    result = cetak_hasil_googling (hasil)
                    verbose = 0


                if tag=='datetime':
                    result = time.strftime("%A") + '<br>' + time.strftime("%d %B %Y") + '<br>' + time.strftime("%H:%M:%S")
                    verbose = 0

                if tag=='clearscreen':
                    intro = "<br>" + header + footer
                    verbose = 0

                    
                # hitung rumus
                if (tag=='rumus'):
                    expr = get_math_exp(message)
                    hasil = evaluate(expr)
                    result = str(hasil)
                    verbose = 0
                    
                # tambahkan tag 'semester' <-------------
                
                if (tag in ['angkatan']):
                    # angkatan yyyy
                    tahunx = re.findall(r'\b\d+\b', message)
                    if (len(tahunx)==0):
                        verbose = 1
                    else:
                        tahun = tahunx[0]
                        selisih = tahunNow - int(tahun)
                        if (selisih>=0 and selisih<7):
                            message_lagi = angkatanx[selisih]
                            lagi = 1

                        # print ('tahun ke= ', selisih, 'msg= ', message_lagi)
                        # cek ulang bulan jan sd Juli tahun berjalan untuk perhitungan tahun ke-
                    verbose = 0

                    
                break


    return result, verbose, lagi, message_lagi
 

 
def clearx():

    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')
 

#8 Interacting with the chatbot
#===============================================================


from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# Read the contents of header.html file
with open('header.html', 'r') as header_file:
    header = header_file.read()

# Read the contents of footer.html file
with open('footer.html', 'r') as footer_file:
    footer = footer_file.read()

#Experiment
    
@app.route("/")
def hello():
    intro = "<br>" + header + footer
    return intro

def html_table(lol):
  #Ori border=1
  hasil = '<table border="1">'
  for sublist in lol:
    hasil = hasil + ' <div class="chat-container"><h4 class="username-message">User</h4><div class="message user-message">' + sublist['question']
    hasil = hasil + ' </div><h4 class="botname-message">FRS 101 Chatbot</h4><div class="message bot-message">' + sublist['answer']
    hasil = hasil + ' </div></div>'

  hasil = hasil + '</table>'
  return (hasil)

def gabung_respons(result, qa_dict):
    tabel = html_table(qa_dict)
    hasil =  "<br>" + header + '<div class="table-container">' + tabel + '</div>' + footer
    return hasil

@app.route('/send_data', methods = ['POST'])
def get_data_from_html():
        global qa_dict
        message = request.form['message']
        message, hasil = get_right_word(message)
        intents = pred_class(message, words, classes)
        result, verbose, lagi, message_lagi = get_response(intents, message, data)
        if (lagi==1):
            intents = pred_class(message_lagi, words, classes)
            result, verbose, lagi, message_lagi = get_response(intents, message_lagi, data)
      
        qa_dict = tambah_dict(qa_dict, create_satu_dict('question', message, 'answer', result))
        return gabung_respons(result, qa_dict)


if __name__ == "__main__":
    app.run()

    