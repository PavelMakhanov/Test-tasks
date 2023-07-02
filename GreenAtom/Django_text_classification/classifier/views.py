from django.shortcuts import render
from django.http import HttpResponse
import joblib
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from string import punctuation
import re
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

denial = ["no", "not", "never", "neither", "nor", "none", "nobody", "nowhere", "nothing",
          "nonexistent", "no way", "no one", "no more", "not anymore",
          "not at all", "not even", "not yet", "not necessarily",
          "not necessarily true", "not necessarily the case", "don't",
          "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
          "shouldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't",
          "hasn't", "haven't", "hadn't", "isn't it", "aren't you", "wasn't he",
          "weren't they", "hasn't she", "haven't we", "hadn't i", "isn't there",
          "aren't they", "wasn't it", "weren't you", "hasn't he", "haven't i",
          "hadn't she", "isn't that", "aren't we", "wasn't she", "weren't there",
          "hasn't it", "haven't they", "hadn't we", "isn't he", "aren't there", "wasn't there",
          "weren't we", "hasn't she", "haven't you", "hadn't they"]
stop_words = list(set(stopwords.words('english')) - set(denial))
m = pymorphy2.MorphAnalyzer()

# Загрузка модели
with open('classifier/models/LR_model.pkl', 'rb') as file:
    model = joblib.load(file)

with open('classifier/models/tfidf.pkl', 'rb') as file:
    tfidf = joblib.load(file)

# функция возвращает только слова
def words_only(text):
    if re.findall("[A-Za-z]+", text):
        return text

def pre_process(text):
    doc_out = ""
    for token in word_tokenize(text):
        if token not in stop_words and token not in punctuation:
            word = words_only(m.parse(token)[0].normal_form)
            if word:
                doc_out = doc_out + " " + word.strip('`')
    return doc_out

def classify_text(request):
    if request.method == 'POST':
        text = request.POST['text']
        text_clear_vec = tfidf.transform([pre_process(text)])
        prediction = model.predict(text_clear_vec)[0]
        if prediction == 0:
            prediction = 'Negative review'
        else:
            prediction = 'Positive review'
        review_score = round((model.predict_proba(text_clear_vec)[0][1])*10,1)
        return render(request, 'classifier/result.html', {'prediction': prediction, 'review_score': review_score})
    return render(request, 'classifier/classify.html')
