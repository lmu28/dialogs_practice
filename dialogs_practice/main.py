import pandas as pd
import re
import string
import pymorphy2
import nltk
import json
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import word_tokenize, ngrams
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


def clean_string(text):
    text = re.split(' |:|\.|\(|\)|,|"|;|/|\n|\t|-|\?|\[|\]|!', text)
    text = ' '.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords_list])
    return text


def string_to_normal_form(string):
    string_lst = string.split()
    for i in range(len(string_lst)):
        string_lst[i] = morph.parse(string_lst[i])[0].normal_form
    string = ' '.join(string_lst)
    return string


def json_to_csv(filename, dialog_id_list, text_list):
    json_string = open(f'dialogs/{filename}', 'r', encoding='utf-8').read()
    parsed_string = json.loads(json_string)
    str = ""
    for i in range(0, len(parsed_string)):
        str += parsed_string[i]["alternatives"][0]["text"] + " "

    text = string_to_normal_form(clean_string(str))
    dialog_id_list.append(len(dialog_id_list) + 1)
    text_list.append(text)


def create_csv_file():
    dialog_id_list = []
    text_list = []
    filenames = os.listdir(os.getcwd() + "/dialogs")
    for filename in filenames:
        json_to_csv(filename, dialog_id_list, text_list)

    df['dialog_id'] = dialog_id_list
    df['text'] = text_list
    df.to_csv('borderline_test.csv', index=False)


def draw():
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.3, hspace=0.2)
    i = 1

    for name, text in zip(df.dialog_id, df.text):
        tokens = word_tokenize(text)
        text_raw = " ".join(tokens)
        wordcloud = WordCloud(colormap='PuBu', background_color='white', contour_width=10).generate(text_raw)
        plt.subplot(4, 3, i, label=name, frame_on=True)
        plt.tick_params(labelsize=10)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(name, fontdict={'fontsize': 7, 'color': 'grey'}, y=0.93)
        plt.tick_params(labelsize=10)
        i += 1
    plt.show()


morph = pymorphy2.MorphAnalyzer()
stopwords_list = nltk.corpus.stopwords.words('russian')
stopwords_list.extend(['это', 'я', 'мы', 'ты', 'вы', 'то', 'ага', 'угу', 'да', 'нет'])
string.punctuation += '—'
result_dict = dict()
df = pd.DataFrame()
# create_csv_file()  #создание самого файла
df = pd.read_csv('borderline.csv', delimiter=',')
df_train = pd.read_csv('borderline_train.csv', delimiter=',')
# draw()


X = df['text']
y = df['dialog_id']


X_train = df_train['text']
y_train = df_train['title']



nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
               ])


nb.fit(X_train, y_train)


Pipeline(steps=[('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB())])

y_pred = nb.predict(X)





f = open('result.txt', 'w',encoding='utf-8')
for i in range(0, len(X)):
    s = str(i+1) + ": " + y_pred[i] + " ====>  " + X[i]
    f.write(s + '\n')
exit()
