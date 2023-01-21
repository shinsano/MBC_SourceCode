from flask import Response, request, jsonify, g, app, Flask, send_from_directory, redirect
import flask_mysqldb
import re
import json
import math
import requests
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import datetime
from noun_supersense_api import manager
import nltk
from nltk.corpus import wordnet as wn
import spacy
from random import randint, randrange
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import random
import string
import ast
from colorthief import ColorThief
from PIL import Image
from io import BytesIO
import base64
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
import pygsheets

gc = pygsheets.authorize(service_file='/home/ubuntu/mac_system/mac-system-moodboard-04690f19786d.json')
sh = gc.open('MB System Log')

nltk.data.path.append("/home/ubuntu/nltk_data/")
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'yuriom'
app.config['MYSQL_PASSWORD'] = 'Piedmont123$%^'
app.config['MYSQL_DB'] = 'api'
mysql = flask_mysqldb.MySQL(app)
PRE_PATH = "/home/ubuntu/mac_system"
SERVER = "http://3.217.222.46/"
sketch_engine_username = "shinsano";
sketch_engine_apikey = "8d430f63f2a74e07902f10fdd2eff35c";

wv = KeyedVectors.load_word2vec_format(PRE_PATH + '/numberbatch-en-19.08.txt.gz', binary=False)

def distance(w0, w1):
    c = mysql.connection.cursor()
    d0 = None
    if (c.execute("SELECT * FROM words WHERE word=%s", (w0,))) > 0:
        d0 = c.fetchone()
    d1 = None
    if (c.execute("SELECT * FROM words WHERE word=%s", (w1,))) > 0:
        d1 = c.fetchone()

    # if d0 == None or d1 == None:
    #     return None
    # a = 0.0
    # for i in range (0, 300):
    #     a = a + (d0[i + 1] * d1[i + 1])
    # b = 0.0
    # for i in range (0, 300):
    #     b = b + (d0[i + 1] * d0[i + 1])
    # c = 0.0
    # for i in range (0, 300):
    #     c = c + (d1[i + 1] * d1[i + 1])
    # return a / (math.sqrt(b) * math.sqrt(c))
    if d0 == None or d1 == None:
        return None
    a = []
    b = []
    for i in range (0, 300):
        a.append(d0[i + 1])
        b.append(d1[i + 1])

    c.close()

    return dot(a, b)/(norm(a)*norm(b))

@app.route('/api/distance', methods=['GET'])
def api_distance():
    if ('first' in request.args) and ('second' in request.args):
        first = request.args['first']
        second = request.args['second']
        return jsonify({
            'first': first,
            'second': second,
            'distance': distance(first, second)})
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/distance_matrix', methods=['POST'])
def api_distance_matrix():
    c = mysql.connection.cursor()
    words = request.json['words']
    words_vector = {}
    result = []
    for w in words:
        if (c.execute("SELECT * FROM words WHERE word=%s", (w,))) > 0:
            d = c.fetchone()
        if d is not None:
            vector = []
            for i in range (0, 300):
                vector.append(d[i + 1])
            words_vector[w] = vector
    for w1 in words:
        row = [w1]
        for w2 in words:
            if w1 == w2:
                row.append(1)
            else:
                row.append(dot(words_vector[w1], words_vector[w2])/(norm(words_vector[w1])*norm(words_vector[w2])))
        result.append(row)

    c.close()
    return jsonify(result)

@app.route('/api/get_noun_adj', methods=['GET'])
def get_noun_adj():
    if ('text' in request.args):
        noun_adj = {'noun' : [], 'adj' : []}
        text = request.args['text']
        tokens = nltk.word_tokenize(text)
        for word, pos in nltk.pos_tag(tokens):
            if (pos in ['JJ', 'JJR', 'JJS']):
                noun_adj['adj'].append(word)
            elif (pos in ['NN', 'NNP', 'NNS', 'NNPS']):
                noun_adj['noun'].append(word)
        return jsonify(noun_adj)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/save_log', methods=['POST'])
def save_log():
    log = request.form['log']
    filename = 'EA_' + str(randrange(100000, 1000000))
    with open('/home/ubuntu/mac_system/logs/' + filename + '.log', 'w') as f:
        f.write(log)
    return jsonify(filename)

def convert(word, from_pos, to_pos):

    # Just to make it a bit more readable
    WN_NOUN = 'n'
    WN_VERB = 'v'
    WN_ADJECTIVE = 'a'
    WN_ADJECTIVE_SATELLITE = 's'
    WN_ADVERB = 'r'

    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return None

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])

    if len(result) != 0:
        return result[0][0]
    else:
        return None

def ranking(candidates):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(cur_dir, 'SingleWord_U_score.csv')
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['word', 'U_score'])
    y = df['U_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        'silent': 1,
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.1,
        'tree_method': 'exact',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'predictor': 'cpu_predictor'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params['max_depth'] = 6
    params['eta'] =0.05
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=1000,
                      early_stopping_rounds=5,
                      evals=[(dtest, 'test')])
    group_prediction = []
    for group in candidates:
        group_df = pd.DataFrame(group['vector'], columns =X_train.columns)
        prediction = list(model.predict(xgb.DMatrix(group_df), ntree_limit=model.best_ntree_limit))
        prediction.insert(0, 5)
        tuple_group = list(zip(group['words'], prediction))
        tuple_group.sort(key=lambda tup: tup[1], reverse = True)
        # new_tg = []
        # for tg in tuple_group:
        #     new_tg.append([tg[0], str(tg[1])])
        # group_prediction.append(new_tg)
        group_prediction.append([tg[0] for tg in tuple_group])
    return group_prediction

def get_vector_matrix(word_matrix):
    cur = mysql.connection.cursor()
    new_matrix = []
    for group in word_matrix:
        format_strings = ','.join(['%s'] * (len(group) - 1))
        cur.execute("SELECT * FROM words WHERE word IN (%s)" % format_strings, tuple(group[1:]))
        vectors = cur.fetchall()
        tmp_group = {
            'words' : [group[0]],
            'vector' : []
        }
        for vector in vectors:
            tmp_group['words'].append(list(vector)[0])
            tmp_group['vector'].append(list(vector)[1:])
        new_matrix.append(tmp_group)

    cur.close()
    return new_matrix

@app.route('/api/get_all_related_words_as_adj', methods=['GET'])
def api_get_all_related_words_as_adj():
    words = request.args['word'].split(',')
    final_words = []
    for word in words:
        related_words = [word]
        results = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()
        for edge in results['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']
            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] != "Antonym") and (edge['rel']['label'] != "DistinctFrom"):
                if not "-" in related_word and not " " in related_word and not "_" in related_word:
                    if check_pos(related_word, 'ADJ'):
                        related_words.append(related_word)
                    else:
                        c_res = convert(related_word, 'n', 'a')
                        if c_res is not None:
                            related_words.append(c_res)
        final_words.append(list(dict.fromkeys(related_words)))
    vector_matrix = get_vector_matrix(final_words)
    candidates_word1 = ranking(vector_matrix)
    return jsonify(candidates_word1)

def filter_xboost(words):
    cur = mysql.connection.cursor()

    format_strings = ','.join(['%s'] * len(words))
    cur.execute("SELECT * FROM words WHERE word IN (%s)" % format_strings, tuple(words))
    vectors = cur.fetchall()
    new_matrix = {
        'words' : [],
        'vector' : []
    }
    for vector in vectors:
        new_matrix['words'].append(list(vector)[0])
        new_matrix['vector'].append(list(vector)[1:])

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(cur_dir, 'SingleWord_U_score.csv')
    df = pd.read_csv(csv_file)
    X = df.drop(columns=['word', 'U_score'])
    y = df['U_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    params = {
        'silent': 1,
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': 0.1,
        'tree_method': 'exact',
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'predictor': 'cpu_predictor'
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params['max_depth'] = 6
    params['eta'] =0.05
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=1000,
                      early_stopping_rounds=5,
                      evals=[(dtest, 'test')])
    group_prediction = []

    group_df = pd.DataFrame(new_matrix['vector'], columns =X_train.columns)
    prediction = list(model.predict(xgb.DMatrix(group_df), ntree_limit=model.best_ntree_limit))
    tuple_group = list(zip(new_matrix['words'], prediction))
    group_prediction = list(filter(lambda tg: tg[1] >= 1.7, tuple_group))

    cur.close()
    return [tg[0] for tg in group_prediction]

@app.route('/api/get_all_combinations_w1_w2', methods=['GET'])
def api_get_all_combinations_w1_w2():
    score_list = [
        [0.15, 0.2, 5.17],
        [-0.05, 0, 4.85],
        [0.05, 0.1, 4.77],
        [0.1, 0.15, 4.75],
        [0.4, 0.45, 4.73],
        [0.2, 0.25, 4.63],
        [0.65, 10, 4.57],
        [-0.1, -0.05, 4.55],
        [0.3, 0.35, 4.55],
        [0, 0.05, 4.54],
        [-10, -0.1, 4.43],
        [0.25, 0.3, 4.30],
        [0.5, 0.55, 4.26],
        [0.35, 0.4, 4.23],
        [0.55, 0.6, 4.06],
        [0.45, 0.5, 3.68],
        [0.6, 0.65, 3.48]
    ]
    words1 = request.args['word'].split(',')
    low = float(request.args['low'])
    high = float(request.args['high'])
    combinations = {}
    related_nouns = []

    for word in words1:
        results = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()
        for edge in results['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']
            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] != "Antonym") and (edge['rel']['label'] != "DistinctFrom"):
                if check_pos(related_word, 'NOUN'):
                    related_nouns.append(related_word)
                else:
                    c_res = convert(related_word, 'a', 'n')
                    if c_res is not None:
                        related_nouns.append(c_res)
    related_nouns = list(dict.fromkeys(related_nouns))

    related_nouns = filter_xboost(related_nouns)

    for word in words1:
        combinations[word] = []
        tmp = []
        for noun in related_nouns:
            dis = distance(word, noun)
            if dis is not None and low <= dis and dis < high:
                for score_item in score_list:
                    if score_item[0] <= dis and dis <= score_item[1]:
                        tmp.append([word + '-' + noun, score_item[2], dis])
                        break
        tmp.sort(key=lambda y: (-y[1], y[2]))
        combinations[word] = [t[0] for t in tmp]
    return jsonify(combinations)

@app.route('/api/get_antonyms_w1_w2', methods=['GET'])
def api_get_antonyms_w1_w2():

    word1 = request.args['word1']
    word2 = request.args['word2']
    candi_word1 = []
    candi_word2 = []

    sym_results = requests.get("https://api.conceptnet.io/c/en/" + word1 + "?filter=/c/en&limit=500").json()
    for sym_edge in sym_results['edges']:
        sym_node_related_word = sym_edge['end'] if sym_edge['start']['label'] == word1 else sym_edge['start']
        sym_related_word = sym_node_related_word['label']
        if (word1 != sym_related_word) and ('language' in sym_node_related_word) and (sym_node_related_word['language'] == 'en') and (sym_edge['rel']['label'] == "RelatedTo" or sym_edge['rel']['label'] == "SimilarTo"):
            results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + sym_related_word + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
            for edge in results['edges']:
                node_related_word = edge['end'] if edge['start']['label'] == sym_related_word else edge['start']
                related_word = node_related_word['label']
                if (sym_related_word != related_word) and (word1 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
                    if check_pos(related_word, 'ADJ'):
                        candi_word1.append(related_word)
                    else:
                        c_res = convert(related_word, 'n', 'a')
                        if (c_res is not None) and (c_res != word1):
                            candi_word1.append(c_res)
    candi_word1 = list(dict.fromkeys(candi_word1))

    sym_results = requests.get("https://api.conceptnet.io/c/en/" + word2 + "?filter=/c/en&limit=500").json()
    for sym_edge in sym_results['edges']:
        sym_node_related_word = sym_edge['end'] if sym_edge['start']['label'] == word2 else sym_edge['start']
        sym_related_word = sym_node_related_word['label']
        if (word2 != sym_related_word) and ('language' in sym_node_related_word) and (sym_node_related_word['language'] == 'en') and (sym_edge['rel']['label'] == "RelatedTo" or sym_edge['rel']['label'] == "SimilarTo"):
            results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + sym_related_word + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
            for edge in results['edges']:
                node_related_word = edge['end'] if edge['start']['label'] == sym_related_word else edge['start']
                related_word = node_related_word['label']
                if (sym_related_word != related_word) and (word2 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
                    if check_pos(related_word, 'NOUN'):
                        candi_word2.append(related_word)
                    else:
                        c_res = convert(related_word, 'a', 'n')
                        if (c_res is not None) and (c_res != word2):
                            candi_word2.append(c_res)
    candi_word2 = list(dict.fromkeys(candi_word2))


    # results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + word1 + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
    # for edge in results['edges']:
    #     node_related_word = edge['end'] if edge['start']['label'] == word1 else edge['start']
    #     related_word = node_related_word['label']
    #     if (word1 != related_word) and (word1 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
    #         if check_pos(related_word, 'ADJ'):
    #             candi_word1.append(related_word)
    #         else:
    #             c_res = convert(related_word, 'n', 'a')
    #             if (c_res is not None) and (c_res != word1):
    #                 candi_word1.append(c_res)
    # candi_word1 = list(dict.fromkeys(candi_word1))



    # results = requests.get("https://api.conceptnet.io/query?start=/c/en/" + word2 + "&filter=/c/en&rel=/r/Antonym&limit=500").json()
    # for edge in results['edges']:
    #     node_related_word = edge['end'] if edge['start']['label'] == word2 else edge['start']
    #     related_word = node_related_word['label']
    #     if (word2 != related_word) and (word2 != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == "Antonym" or edge['rel']['label'] == "DistinctFrom"):
    #         if check_pos(related_word, 'NOUN'):
    #             candi_word2.append(related_word)
    #         else:
    #             c_res = convert(related_word, 'a', 'n')
    #             if (c_res is not None) and (c_res != word2):
    #                 candi_word2.append(c_res)
    # candi_word2 = list(dict.fromkeys(candi_word2))


    return jsonify({
            word1: candi_word1,
            word2: candi_word2
        })

@app.route('/api/noun_adj_convertor', methods=['GET'])
def api_noun_adj_convertor():
    if ('word' in request.args and 'from' in request.args and 'to' in request.args):
        words = request.args['word'].split(',')
        from_pos = request.args['from']
        to_pos = request.args['to']

        final_result = []

        for word in words:
            c_res = convert(word, from_pos, to_pos)
            if c_res is not None:
                final_result.append(c_res)

        return jsonify(final_result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/vector', methods=['GET'])
def api_vector():
    if ('word' in request.args):
        word = request.args['word']
        c = mysql.connection.cursor()
        d0 = 0
        if (c.execute("SELECT * FROM words WHERE word=%s", (word,))) > 0:
            d0 = c.fetchone()
        c.close()
        return jsonify(d0)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/supersense_noun', methods=['GET'])
def api_supersense_noun():
    if ('word' in request.args):
        word = request.args['word']
        ss_obj = manager.SuperSense()
        classes = []
        result = {
            'Tops' : 0,
            'act' : 0,
            'animal' : 0,
            'artifact' : 0,
            'attribute' : 0,
            'body' : 0,
            'cognition' : 0,
            'communication' : 0,
            'event' : 0,
            'feeling' : 0,
            'food' : 0,
            'group' : 0,
            'location' : 0,
            'motive' : 0,
            'object' : 0,
            'person' : 0,
            'phenomenon' : 0,
            'plant' : 0,
            'possession' : 0,
            'process' : 0,
            'quantity' : 0,
            'relation' : 0,
            'shape' : 0,
            'state' : 0,
            'substance' : 0,
            'time' : 0
        }
        SuperSense_total = ss_obj.sense_info_list_for_lemma_pos[(word, 'n')]
        for sense_info in SuperSense_total:
            these_classes = ss_obj.get_classes_for_synset_pos(sense_info.synset, 'n')
            if these_classes is not None:
                classes.extend(list(map(lambda x: x.replace('noun.', ''), these_classes)))
        if len(classes) != 0:
            if not isinstance(classes[0],list):
                sorted_classes = list(set(classes))
        for c_item in sorted_classes:
            result[c_item] = classes.count(c_item) / len(SuperSense_total)
        return jsonify(result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/adjective-frequency', methods=['GET'])
def api_frequency():
    if ('word' in request.args):
        word = request.args['word'] + "-j"

        c = mysql.connection.cursor()
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word,)) > 0:
            d = c.fetchone()
        c.close()
        return jsonify({
            'word': request.args['word'],
            'frequency': d[1] if d != None else None,
            'relative-frequency': d[2] if d != None else None})
    else:
        return "Error: Please specify appropriate fields."

def check_pos(word, pos):
    doc = nlp(word)
    return doc[0].pos_ == pos

@app.route('/api/list-related-adjectives', methods=['GET'])
def api_list_related_adjectives():
    if ('word' in request.args):
        word = request.args['word']
        relation = None
        if ("relation" in request.args):
            relation = request.args['relation']
        limit = 500
        if ("limit" in request.args):
            limit = int(request.args['limit'])
        results = []
        related_words = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en" + "&limit=" + str(limit) + ("" if relation == None else ("&filter=/r/" + relation))).json()

        for edge in related_words['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']

            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (relation == None or relation == edge['rel']['label']):
                c = mysql.connection.cursor()
                d = None
                if c.execute("SELECT * FROM frequencies WHERE word=%s", (related_word + '-j',)) > 0:
                    d = c.fetchone()
                    if d[1] != None and d[2] != None:
                        results.append({
                            'word': related_word,
                            'frequency': d[1] if d != None else None,
                            'relative-frequency': d[2] if d != None else None,
                            'distance': distance(word, related_word)})
                c.close()

        return jsonify(results)
    else:
        return "Error: Please specify appropriate fields."


@app.route('/api/list-antonyms-of-adjective', methods=['GET'])
def api_list_antonyms_of_adjective():
    if ('word' in request.args):
        word = request.args['word']
        limit = 10
        if ("limit" in request.args):
            limit = int(request.args['limit'])
        done = []
        results = []
        start = datetime.datetime.now()

        related_words = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500").json()

        for edge in related_words['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']

            if ((datetime.datetime.now() - start).seconds < 4) and len(done) < limit and (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and ("Antonym" != edge['rel']['label']):
                antonyms = requests.get("https://api.conceptnet.io/c/en/" + related_word + "?filter=/c/en&filter=/r/Antonym&limit=2000").json()

                for edge in antonyms['edges']:
                    node = edge['end'] if edge['start']['label'] == word else edge['start']
                    antonym = node['label']

                    if ((datetime.datetime.now() - start).seconds < 4) and len(done) < limit and (not (antonym in done)) and (antonym != related_word) and ('language' in node) and (node['language'] == 'en') and ("Antonym" == edge['rel']['label']):
                        c = mysql.connection.cursor()
                        d = None
                        if c.execute("SELECT * FROM frequencies WHERE word=%s", (antonym + '-j',)) > 0:
                            d = c.fetchone()
                            if d[1] != None and d[2] != None:
                                done.append(antonym)
                                results.append({
                                    'word': antonym,
                                    'frequency': d[1] if d != None else None,
                                    'relative-frequency': d[2] if d != None else None,
                                    'distance': distance(word, antonym)})
                        c.close()

        return jsonify(results)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/relative-frequency_distance', methods=['GET'])
def api_relative_frequency_distance():
    if ('word1' in request.args):
        word1 = request.args['word1']
    if ('word2' in request.args):
        word2 = request.args['word2']

    if word1 != None and word2 != None:
        result = {
            'distance': distance(word1, word2)
        }
        c = mysql.connection.cursor()
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word1 + '-j',)) > 0:
            d = c.fetchone()
            if d[1] != None and d[2] != None:
                result['word1'] = d[2] if d != None else None
        d = None
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word2 + '-n',)) > 0:
            d = c.fetchone()
            if d[1] != None and d[2] != None:
                result['word2'] = d[2] if d != None else None
        c.close()
        return jsonify(result)
    else:
        return "Error: Please specify appropriate fields."

@app.route('/api/is-adjective', methods=['GET'])
def api_is_adjective():
    if ('word' in request.args):
        word = request.args['word']
        c = mysql.connection.cursor()
        result = False
        if c.execute("SELECT * FROM frequencies WHERE word=%s", (word + '-j',)) > 0:
            result = True
        c.close()
        return jsonify({'word': word, 'result': result})
    else:
        return "Error: Please specify appropriate fields."

# moodboard Start
@app.route('/api/search', methods=['POST'])
def api_search():
    keywords = request.json
    combine_words = [[], []]
    result = [[], []]

    url = 'https://duckduckgo.com/'
    headers = {
        'dnt': '1',
        'crossDomain': 'true',
        #'accept-encoding': 'gzip, deflate, sdch, br',
        'x-requested-with': 'XMLHttpRequest',
        'accept-language': 'en-GB,en-US;q=0.8,en;q=0.6,ms;q=0.4',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'accept': 'application/json', # text/javascript, */*; q=0.01',
        'referer': 'https://duckduckgo.com/',
        'authority': 'duckduckgo.com',
        "Access-Control-Allow-Credentials": "true"
    }
    if len(keywords) == 4:
        for index, keyword_group in enumerate([keywords[:2], keywords[2:]]):
            for keyword in keyword_group:
                response = requests.get('https://app.sketchengine.eu/bonito/run.cgi/wsketch', auth=(sketch_engine_username, sketch_engine_apikey), params={
                    'lemma': keyword.lower(),
                    'lpos': '-j',
                    'corpname': 'preloaded/ententen20_tt31',
                    'format': 'json',
                    'maxitems': 2
                }).json()
                data = response['Gramrels']
                for item in data:
                    if item['name'] == 'nouns modified by \"%w\"':
                        for word in item['Words']:
                            combine_words[index].append(keyword + ' ' +word['word'])
                        break

        for index, combine_group in enumerate(combine_words):
            for keyword_query in combine_group:
                params = {
                    'q': keyword_query
                };

                res = requests.post(url, data=params)
                searchObj = re.search(r"vqd=\'([\d-]+)\'", res.text, re.M|re.I)

                params = (
                    ('l', 'wt-wt'),
                    ('o', 'json'),
                    ('kp', '1'),
                    ('q', keyword_query),
                    ('vqd', searchObj.group(1)),
                    ('iax', 'images'),
                    ('ia', 'images'),
                    ('iaf','type:photo-photo,size:imagesize-medium'),
                    ('f', ',,,'),
                    ('p', '2')
                )

                requestUrl = url + "i.js";
                res = requests.get(requestUrl, headers=headers, params=params);

                data = json.loads(res.text);
                search_results = getJson(data["results"]);

                result[index] += search_results[:5]
    else:
        for index, keyword in enumerate(keywords):
            params = {
                'q': keyword
            };

            res = requests.post(url, data=params)
            searchObj = re.search(r"vqd=\'([\d-]+)\'", res.text, re.M|re.I)

            params = (
                ('l', 'wt-wt'),
                ('o', 'json'),
                ('kp', '1'),
                ('q', keyword),
                ('vqd', searchObj.group(1)),
                ('iax', 'images'),
                ('ia', 'images'),
                ('iaf','type:photo-photo,size:imagesize-medium'),
                ('f', ',,,'),
                ('p', '2')
            )

            requestUrl = url + "i.js";
            res = requests.get(requestUrl, headers=headers, params=params);

            data = json.loads(res.text);
            search_results = getJson(data["results"]);

            result[index] = search_results[:5]

    return jsonify(result[0] + result[1])

# MB Start
@app.route('/api/search_new', methods=['GET'])
def search_new():
    word = request.args['word']
    
    url1 = 'https://www.behance.net/search/images?search=' + word + '&field=industrial%20design';
    url2 = 'https://www.behance.net/search/images?search=' + word + '&field=architecture';
    url3 = 'https://www.behance.net/search/images?search=' + word + '&field=fashion';

    response = requests.get(url1)
    soup = BeautifulSoup(response.text, "html.parser")
    items1 = soup.find_all("img", itemprop="thumbnail")

    response = requests.get(url2)
    soup = BeautifulSoup(response.text, "html.parser")
    items2 = soup.find_all("img", itemprop="thumbnail")

    response = requests.get(url3)
    soup = BeautifulSoup(response.text, "html.parser")
    items3 = soup.find_all("img", itemprop="thumbnail")

    return jsonify([[ item1['src'] for item1 in items1], [ item2['src'] for item2 in items2], [ item3['src'] for item3 in items3]])

@app.route('/api/save_nwc_log', methods=['POST'])
def save_nwc_log():
    data = request.json
    log = data['log']
    save_log_sheet(log, 2)
    return jsonify({'status': 'ok'})


@app.route('/api/next_search_new2', methods=['POST'])
def next_search_new2():
    data = request.json
    labels = data['labels']
    word1 = data['word1']
    word2 = data['word2']
    log = data['log']
    feedbacks = data['feedbacks']
    sheet = data['sheet']
    for image in labels:
        words = list(filter(lambda item: '-' not in item, image['image_labels'].keys()))
        while len(words) > 0:
            try:
                X = wv[words]
                break
            except KeyError as e:
                m = re.search("'([^']*)'", e.args[0])
                key = m.group(1)
                words.remove(key)
        for index, _ in enumerate(X):
            X[index] =  X[index] * image['image_labels'][words[index]]
            dis = wv.distances(words[index], (word1, word2))
            if dis[0] < dis[1]:
                X[index] = X[index] * image['image_weight'][0]
            else:
                X[index] = X[index] * image['image_weight'][1]
        image['mean_vector'] = np.mean(X, axis=0)
        dis_iv = wv.distances(image['mean_vector'], (word1, word2))
        log[int(image['label_index']) + 6] = str(dis_iv[0])
        log[int(image['label_index']) + 15] = str(dis_iv[1])
    final_vector = np.mean([ image['mean_vector'] for image in labels ], axis=0)
    final_dis = wv.distances(final_vector, (word1, word2))
    log[24] = str(final_dis[0])
    log[25] = str(final_dis[1])
    nearest_words = []
    if len(feedbacks) == 0:
        nearest_words = wv.most_similar([ image['mean_vector'] for image in labels ], topn=20)
    else:
        nearest_words = wv.most_similar(positive = [ image['mean_vector'] for image in labels ], negative = feedbacks, topn=20)
    log[26] = '\n'.join(list(map(lambda x: x[0], nearest_words)))
    save_log_sheet(log, sheet)
    return jsonify(nearest_words)


def save_log_sheet(data, sheet):
    wks = sh[int(sheet)]
    wks.append_table(data)


def getJson(objs):
    imageList = []
    for obj in objs[:40]:
        if obj["image"].endswith(".png") or obj["image"].endswith(".jpg") or obj["image"].endswith(".jpeg"):
            imageList.append(obj["image"])
    return imageList


@app.route('/api/load', methods=['POST'])
def api_load():
    data = request.json
    dropped_image = str(data['image'])
    search_term = data['search_term']

    file_name, labels, palette = load_images_search(search_term, dropped_image)

    pyDictName = {'imgSrc': file_name, 'labels': labels, 'colorPalette': palette}
    return jsonify(pyDictName)


@app.route('/api/get_next_word', methods=['POST'])
def api_get_next_word():
    data = request.json
    result = ['','']
    word = data['word']
    word_anti = data['word_anti']

    c = mysql.connection.cursor()

    if len(word) == 0:
        result[0] = word[0]
    else:
        freq = 999999999
        for w in word:
            adj_word = w + "-j"
            d = None
            if c.execute("SELECT * FROM frequencies WHERE word=%s", (adj_word,)) > 0:
                d = c.fetchone()
                if d != None and freq > d[1]:
                    result[0] = w
                    freq = d[1]

    if len(word_anti) == 0:
        result[1] = word_anti[0]
    else:
        freq = 999999999
        for w in word_anti:
            adj_word = w + "-j"
            d = None
            if c.execute("SELECT * FROM frequencies WHERE word=%s", (adj_word,)) > 0:
                d = c.fetchone()
                if d != None and freq > d[1]:
                    result[1] = w
                    freq = d[1]
    c.close()

    pyDictName = {'result': result}
    return jsonify(pyDictName)


@app.route('/api/get_labels', methods=['POST'])
def api_get_labels():
    data = request.json
    vision_labels = data['vision_labels']
    result = {}
    c = mysql.connection.cursor()

    for word_str in vision_labels:
        word = word_str.lower()
        results = []
        related_words = requests.get("https://api.conceptnet.io/c/en/" + word + "?filter=/c/en&limit=500&filter=/r/RelatedTo&filter=/r/HasProperty").json()
        for edge in related_words['edges']:
            node_related_word = edge['end'] if edge['start']['label'] == word else edge['start']
            related_word = node_related_word['label']
            if (word != related_word) and ('language' in node_related_word) and (node_related_word['language'] == 'en') and (edge['rel']['label'] == 'RelatedTo'):
                if c.execute("SELECT * FROM frequencies WHERE word=%s", (related_word + '-j',)) > 0:
                    results.append({
                        'word': related_word,
                        'distance': distance(word, related_word)})        
        results.sort(key=lambda w:-w['distance'])
        for item in results[:5]:
            result[item['word']] = vision_labels[word_str]
    
    c.close()
    return jsonify(result)


@app.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    data_info = request.files['image'].read()

    filename = randomString('x.png');
    fullPath = "static/moodboard/images/" + filename

    outfile = open(PRE_PATH + '/' + fullPath, 'wb')
    outfile.write(data_info)
    outfile.close()

    palette = get_data(filename, ['upload'], 'upload')
    labels = 0;

    pyDictName = {'imgSrc': fullPath, 'labels': labels, 'colorPalette': palette}
    return jsonify(pyDictName)


@app.route('/api/get_image_position', methods=['POST'])
def api_get_image_position():
    data = request.json
    image_labels = data['labels']
    image_axis = data['axis']
    image_id = data['id']
    result = {'id': image_id}

    for axis in image_axis:
        total = 0
        count = 0
        for key in image_labels:
            dist = distance(axis, key)
            if dist is not None and dist > 0:
                total += dist * image_labels[key]
                count += 1
        if count == 0:
            result[axis] = 0
        else:
            result[axis] = total / count

    return jsonify(result)


@app.route('/api/load_labels', methods=['POST'])
def api_load_labels():
    data = request.json
    image_labels = data['labels']
    name = str(data['path'])
    name = name.replace(SERVER + 'static/moodboard/images/', '');

    labels = insert_lables(name, image_labels)

    pyDictName = {'labels': labels}
    return jsonify(pyDictName)


@app.route('/api/load_crop', methods=['POST'])
def api_load_crop():
    data = request.json
    image_raw = data['image']
    # data:image/png;base64,
    datatype = image_raw[image_raw.find('/') + 1:image_raw.find(';')]
    image_clear = image_raw[image_raw.find(','):]

    cropped_image = image_clear
    search_term = data['search_term']
    filename = randomString('x.' + datatype);

    fullPath = "static/moodboard/images/" + filename
    with open(PRE_PATH + '/' + fullPath, "wb") as fh:
        fh.write(base64.b64decode(cropped_image))

    palette = get_data(filename, search_term, 'cropped')
    labels = 0;

    pyDictName = {'imgSrc': fullPath, 'labels': labels, 'colorPalette': palette}
    return jsonify(pyDictName)


def load_images_search(word, url):
    # get images and filter text out
    if not url.startswith('http'):
        url = "http://" + url

    print(url)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'})
    im = Image.open(BytesIO(response.content))

    if url.endswith('png'):
        im = im.convert('RGBA')
    else:
        im = im.convert('RGB')

    filename = randomString(url);
    fullPath = "static/moodboard/images/" + filename

    # load data to db
    db_results = already_there(url)

    if db_results:
        db_results = list(db_results)
        db_results[1] = 0 if db_results[1] == None else eval(db_results[1])
        db_results[2] = 0 if db_results[2] == None else eval(db_results[2])
        return "static/moodboard/images/" + db_results[0], db_results[1], db_results[2]
    else:
        im.save(PRE_PATH + '/' + fullPath)
        palette = get_data(filename, word, str(url))
        return fullPath, 0, palette

def randomString(url, stringLength=20):
    """Generate a random string of fixed length """
    Letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    url_split = url.split(".")
    format_ = url_split[-1]
    s = ''.join(random.choice(Letters) for i in range(stringLength)) + '.' + format_

    if alreadyNameInDB(s):
        randomString(url)
    else:
        return s

def alreadyNameInDB(name):
    c = mysql.connection.cursor()
    if c.execute("SELECT name FROM images WHERE name=%s", (name,)) > 0:
        c.close()
        return True
    else:
        c.close()
        return False

def already_there(img_url):
    c = mysql.connection.cursor()
    c.execute("SELECT name,labels,palette FROM images WHERE url = %s;", (img_url,))
    img_data = c.fetchone()
    c.close()
    return img_data

def already_there_values(name):
    cur = mysql.connection.cursor()
    cur.execute("SELECT name,labels,palette FROM images WHERE name = %s;", (name,))
    img_data = cur.fetchone()
    cur.close()
    return img_data

def get_data(file, word, url):
    path = "/static/moodboard/images"

    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        width, height, palette = get_image_features(file, path, -1)
        inserted = insert_element(file, path, width, height, palette, word, url)
        return palette
    else:
        return 0

def get_image_features(el_name, path, transparency, remove_criterion=10):

    #color palette
    img = Image.open(PRE_PATH + path + '/' + el_name)
    color_thief = ColorThief(PRE_PATH + path + '/' + el_name)
    palette = color_thief.get_palette(color_count=10)

    if remove_criterion is not None:
        palette = remove_similar_colors(palette, remove_criterion)

    # dimension
    width,height = img.size

    return width, height, palette

def remove_similar_colors(palette, remove_criterion, verbose=False):

    remove = []

    for i, color in enumerate(palette[:-1]):
        if verbose:
            print (i, color)
        for j, c2 in enumerate(palette[i+1:]):
            if verbose:
                print ('\tcompared to', j, c2)
            if (abs(color[0] - c2[0]) < remove_criterion) and (abs(color[1] - c2[1]) < remove_criterion) and (abs(color[2] - c2[2]) < remove_criterion):
                remove.append(j+i)
                if verbose:
                    print ('\t\twill be deleted')

    remove = sorted(list(set(remove)))
    for i in reversed(remove):
        del palette[i]

    return palette

def insert_element(el_name, path, width, height, palette, word, url):

    cur = mysql.connection.cursor()

    cur.execute('''
        INSERT INTO images (name, location, width, height, palette, query_word, url ) 
        VALUES 
        (%s, %s, 
        %s, %s, 
        %s, %s, 
        %s
        )
        ''', (
        el_name, path, width, height, json.dumps(palette), json.dumps(word), url))
    mysql.connection.commit()
    cur.close()
    return "Done"

def insert_lables(img_name, labels):
    cur = mysql.connection.cursor()

    labels_to_insert = labels
    cur.execute('''
        Update images set labels = %s where name = %s ''', (json.dumps(labels_to_insert), img_name))
    mysql.connection.commit()
    cur.close()
    return "Done"

#moodboard End

@app.route('/')
def index():
    return redirect('index.html')

@app.route('/<path:path>', methods=['GET'])
def send_static(path):
    return send_from_directory('static', path)

@app.teardown_appcontext
def close_connection(exception):
   db = getattr(g, '_database', None)
   if db is not None:
      db.close()

# app.run(host='0.0.0.0', port=80)
