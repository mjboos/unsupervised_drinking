from __future__ import division
from json import loads
import pandas as pd
from itertools import chain
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

#TODO: simple word preprocessing
def preprocess_ingredient(text):
    import unicodedata as ud
    if pd.isna(text):
        return text
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = text.lower()
    text = re.sub(r'[\n\r]', r' ', text)
    if text.isspace():
        text = np.nan
    return text

def preprocess_measure(text):
    import unicodedata as ud
    if pd.isna(text):
        return text
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = text.lower()
    text = re.sub(r'[\n\r]', r' ', text)
    if text.isspace():
        text = np.nan
    return text

def load_data(preprocess=True):
    data = pd.read_csv('all_drinks.csv', encoding='utf-8')
#    data.fillna('NAN', inplace=True)
    if preprocess:
        data[['strIngredient{}'.format(i) for i in range(1,16)]] = data[['strIngredient{}'.format(i) for i in range(1,16)]].applymap(preprocess_ingredient)
        data.iloc[:,data.columns.str.contains('Measure')] = data.iloc[:,data.columns.str.contains('Measure')].applymap(preprocess_measure)
    return data

def tokenize_data(data):
    mlb = MultiLabelBinarizer()
    ingredients_series = data[['strIngredient{}'.format(i) for i in range(1,16)]].apply(lambda x : x.dropna().unique().tolist(), axis=1)
    transformed_ingredients = mlb.fit_transform(ingredients_series)
    return transformed_ingredients, mlb

def make_tokenizer_dict(data):
    ingredients = pd.Series(data[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
    ingredient_counts = ingredients.value_counts()
    token_to_word = {0:np.NaN}
    word_to_token = {0:0}
    for i, (ingredient, n) in enumerate(ingredient_counts.items()):
        token_to_word[i+1] = ingredient
        word_to_token[ingredient] = i+1
    return token_to_word, word_to_token

def tokenize_by_hand_data(data):
    token2word, word2token = make_tokenizer_dict(data)
    data[['strIngredient{}'.format(i) for i in range(1,16)]] = data[['strIngredient{}'.format(i) for i in range(1,16)]].fillna(0).applymap(lambda x : word2token[x])
    return data, token2word, word2token

list_of_alcohol = ['vodka', 'rum', 'whiskey', 'tequila', 'sambuca', 'curacao']
