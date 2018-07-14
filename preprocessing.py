from __future__ import division
from json import loads
import pandas as pd
from itertools import chain
import numpy as np
import re
from os.path import join, dirname
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
    data = pd.read_csv(join(dirname(__file__), 'all_drinks.csv'), encoding='utf-8')
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

def from_drinks_to_measure(drinks_df, ingredients_vector, mlb):
    '''Expects the original (pre-processed) DataFrame drinks_df,
    the ingredients_vector and MultiLabelBinarizer given by tokenize_data'''
    measures = drinks_df[['strMeasure{}'.format(i) for i in range(1,16)]]
    ingredient_in_df = drinks_df[['strIngredient{}'.format(i) for i in range(1,16)]]
    list_of_transl_dicts = [dict(zip(ingredient_in_df.values[i], measures.values[i])) for i in range(drinks_df.shape[0])]
    print(list_of_transl_dicts[0])
    measure_vector = np.full(ingredients_vector.shape, '', dtype='object')
    for i in range(measure_vector.shape[0]):
        for j in np.where(ingredients_vector[i])[0]:
            measure_vector[i,j] = list_of_transl_dicts[i][mlb.classes_[j]]
    return measure_vector

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

def contains_juice(ingredient_vector):
    pass

def contains_cream(ingredient_vector):
    pass

def alcohol_content(ingredient_vector):
    pass

list_of_juices = ['banana', 'grenadine', 'lemon', 'lemon juice', 'lemon-lime soda', 'lemonade', 'lime', 'lime juice', 'pineapple juice']

list_of_alcohol = ['vodka', 'amaretto', "bailey's irish cream", 'bitters', 'blended whiskey', 'blue curacao', 'bourbon', 'brandy', 'dark rum', 'dry vermouth', 'gin', 'kahlua']

#exclude_by_hand = ['apricot brandy', 'absolut citron', 'benedictine', 'angostura bitters', 'cherry brandy', 'galliano', 'grand marnier', 'maraschino liqueur', 'orange bitters', 'peach schnapps', 'sloe gin', 'sweet and sour', 'wild turkey', 'white creme de menthe', 'bitters'] 

exclude_by_hand = [u'7-up', u'absolut citron', u'absolut kurant', u'absolut peppar', u'advocaat', u'apricot brandy',
 u'allspice', u'angelica root', u'angostura bitters', u'an\u0303ejo rum', 'aquavit', 'almond flavoring',
 u'apple brandy', u'applejack', u'asafoetida', u'bacardi limon',
 u'benedictine', u'bitters', u'black sambuca', u'blackberry brandy',
 u'blackcurrant cordial', u'blackcurrant squash', u'blueberry schnapps',
 u'butterscotch schnapps', u'cantaloupe', u'caramel coloring',
 u'chambord raspberry liqueur', u'cherry grenadine', u'cherry heering',
 u'cherry liqueur', u'coconut liqueur', u'creme de cassis', u'creme de mure',
 u'crown royal', u'dark creme de cacao', u'demerara sugar', u'drambuie',
 u'dubonnet rouge', u'erin cream', u'everclear', u'firewater',
 u'food coloring', u'frangelico', u'fresca', u'galliano', u'glycerine',
 u'godiva liqueur', u'grand marnier', u'half-and-half', u'hot damn', u'jello',
 u'kool-aid', u'licorice root', u'lillet blanc', u'melon liqueur',
 u'midori melon liqueur', u'mini-snickers bars', u'mountain dew',
 u'oreo cookie', u'orgeat syrup', u'papaya', u'peach bitters', u'peach brandy',
 u'peach vodka', u'peychaud bitters', u'pisang ambon', u'pisco', u'ricard',
 u'rumple minze', u'sarsaparilla', u'schweppes russchian', u'sherbet',
 u'sirup of roses', u'sloe gin', u'st. germain', u'surge', u'tia maria',
 u'vanilla vodka', u'wormwood', u'yukon jack', u'zima']
