import pandas as pd
import numpy as np
import re
from os.path import join, dirname
from sklearn.preprocessing import MultiLabelBinarizer


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
    if preprocess:
        data[['strIngredient{}'.format(i) for i in range(1, 16)]] = data[['strIngredient{}'.format(i) for i in range(1, 16)]].applymap(preprocess_ingredient)
        data.iloc[:, data.columns.str.contains('Measure')] = data.iloc[:, data.columns.str.contains('Measure')].applymap(preprocess_measure)
    return data


def tokenize_data(data):
    mlb = MultiLabelBinarizer()
    ingredients_series = data[['strIngredient{}'.format(i) for i in range(1,16)]].apply(lambda x : x.dropna().unique().tolist(), axis=1)
    transformed_ingredients = mlb.fit_transform(ingredients_series)
    return transformed_ingredients, mlb


def from_drinks_to_measure(drinks_df, ingredients_vector, mlb):
    '''Expects the original (pre-processed) DataFrame drinks_df,
    the ingredients_vector and MultiLabelBinarizer given by tokenize_data'''
    measures = drinks_df[['strMeasure{}'.format(i) for i in range(1, 16)]]
    ingredient_in_df = drinks_df[['strIngredient{}'.format(i) for i in range(1, 16)]]
    list_of_transl_dicts = [dict(zip(ingredient_in_df.values[i], measures.values[i])) for i in range(drinks_df.shape[0])]
    measure_vector = np.full(ingredients_vector.shape, '', dtype='object')
    for i in range(measure_vector.shape[0]):
        for j in np.where(ingredients_vector[i])[0]:
            measure_vector[i, j] = list_of_transl_dicts[i][mlb.classes_[j]]
    return measure_vector
