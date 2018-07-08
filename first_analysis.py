from __future__ import division
from json import loads
import pandas as pd
from itertools import chain
import numpy as np
import spacy
import matplotlib.pyplot as plt
import preprocessing as pre
import seaborn as sns
sns.set_style('whitegrid')
#TODO: convert into scipy sparse matrix format with embedder
#TODO: convert oz into mg
#TODO: sort by exoticness

drinks = pre.load_data()

ingredients = pd.Series(drinks[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
ingredient_counts = ingredients.value_counts()
#ingredient_counts.iloc[ingredient_counts.values>1].iloc[::-1].plot.barh(figsize=(8,25))
#plt.savefig('common_ingredients.svg')

tokenized_drinks, token2word, word2token = pre.tokenize_data(drinks)
