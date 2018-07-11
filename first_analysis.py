from __future__ import division
from json import loads
import pandas as pd
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pre
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

sns.set_style('whitegrid')

def mask_cocktails(included_ingredients, ingredients_vector):
    cocktails_to_consider = np.logical_not(ingredients_vector[:,included_ingredients==0].any(axis=1))
#    masked_recipes = np.ma.masked_array(ingredients_vector, np.tile(np.logical_not(included_ingredients)[None],(ingredients_vector.shape[0],1)))
    masked_recipes = np.ma.masked_array(ingredients_vector, np.logical_or(np.logical_not(cocktails_to_consider)[:,None], np.tile(np.logical_not(included_ingredients)[None], (ingredients_vector.shape[0],1))))
    return masked_recipes


def stepwise_exclusion(included_ingredients, ingredients_vector):
    '''Finds the ingredient to exclude that leaves as much drink options as possible
    INPUT
    included_ingredients    -       binary vector of length n, indicating which ingredients are included
    ingredients_vector      -       binary array of shape (n_recipes,n), indicating which ingredients are necessary for a given recipe
    OUTPUT
    binary vector of length n with one less ingredient'''
    masked_recipes = mask_cocktails(included_ingredients, ingredients_vector)
    ingredient_to_exclude = masked_recipes.sum(axis=0).argmin()
    new_included_ingredients = included_ingredients.copy()
    new_included_ingredients[ingredient_to_exclude] = 0
    return new_included_ingredients

#TODO: convert into scipy sparse matrix format with embedder
#TODO: convert oz into mg
#TODO: sort by exoticness

drinks = pre.load_data()

ingredients = pd.Series(drinks[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
ingredient_counts = ingredients.value_counts()
#ingredient_counts.iloc[ingredient_counts.values>1].iloc[::-1].plot.barh(figsize=(8,25))
#plt.savefig('common_ingredients.svg')

ingredients_vector, mlb = pre.tokenize_data(drinks)

included_ingredients = np.ones(ingredients_vector.shape[1])

reduced_ingredients = [included_ingredients]
cocktails_left = [ingredients_vector.shape[0]]
excluded_cocktail = ['none']

for i in range(ingredients_vector.shape[1]):
    reduced_ingredients.append(stepwise_exclusion(reduced_ingredients[-1], ingredients_vector))
    cocktails_left.append(np.logical_not(ingredients_vector[:, reduced_ingredients[-1]==0].any(axis=1)).sum())
    excluded_cocktail.append(mlb.classes_[np.where(reduced_ingredients[-2]-reduced_ingredients[-1])])

sns.set_style('white')
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(np.arange(309),cocktails_left)
#plt.xticks(np.arange(309), excluded_cocktail, rotation=90)
fig.savefig('cocktails_per_ingredients_excluded.png')
plt.xlabel('Ingredients excluded')
plt.ylabel('Cocktail recipes')
plt.close()

ingredient_list_to_use = reduced_ingredients[-80]
print('Uns bleiben {} Rezepte mit diesen Zutaten:'.format(cocktails_left[-80]))
print(mlb.classes_[np.where(ingredient_list_to_use)])

