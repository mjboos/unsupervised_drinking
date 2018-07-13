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

def find_ingredient(ingredient, ingredient_names):
    import re
    ingr_ind = np.zeros(len(ingredient_names))
    if '*' in ingredient:
        ingredient = ingredient.strip('*')
        regex = r'\b({})\b'.format(ingredient)
        matchfunc = re.search
    else:
        regex = r'\b({})\b'.format(ingredient)
        matchfunc = re.match

    for i, ingredient_name in enumerate(ingredient_names):
        match = matchfunc(regex, ingredient_name, flags=re.IGNORECASE)
        if match is not None:
            ingr_ind[i] = 1
    return ingr_ind

def flexible_load_defaults(ingredient_names, default_fn='default_ingredients.csv'):
    '''Loads default ingredients from csv
    wild cards are allowed, e.g.
    *rum*
    to include all ingredients containing rum in their name (light rum, spiced rum etc)'''
    ingredients = np.genfromtxt(default_fn, encoding='utf-8', dtype='str', delimiter=',')
    ingredients = [ingredient.strip() for ingredient in ingredients]
    print(ingredients)
    ingredient_indicator = np.zeros(len(ingredient_names))
    for ingredient in ingredients:
        try:
            ingredient_position = np.where(find_ingredient(ingredient, ingredient_names))[0]
            ingredient_indicator[ingredient_position] = 1
        except:
            continue
    return ingredient_indicator

def mask_cocktails(included_ingredients, ingredients_vector):
    cocktails_to_consider = np.logical_not(ingredients_vector[:,included_ingredients==0].any(axis=1))
#    masked_recipes = np.ma.masked_array(ingredients_vector, np.tile(np.logical_not(included_ingredients)[None],(ingredients_vector.shape[0],1)))
    masked_recipes = np.ma.masked_array(ingredients_vector, np.logical_or(np.logical_not(cocktails_to_consider)[:,None], np.tile(np.logical_not(included_ingredients)[None], (ingredients_vector.shape[0],1))))
    return masked_recipes

def make_description_string(masked_recipes, ingredient_names):
    return ['<br/>'.join(ingredient_names[np.where(rec)[0]]) for rec in masked_recipes[~masked_recipes.mask.all(axis=1)]]

def make_bokeh_plot(comps, recipe_names, description_strings):
    import bokeh.plotting as bpl

    bpl.output_file("drinksss.html")

    source = bpl.ColumnDataSource(data=dict(
        x=comps[:,0],
        y=comps[:,1],
        desc=recipe_names,
        ingredients=description_strings
    ))

    TOOLTIPS = [
    #    ("index", "$index"),
    #    ("(x,y)", "($x, $y)"),
        ("Cocktail", "@desc"),
        ("Ingredients", "@ingredients{safe}")
    ]
    p = bpl.figure(tools="reset,pan,wheel_zoom", tooltips=TOOLTIPS,
               title="Unsupervised Cocktails")

    p.scatter('x', 'y', marker='o', size=10, source=source)
    bpl.show(p)


drinks = pre.load_data()

ingredients = pd.Series(drinks[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
ingredient_counts = ingredients.value_counts()
#ingredient_counts.iloc[ingredient_counts.values>1].iloc[::-1].plot.barh(figsize=(8,25))
#plt.savefig('common_ingredients.svg')

ingredients_vector, mlb = pre.tokenize_data(drinks)

ingredient_names = np.array([ingredient.lower() for ingredient in mlb.classes_])

# load default cocktail ingredients
included_ingredients = flexible_load_defaults(ingredient_names)


from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE

masked_recipes = mask_cocktails(included_ingredients, ingredients_vector)
masked_recipes2 = masked_recipes[~masked_recipes.mask.all(axis=1)][:,~masked_recipes.mask.all(axis=0)]
selected_recipes = np.logical_not(masked_recipes.mask.all(axis=1))
recipe_names = drinks['strDrink'].iloc[selected_recipes].values
recipes_to_track = ['Margarita', 'Mojito', 'Cuba Libre', 'Tequila Sunrise', 'Long Island Iced Tea']
recipe_idx = [np.where(recipe_names==recipe)[0] for recipe in recipes_to_track]
selected_ingredient_names = mlb.classes_[~masked_recipes.mask.all(axis=0)]

ingredients_to_track = ['vodka', 'gin']
alc_idx = [np.where(mlb.classes_==ingr)[0] for ingr in ingredients_to_track]

colors = [{0:'g',1:'r',2:'b'}[np.where(masked_recipes[i,alc_idx])[0][0]] if masked_recipes[i,alc_idx].any() else 'y' for i in range(masked_recipes[~masked_recipes.mask.all(axis=1)].shape[0])] 


tsne = TSNE(n_components=2, early_exaggeration=20, perplexity=5, init='pca', random_state=1)
comps = tsne.fit_transform(masked_recipes2)

description_string = make_description_string(masked_recipes, ingredient_names)

