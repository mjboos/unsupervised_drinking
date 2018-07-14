from __future__ import division
from json import loads
import pandas as pd
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pre
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import seaborn as sns
from sklearn.manifold import LocallyLinearEmbedding, MDS, TSNE
import bokeh.plotting as bpl
from bokeh.themes import Theme
from os.path import join, dirname
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
    ingredients = np.genfromtxt(join(dirname(__file__),default_fn), encoding='utf-8', dtype='str', delimiter=',')
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

def make_description_string(masked_recipes, measures, ingredient_names):
    description_list = []
    for drink, rec in enumerate(masked_recipes[~masked_recipes.mask.all(axis=1)]):
        drink_string = ''.join(['<br/> <strong>{0}</strong> - {1}'.format(
            ingredient_names[i], measures[~masked_recipes.mask.all(axis=1)][drink,i]) for i in np.where(rec)[0]])
        drink_string = '<p><font size="3">' + drink_string + '</font></p>'
        description_list.append(drink_string)
    return description_list

def make_description_without_measure_string(masked_recipes, ingredient_names):
    return ['<br/>'.join(ingredient_names[np.where(rec)[0]]) for rec in masked_recipes[~masked_recipes.mask.all(axis=1)]]

def make_bokeh_plot(comps, recipe_names, description_strings):
    colors = color_values[~masked_recipes.mask.all(axis=1)]
    source = bpl.ColumnDataSource(data=dict(
        colors=colors,
        x=comps[:,0],
        y=comps[:,1],
        images=[join('unsupervised_cocktails','static','{}.jpg'.format(i)) for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
#        images=['<img src="randomcocktails/{0}.jpg" height="42" alt="{0}" width="42" style="float: left; margin: 0px 15px 15px 0px;" border="2"></img>'.format(i) for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
#        desc=['<p><b><font size="3">'+rec+'</font></b></p>' for rec in recipe_names],
        desc = recipe_names,
        ingredients=description_strings
    ))
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@images" height="150" alt="@images" width="150"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <br>
                <span style="font-size: 17px; font-weight: bold;">@desc</span>
            </div>
            <div>
            <br>
                <span>@ingredients{safe}</span>
            </div>
        </div>
    """
#    TOOLTIPS = [
    #    ("index", "$index"),
    #    ("(x,y)", "($x, $y)"),
#        ("Cocktail", "@desc{safe}"),
#        ("","@images{safe}"),
#        ("Ingredients", "@ingredients{safe}")
#    ]
    p = bpl.figure(tools="reset,pan,wheel_zoom", tooltips=TOOLTIPS,
               title="Unsupervised Cocktails")

    p.scatter('x', 'y', marker='o', size=10, fill_color='colors', source=source)
    p.axis.visible = False
    return p, source

color_list = np.array(('#d18096', '#483496', '#00FFD0', '#00FF00', '#3FE7C3'))

drinks = pre.load_data()

#remove stupid ingredients
#exclude_these = np.vstack([drinks.apply(lambda x: word in x.values, axis=1) for word in pre.exclude_by_hand]).any(axis=0)
#drinks = drinks.drop(drinks.index[np.where(exclude_these)[0]])
print(drinks.shape)

lblenc = LabelEncoder()
alcohol_ind = lblenc.fit_transform(drinks['strAlcoholic'].values)
color_values = color_list[alcohol_ind]
ingredients = pd.Series(drinks[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
ingredient_counts = ingredients.value_counts()
#ingredient_counts.iloc[ingredient_counts.values>1].iloc[::-1].plot.barh(figsize=(8,25))
#plt.savefig('common_ingredients.svg')
ingredients_vector, mlb = pre.tokenize_data(drinks)
measure_vector = pre.from_drinks_to_measure(drinks, ingredients_vector, mlb)

ingredient_names = np.array([ingredient.lower() for ingredient in mlb.classes_])

# load default cocktail ingredients
included_ingredients = flexible_load_defaults(ingredient_names)

recipes_to_track = ['Margarita', 'Mojito', 'Cuba Libre', 'Tequila Sunrise', 'Long Island Iced Tea']
#selected_ingredient_names = mlb.classes_[~masked_recipes.mask.all(axis=0)]

def get_masked_recipes_and_names(included_ingredients, ingredients_vector):
    masked_recipes = mask_cocktails(included_ingredients, ingredients_vector)
    #masked_recipes2 = masked_recipes[~masked_recipes.mask.all(axis=1)][:,~masked_recipes.mask.all(axis=0)]
    selected_recipes = np.logical_not(masked_recipes.mask.all(axis=1))
    recipe_names = drinks['strDrink'].iloc[selected_recipes].values
    return masked_recipes, recipe_names

def make_tsne(masked_recipes, ingredient_names, recipe_names, alcohol_indicator=None):
    if alcohol_indicator is None:
        alcohol_indicator = np.zeros(masked_recipes[~masked_recipes.mask.all(axis=1)].shape[0])
    tsne = TSNE(n_components=2, early_exaggeration=20, perplexity=20, init='pca', random_state=1)
    comps = tsne.fit_transform(masked_recipes[~masked_recipes.mask.all(axis=1)][:,~masked_recipes.mask.all(axis=0)])
    print('t-SNE plot with: \n {}'.format(comps.shape))
    description_string = make_description_string(masked_recipes, measure_vector, ingredient_names)
    return comps, description_string

from bokeh.models.widgets import CheckboxGroup, Button
from bokeh.layouts import widgetbox, row, layout, column

bpl.output_file("unsupervised_cocktails.html")

masked_recipes, recipe_names = get_masked_recipes_and_names(included_ingredients, ingredients_vector)

checkbox_group = CheckboxGroup(
        labels=ingredient_names.tolist(), active=[0, 1])
checkbox_group.active = np.where(included_ingredients)[0].tolist()
comps, description_string = make_tsne(masked_recipes, ingredient_names, recipe_names)
tsne_plot, source = make_bokeh_plot(comps, recipe_names, description_string)

def update():
    '''Updates t-SNE plot'''
    print('Now in update')
    new_included_ingredients = np.zeros(len(included_ingredients))
    new_included_ingredients[checkbox_group.active] = 1
    masked_recipes, recipe_names = get_masked_recipes_and_names(new_included_ingredients, ingredients_vector)
    recipe_idx = [np.where(recipe_names==recipe)[0] for recipe in recipes_to_track]
    comps, description_string = make_tsne(masked_recipes, ingredient_names, recipe_names)
    colors = color_values[~masked_recipes.mask.all(axis=1)]
    data_dict=dict(
        colors=colors,
        x=comps[:,0],
        y=comps[:,1],
        images=[join('unsupervised_cocktails','static','{}.jpg'.format(i)) for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
#        images=['<img src="randomcocktails/{0}.jpg" height="42" alt="{0}" width="42" style="float: left; margin: 0px 15px 15px 0px;" border="2"></img>'.format(i) for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
#        desc=['<p><b><font size="3">'+rec+'</font></b></p>' for rec in recipe_names],
        desc = recipe_names,
        ingredients=description_strings)
    source.data = data_dict
    print(mlb.classes_[~masked_recipes.mask.all(axis=0)])

update_button = Button(label="Update")
update_button.on_click(update)
widget = widgetbox(update_button, checkbox_group)
overall_layout = row(tsne_plot, widget)
bpl.curdoc().theme = Theme(join(dirname(__file__),"theme.yaml"))
bpl.curdoc().add_root(overall_layout)
