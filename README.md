# Unsupervised Drinking

![Usage](https://github.com/mjboos/unsupervised_drinking/blob/master/usage.gif "Tasty...")

This is a small bokeh app that solves the age old question: what should I drink next?
It uses unsupervised learning ([t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)) to show all cocktail recipes that are possible with the ingredients you have at home and lets you explore the mythical cocktail space (which is two dimensional, so you can actually see something).

Finally, you can get drunk in a data-driven way.

## How to set up

An easy way to set everything up from the terminal:
```
git clone https://github.com/mjboos/unsupervised_drinking.git
cd unsupervised_drinking
pip install -r requirements.txt
```

You can run bokeh in a jupyter notebook or output it to an html file, but the full package only works using a bokeh server:
`bokeh serve --show unsupervised_drinking`

## How to use

You can mark the location of a cocktail you like by selecting it in the drop-down menu, then explore its neighbourhood.
But always remember: don't drink and ~~program~~ drive. 

## How to fine-tune

You can actually adapt with which cocktails the app starts as a default (so you don't have to click on everything everytime you restart it). Just edit the `default_ingredients.csv` file.
