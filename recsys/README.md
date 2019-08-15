# Recommender Systems Tutorial
In this tutorial, we provide a simple walkthrough of how to use Snorkel to build a recommender system.
We consider a setting similar to the [Netflix challenge](https://www.kaggle.com/netflix-inc/netflix-prize-data), but with books instead of movies.
We use book metadata and review text to create labeling functions that estimate user ratings for books they have read or plan to read.
We then use Snorkel's `LabelModel` to combine the outputs of those labeling functions, and train a model to predict whether a user will read and like any given book (and therefore what books should be recommended to the user) based only on what books the user has interacted with in the past.
