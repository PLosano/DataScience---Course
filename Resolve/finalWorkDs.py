#LIBRERIES
import pandas as pd
import numpy as np
from pandas import * 
from datetime import datetime
from scipy.sparse import csr_matrix 
from lightfm import LightFM
from tqdm import tqdm

#DATA 

# Metadata upload

col_names = ["asset_id", "content_id", "title", "reduced_title", "episode_title", "show_type", "released_year", "country_of_origin", "category", "keywords", "description", "reduced_desc","cast_first_name", "credits_first_name", "run_time_min", "audience", "made_for_tv", "close_caption", "sex_rating", "violence_rating", "language_rating", "dialog_rating", "fv_rating", "pay_per_view", "pack_premium_1", "pack_premium_2", "create_date", "modify_date", "start_vod_date", "end_vod_date"]
metadata = pd.read_csv("C:/Users/Losano Paula/Documents/Datos Paula/NOTEBOOK/data/metadata.csv", sep=';', header=None, names=col_names)
 
# Columns that do not contribute and I am going to eliminate: 
# title, episode_title, released_year, country_of_origin, keywords, description, reduced_desc, cast_first_name, credits_first_name

metadata = metadata.drop(columns = [ "title", "episode_title", "released_year", "country_of_origin", "keywords", "description", "reduced_desc", "cast_first_name", "credits_first_name"] ,axis=1)
metadata

# Train upload
 
train= pd.read_csv("C:/Users/Losano Paula/Documents/Datos Paula/NOTEBOOK/data/train.csv")
train

# Now we add de column rating based on 'resume'. This is a trivial based in if the account resume or not the content

train['rating'] = [1 if i ==0 else 5 for i in train['resume']] 
train

# Union metadata and train 
# We have to union metada and train because we need columns of both DF 

df = pd.merge(train, metadata, left_on='asset_id', right_on='asset_id', how='left')

# Look for nulls and drop it

df.isna().sum()
df = df.dropna()

# Change the date format to operate with the dates later

df['tunein'] = pd.to_datetime(df['tunein'], format='%Y-%m-%d %H:%M:%S')
df['tuneout'] = pd.to_datetime(df['tuneout'], format='%Y-%m-%d %H:%M:%S')
df['start_vod_date'] = pd.to_datetime(df['tuneout'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)
df['end_vod_date'] = pd.to_datetime(df['end_vod_date'], format='%Y-%m-%dT%H:%M:%S').dt.tz_localize(None)

df.head()

# TRAIN AND TEST
# Split in train and test know that we take the data until March 1, 2021.

train = df[df['tunein'] < datetime(year=2021, month=3, day=1)].copy()
test = df[df['tunein'] >= datetime(year=2021, month=3, day=1)].copy()

train.head(3)
test.head(3)

# Cold start
# Search for accounts who are in the test set and are NOT on the train

test[~test.account_id.isin(train.account_id.unique())].account_id.nunique()

# INTERACTION MATRIX
# We have the sets splited into train and test, we are going to build the interaction matrix.
# To do that we use only the 3 columns that have interactions in the data set train: Account_id, content_id and raiting
# We used the pivot table from pandas

interactions = train[['account_id', 'content_id', 'rating']].copy()
interactions_matrix = pd.pivot_table(interactions, index='account_id', columns='content_id', values='rating') #pivot tables
interactions_matrix = interactions_matrix.fillna(0) # fill na interactions_matrix

interactions_matrix.head()
interactions_matrix.shape

# We are goint to transform the matrix into a csr matrix because sparse matrices can be used in arithmetic operations: they support addition, subtraction, multiplication, division, and matrix power.

interactions_matrix_csr = csr_matrix(interactions_matrix.values)
interactions_matrix_csr

# Generate a dictionary that will serve as an index later. It has as keys the id of each user and as value the index (row number in the matrix)

# Dictionary for accounts (users)

user_ids = list(interactions_matrix.index)
user_dict = {}
counter = 0
for i in user_ids:
        user_dict[i] = counter
        counter += 1

# Dictionary for items (content)

item_id = list(interactions_matrix.columns)
item_dict = {}
counter = 0 
for i in item_id:
    item_dict[i] = counter
    counter += 1

# MODEL
# We are going to use LFM (light fm) to train our model

model = LightFM(random_state=0,
                loss='warp',
                learning_rate=0.03,
                no_components=100)

model = model.fit(interactions_matrix_csr,
                  epochs=100,
                  num_threads=16, verbose=False)

# COLD START
# ¿What happend with the users we dont have ir train?
# One option is recommend the most popular content 

train.groupby("content_id", as_index=False).agg({"account_id":"nunique"})

popularity_df = train.groupby("content_id", as_index=False).agg({"account_id":"nunique"}).sort_values(by="account_id", ascending=False)
popularity_df.columns=["content_id", "popularity"]
popularity_df.head()

# Top 10,  most popular content 

popular_content = popularity_df.content_id.values[:10]
popularity_df.head(10).content_id.values
popular_content

# Now we are going to generate 20 recommendations for all users (account_id)
# Premise: 
    # Filter content that the user previously viewed
    # If the user is not in the train set, recommend the 20 most popular contents (cold user)

# Dict where we are going to store the recommendations
recomms_dict = {
    'account_id': [],
    'recomms': []
}

# We get number of users (account_id) and number of items (content_id)
n_users, n_items = interactions_matrix.shape
item_ids = np.arange(n_items)

# For each user (account_id) of the test dataset, we generate recommendations
for user in tqdm(test.account_id.unique()):
    # Validate if the user (account_id) is in the interactions matrix (interactions_matrix.index)
    if user in list(interactions_matrix.index):
      # If the user (account_id) is in train, is not a cold start. We use the model to recommend
      user_x = user_dict[user] # We look for the index of the user (account_id) in the matrix (we transform id to index)

      # Generate predicitons for user (account_id) x
      preds = model.predict(user_ids=user_x, item_ids = item_ids)

      # Based on the example above, order the predictions from least to greatest and settle for 50.
      scores = pd.Series(preds)
      scores.index = interactions_matrix.columns
      scores = list(pd.Series(scores.sort_values(ascending=False).index))[:50]

      # Obtain a list of contents previously seen by the user (in the train set)
      watched_contents = train[train.account_id == user].content_id.unique()

      # Filter contents already seen and keep the first 20
      recomms = [x for x in scores if x not in watched_contents][:20]

      # Save the recomms in the dictionary 
      recomms_dict['account_id'].append(user)
      recomms_dict['recomms'].append(scores)
    
    # In this else we will treat the users (account_id) that are not in the matrix (cold start)
    else:
      recomms_dict['account_id'].append(user)
      # We recommend popular content
      recomms_dict['recomms'].append(popular_content)

recomms_df = pd.DataFrame(recomms_dict)
recomms_df

# Now is time to compare our recomms against what users actually saw (test).

ideal_recomms =  test.sort_values(by=["account_id", "rating"], ascending=False)\
                  .groupby(["account_id"], as_index=False)\
                  .agg({"content_id": "unique"})\
                  .head()
ideal_recomms.head()

# MAP
# Measure MAP
# First of all, we are going to unite our recommendations with the ideal set in the same dataframe.

df_map = ideal_recomms.merge(recomms_df, how="left", left_on="account_id", right_on="account_id")[["account_id", "content_id", "recomms"]]
df_map.columns = ["account_id", "ideal", "recomms"]
df_map.head()

aps = [] # empty list to store the AP of each recommendation

for pred, label in df_map[["ideal", "recomms"]].values:
  n = len(pred) # number of recommended items (content_id)
  arange = np.arange(n, dtype=np.int32) + 1. # indexa in base 1
  rel_k = np.in1d(pred[:n], label) #list of booleans indicating the relevance of each item
  tp = np.ones(rel_k.sum(), dtype=np.int32).cumsum() # list with true positives counter
  denom = arange[rel_k] # positions where the related items are found
  ap = (tp / denom).sum() / len(label) # average precision
  aps.append(ap)

MAP = np.mean(aps)
print(f'mean average precision = {round(MAP, 5)}')

# This line I used to know the names of the contents, I promiss I will improved and create a function 
#data[(data['content_id'] == number of content)].head(1)