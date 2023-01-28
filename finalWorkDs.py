#Used libreries 
import pandas as pd
import numpy as np
from pandas import * 
from datetime import datetime

#Data 

#Metadata upload

col_names = ["asset_id", "content_id", "title", "reduced_title", "episode_title", "show_type", "released_year", "country_of_origin", "category", "keywords", "description", "reduced_desc","cast_first_name", "credits_first_name", "run_time_min", "audience", "made_for_tv", "close_caption", "sex_rating", "violence_rating", "language_rating", "dialog_rating", "fv_rating", "pay_per_view", "pack_premium_1", "pack_premium_2", "create_date", "modify_date", "start_vod_date", "end_vod_date"]

metadata = pd.read_csv("C:/Users/Losano Paula/Documents/Datos Paula/NOTEBOOK/data/metadata.csv", sep=';', header=None, names=col_names)
 
#Columns that do not contribute and I am going to eliminate: 
#title, episode_title, released_year, country_of_origin, keywords, description, reduced_desc, cast_first_name, credits_first_name

metadata = metadata.drop(columns = [ "title", "episode_title", "released_year", "country_of_origin", "keywords", "description", "reduced_desc", "cast_first_name", "credits_first_name"] ,axis=1)
