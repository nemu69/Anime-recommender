from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy as sp #pivot egineering
#ML model
from sklearn.metrics.pairwise import cosine_similarity


#LINK TO THE DATASET https://www.kaggle.com/hernan4444/anime-recommendation-database-2020


def cosine_similarity(name):
	# Vectors
	vec_a = [1, 2, 3, 4, 5]
	vec_b = [1, 3, 5, 7, 9]
	# Dot and norm
	
	dot = np.dot(vec_a, vec_b)
	norm_a = sum(a*a for a in vec_a) ** 0.5
	norm_b = sum(b*b for b in vec_b) ** 0.5
	# Cosine similarity
	cos_sim = dot / (norm_a*norm_b)
	return cos_sim


def add_column_foreign_key(data, data_to_add, column_name):
	"""
	This function is used to join 2 dataframes according to the foreign key.
	"""
	#add a column according to anime_id
	return data.merge(data_to_add, on=column_name, how='left')

if __name__ == "__main__":

	# Load data
	data = pd.read_csv("datasets/anime.csv")
	#user_data = pd.read_csv("datasets/animelist.csv")
	data_rating = pd.read_csv("datasets/rating_complete.csv", nrows=5000000)

	# Analyze data
	
	print(data_rating.columns)
	print(data.columns)
	
	print(data["Type"].unique())

	#rename column MAL_ID to anime_id
	data = data.rename(columns={"MAL_ID": "anime_id"})

	data_rating = add_column_foreign_key(data_rating, data[["Type", "Name", "anime_id"]], "anime_id")

	print(data_rating.head())
		
	#Keep only TV series
	data_rating = data_rating[data_rating["Type"] == "TV"]
	print("The number of anime is : ",data_rating["anime_id"].unique().size) # # 4632

	data_rating = data_rating[['user_id', 'Name', 'rating']]
	print("The number of rating is : ",data_rating["rating"].count()) # 13390829

	print(data_rating.head())

	pivot_table = data_rating.pivot_table(index=['user_id'], columns=['Name'], values='rating')
	print(pivot_table.head())
	# step 1
	pivot_n = pivot_table.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

	# step 2
	pivot_n.fillna(0, inplace=True)

	# step 3
	pivot_n = pivot_n.T

	# step 4
	pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

	# step 5
	piv_sparse = sp.sparse.csr_matrix(pivot_n.values)

	#model based on anime similarity
	anime_similarity = cosine_similarity(piv_sparse)

	#Df of anime similarities
	ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)
	ani_sim_df.to_csv("datasets/anime_similarity.csv", index=True)

	ani_name = 'Shingeki no Kyojin'
	print(ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6])
	
