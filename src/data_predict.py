import pandas as pd
import numpy as np
import sys
import os

def anime_recommendation(ani_name, ani_sim_df):
	"""
	This function will return the top 5 shows with the highest cosine similarity value and show match percent
	
	example:
	>>>Input: 
	
	anime_recommendation('Death Note')
	
	>>>Output: 
	
	Recommended because you watched Death Note:

					#1: Code Geass: Hangyaku no Lelouch, 57.35% match
					#2: Code Geass: Hangyaku no Lelouch R2, 54.81% match
					#3: Fullmetal Alchemist, 51.07% match
					#4: Shingeki no Kyojin, 48.68% match
					#5: Fullmetal Alchemist: Brotherhood, 45.99% match 

			   
	"""
	number = 1
	print('Recommended because you watched {}:\n'.format(ani_name))
	for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
		print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
		number +=1 

if __name__ == "__main__":

	# Load data
	ani_sim_df = pd.read_csv("datasets/anime_similarity.csv", index_col=0)

	anime_name = sys.argv[1]

	try:
		anime_name_df = ani_sim_df[anime_name]
	except:
		print("Anime not found")
		sys.exit(1)
		
	anime_recommendation(anime_name, ani_sim_df)
	
