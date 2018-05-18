#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 06:11:48 2018

@author: shanmukha
"""

from surprise import Reader,Dataset,SVD,evaluate,NMF
import zipfile

#Unzip the file
"""file = zipfile.ZipFile('/home/shanmukha/AnacondaProjects/Spyder_projects/Recommendation_trail/ml-100k.zip','r')
file.extractall()
file.close()
"""
#Read dataset
reader = Reader(line_format='user item rating timestamp',sep='\t')
dataset = Dataset.load_from_file(file_path='./ml-100k/u.data',reader=reader)

#Split dataset
dataset.split(n_folds=5)

#Using SVD,NMF
algo1 = SVD()
algo2 = NMF()
#evaluate(algo,dataset,measures=['RMSE','MAE'])

#Training entire dataset
train_data = dataset.build_full_trainset()
algo1.fit(train_data)
algo2.fit(train_data)

#predicting
user = str(196)
item = str(302)
actual_rating = 4
print(algo1.predict(user,item,actual_rating))
print(algo2.predict(user,item,actual_rating))