#Task 6 : Prediction using Decision Tree Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Iris.csv')
print("Data imported successfully")

feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = dataset[feature_cols]
y = dataset.Species

from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier(random_state = 0)
regressor.fit(X, y)
print("Decision Tree Classifier Created")

from sklearn.tree import plot_tree
plt.figure(figsize=(30,20))

tree_img= plot_tree(regressor,feature_names = feature_cols, class_names=dataset['Species'].unique().tolist(), 
                    precision=4,label="all",filled=True)
plt.show()