import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
  return np.sqrt(np.sum((x1- x2)**2))

class kNN:
  def __init__(self, k=3):
    self.k= k
    
  def fit(self, X, y):
    self.X_train= X
    self.y_train= y
    
  def predict(self, X):
    y_pred= [self._predict(x) for x in X]
    return np.array(y_pred)
    
  def _predict(self, x):
    distances= [euclidean_dist(x, x_train) for x_train in self.X_train]
    
    k_idx= np.argsort(distances)[:self.k]
    # print(k_idx)
    k_neighbour_labels= [self.y_train[i] for i in k_idx]
    # print(k_neighbour_labels)
    most_common_labels= Counter(k_neighbour_labels).most_common(1)
    
    return most_common_labels[0][0]
    
    

if __name__== '__main__':
  
  from matplotlib.colors import ListedColormap
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  
  cmap= ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  def accuracy(y_groundtruth, y_pred):
    accuracy= np.sum(y_groundtruth== y_pred)/ len(y_groundtruth)
    return accuracy
  
  iris= datasets.load_iris()
  X, y= iris.data, iris.target
  
  X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state= 1234)
  
  k= 7
  clf= kNN(k=k)
  clf.fit(X_train, y_train)
  predictions= clf.predict(X_test)
  
  print('Accuracy for our kNN classifier:', accuracy(y_test, predictions))