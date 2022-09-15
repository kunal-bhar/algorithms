import numpy as np

def r2_score(y_truth, y_pred):
  corr_matrix= np.corrcoef(y_truth, y_pred)
  corr= corr_matrix[0, 1]
  return corr**2

class LinearRegression:
  def __init__(self, learning_rate= 0.001, n_iters= 1000):
    self.learning_rate= learning_rate
    self.n_iters= n_iters
    self.weights= None
    self.bias= None
  
  def fit(self, X, y):
    n_samples, n_features= X.shape
    
    self.weights= np.zeros(n_features)
    self.bias= 0
    
    for _ in range(self.n_iters):
      y_predicted= np.dot(X, self.weights)+ self.bias
      dw= (1/ n_samples)* np.dot(X.T, (y_predicted- y))
      db= (1/ n_samples)* np.sum(y_predicted- y)
      
      self.weights-= self.learning_rate* dw
      self.bias-= self.learning_rate* db
    
  def predict(self, X):
    y_approximated= np.dot(X, self.weights)+ self.bias
    return y_approximated

if __name__== '__main__':
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn import datasets
  
  def mean_squared_error(y_truth, y_pred):
    return np.mean((y_truth- y_pred)**2)
  
  X, y= datasets.make_regression(
    n_samples= 100, n_features= 1, noise= 15, random_state= 4
  )
  
  X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size= 0.25, random_state= 1234
  )
  
  regressor= LinearRegression(learning_rate= 0.01, n_iters= 1000)
  regressor.fit(X_train, y_train)
  predictions= regressor.predict(X_test)
  
  mse= mean_squared_error(y_test, predictions)
  print('Mean Squared Error for our Linear Regressor:', mse)
  
  accuracy= r2_score(y_test, predictions)
  print('R^2 Accuracy for our Linear Regressor:', accuracy)
  
  y_pred_line= regressor.predict(X)
  
  cmap= plt.get_cmap('viridis')
  fig= plt.figure(figsize= (8, 6))
  m1= plt.scatter(X_train, y_train, color= cmap(0.9), s= 10)
  m2= plt.scatter(X_test, y_test, color= cmap(0.5),s= 10)
  plt.plot(X, y_pred_line, color= 'black', linewidth= 2, label= 'Prediction')
  plt.show()