import numpy as np
from collections import Counter

from DecisionTree import DecisionTree


def bootstrap_sample(X, y):
    n_samples= X.shape[0]
    idxs= np.random.choice(n_samples, n_samples, replace= True)
    
    return X[idxs], y[idxs]

def most_common_label(y):
    counter= Counter(y)
    most_common= counter.most_common(1)[0][0]
    
    return most_common


class RandomForest:
    
    def __init__(
            self, n_trees= 10, min_samples_to_continue= 2, max_depth= 10, n_feats= None
            ):
        self.n_trees= n_trees
        self.min_samples_to_continue= min_samples_to_continue
        self.max_depth= max_depth
        self.n_feats= n_feats
        self.trees= []

    def fit(self, X, y):
        self.trees= []
        
        for _ in range(self.n_trees):
            
            tree= DecisionTree(
                min_samples_to_continue= self.min_samples_to_continue, max_depth= self.max_depth, n_feats= self.n_feats
                )
            
            X_sample, y_sample= bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        tree_preds_init= np.array([tree.predict(X) for tree in self.trees])
        tree_preds= np.swapaxes(tree_preds_init, 0 ,1)
        
        y_pred= [most_common_label(tree_pred) for tree_pred in tree_preds]
        
        return np.array(y_pred)
    
    
if __name__== '__main__':
    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    def accuracy(y_truth, y_pred):
        accuracy= np.sum(y_truth== y_pred)/ len(y_truth)
        return accuracy
    
    data= datasets.load_breast_cancer()
    X= data.data
    y= data.target
    
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size= 0.25, random_state= 123
        )
    
    clf= RandomForest(n_trees= 11, max_depth= 10)
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    
        
    print('Accuracy for our Random-Forest Classifier:', accuracy(y_test, y_pred))
