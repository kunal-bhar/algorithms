import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1- x2)** 2))


class KMeans:
    def __init__(self):
        pass
    
    def predict(self):
        pass
    

if __name__== '__main__':
    
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers= 3, n_samples= 500, n_features= 2, shuffle= True, random_state= 40
    )
    print(X.shape)

    clusters= len(np.unique(y))
    print(clusters)

    k = KMeans()

    k.plot()


