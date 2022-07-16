import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with hamming distance """

  def __init__(self, k=1):
    """
    Initializing the KNN object

    Inputs:
    - k: The number of nearest neighbors.
    """
    self.k = k
    
  def fit(self, X_train, y_train):
    """
    This method fits the data, which means
    memorizing the training data. 
    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D) 
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
    self.X_train = X_train
    self.y_train = y_train

  
  def predict(self, X_test): 
    """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just 
    memorizing the training data. 
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) 

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists, k=self.k)


  def compute_distances(self, X_test):
    """
    Compute the hamming distance between each test point in X_test and each training point
    in self.X_train.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the hamming distance between the ith test point and the jth training
      point.
    """
    ##### YOUR CODE STARTS HERE ##### 

    dist = np.zeros((len(X_test), len(self.X_train)))
    for i in range(len(X_test)):
      for j in range(len(self.X_train)):
        dist[i,j]= np.sum(X_test[i] != self.X_train[j])



    ##### YOUR CODE ENDS HERE #####          
    return dist

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data  
    """
    ##### YOUR CODE STARTS HERE #####  
    n_dists = np.argsort(dists , axis = 1)
    n_dists =  n_dists[:, :k]
    clas = np.unique(self.y_train)
    for i in range(len(self.y_train)):
      for j in range(len(clas)):
        if self.y_train[i] == clas[j]:
          self.y_train[i] = j
    y_pred = np.zeros((n_dists.shape[0]), dtype = str)
    for i in range(n_dists.shape[0]):
      u,c = np.unique(self.y_train[n_dists[i]], return_counts=True)
      y_pred[i] = u[np.argmax(c)]



    ##### YOUR CODE ENDS HERE #####              
    return y_pred
