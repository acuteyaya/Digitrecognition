from sklearn.datasets import load_digits
import numpy as np
from sklearn import neighbors
import joblib
if __name__ == '__main__':
    digits = np.zeros([1797, 64])
    digitstemp = load_digits()
    for i in range(0,1797):
        x=digitstemp.images[i,:,:]
        x=x.ravel()
        digits[i]=x
    knn = neighbors.KNeighborsClassifier(n_neighbors=20)
    path = 'mnist'  # 路径
    X_train = digits
    y_train= digitstemp.target
    X_train = X_train.tolist()
    y_train = y_train.tolist()

    knn.fit( X_train ,  y_train )
    joblib.dump(knn, 'knn.pkl')



