
import os
import numpy as np
import struct
from sklearn import neighbors
import pickle
import joblib
def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
if __name__ == '__main__':
    path = 'mnist'  # 路径
    X_train, y_train = load_mnist(path, kind='train')  # X_train : 60000*784
    X_test , y_test  = load_mnist(path, kind='t10k')  # X_test : 10000*784
    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test  = X_test.tolist()
    y_test  = y_test.tolist()
    X_train=X_train[0:1000]
    y_train=y_train[0:1000]
    X_test= X_test[0:3000]
    y_test=y_test[0:3000]
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_test, y_test )

    joblib.dump(knn, 'knn.pkl')
    y_train_pre = knn.predict(X_train)

    sum=0
    j=0
    yalen=len(y_train_pre)
    print(yalen)
    for i in y_train_pre:
        if(i==y_train[j]):
            sum=sum+1
        j=j+1
    print(sum/yalen)
    '''
    # 初始化KNN，训练数据、测试KNN，k=5
    knn = cv2.ml.KNearest_create()
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    ret, result, neighbours, dist = knn.findNearest(X_test, k=5)

    # 分类的准确率
    # 比较结果和test_labels
    matches = result == y_test
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print(accuracy)
    '''