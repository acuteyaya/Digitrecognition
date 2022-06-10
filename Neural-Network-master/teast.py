import cv2
import numpy as np
import sys
from scipy.special import expit
#droid=r'http://192.168.43.208:4747/video'
droid=r'http://192.168.31.244:4747/video'
yaya=np.zeros((1,784),np.uint8)
class NeuralNetMLP(object):
    def __init__(self, n_output, n_features, n_hidden=30, l1=0.0,
                 l2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0,
                 shuffle=True, minibatches=1, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self, y, k):
        '''

        :param y:
        :param k:
        :return:
        '''
        onehot = np.zeros((k, y.shape[0]))
        for idx, val, in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        return expit(z)

    def _sigmoid_gradient(self, z):
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how =='row':
            X_new = np.ones((X.shape[0]+1, X.shape[1]))
            X_new[1:,:] = X
        else:
            raise AttributeError("'how' must be 'column' or 'row'")
        return X_new

    def _feedforward(self, X, w1, w2):
        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

    def _L1_reg(self, lambda_, w1, w2):
        return (lambda_/2.0) * (np.abs(w1[:,1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        term1 = -y_enc * (np.log(output))
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, w1, w2)
        L2_term = self._L2_reg(self.l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w2, w1):
        # 反向传播
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)
        # 调整
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))

        return grad1, grad2

    def predict(self, X):
        #print(type(X))
        #print(X.shape)
        a1, z2, a2, z3, a3 = self._feedforward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            # 自适应学习率
            self.eta /= (1 + self.decrease_const*i)
            #print(type(self.eta))
            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)
            for idx in mini:
                # 前馈
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.w1, self.w2)
                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)
                self.cost_.append(cost)

                # 通过反向传播计算梯度
                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx],
                                                  w1=self.w1, w2=self.w2)

                # 更新权重
                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
    def saveya(self):
        np.save("w1.npy", self.w1)
        np.save("w2.npy", self.w2)
    def loadya(self):
        self.w1 = np.load("w1.npy")
        self.w2 = np.load("w2.npy")

def findbox(box):
    list1=[]
    xmin=xmax=box[0][0]
    ymin=ymax=box[0][1]
    for i in box:
        if (i[0] > xmax):
            xmax = i[0]
        if (i[0] < xmin):
            xmin = i[0]
        if (i[1] > ymax):
            ymax = i[1]
        if (i[1] < ymin):
            ymin = i[1]
    '''
    if(1):
        centertemp1=(xmax+xmin)//2
        centertemp2=(ymax+ymin)//2
        lentemp=max((xmax-xmin),(ymax-ymin))//2
        xmin = centertemp1 - lentemp
        ymin = centertemp2 - lentemp
        xmax = centertemp1 + lentemp
        ymax = centertemp2 + lentemp
    lentemp = 10
    list1.append(xmin-lentemp)
    list1.append(ymin-lentemp)
    list1.append(xmax+lentemp)
    list1.append(ymax+lentemp)
    '''
    yatemp=min(xmin,ymin)//5
    list1.append(xmin+yatemp)
    list1.append(ymin+yatemp)
    list1.append(xmax-yatemp)
    list1.append(ymax-yatemp)

    return list1
if __name__ == '__main__':
    nn = NeuralNetMLP(n_output=10,
                      n_features=yaya.shape[1],
                      n_hidden=50,
                      l2=0.1,
                      l1=0.0,
                      epochs=500,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      shuffle=True,
                      minibatches=50,
                      random_state=1)
    nn.loadya()

    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(droid)
    kcout = 0
    yapreend = -1
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yaframe0 = gray.copy()
        ret, yaframe = cv2.threshold(yaframe0,50, 255, 1)
        yaframe = cv2.dilate(yaframe, None, iterations=4)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀


        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(gray, 30, 120)
        cv2.imshow('yy1', gray)
        contours, hierarchy = cv2.findContours(binary,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        draw_img = frame.copy()
        j = 0
        yapre = -1
        for i in range(len(contours)):
            # 筛掉面积过小的轮廓
            j = j + 1
            area = cv2.contourArea(contours[i])
            if (area < 5000 and area <= 10000):
                continue
            # 找到包含轮廓的最小矩形框
            rect = cv2.minAreaRect(contours[i])
            # 计算矩形框的四个顶点坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box)
            temp = findbox(box)
            # print(temp)
            if (temp[0] >= 0 and temp[1] >= 0 and temp[2] >= 0 and temp[3] >= 0):
                roi = yaframe[temp[1]:temp[3], temp[0]:temp[2]]
                lentempx = temp[2] - temp[0]
                lentempy = temp[3] - temp[1]
                if (lentempx > lentempy):
                    lentemp = lentempx - lentempy
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((lentemp, lentemp), (0, 0)), 'constant', constant_values=0)
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)


                lentemp = lentempx // 3
                roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                cv2.imshow('sz', roi)
                # cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,255,255), 2)
                roi = cv2.resize(roi, (28, 28))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 784))
                y_train_pred = nn.predict(img)
                yapre = y_train_pred[0].item()
                kcout = kcout + 1
                l[yapre] = l[yapre] + 1
                if (kcout == 20):

                    print(l)
                    yapre = l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    kcout = 0
                    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    #yapreend = 1
                    #break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
