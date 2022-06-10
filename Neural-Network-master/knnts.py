import joblib
import cv2
import numpy as np
droid=r'http://192.168.31.244:4747/video'
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
    knn = joblib.load('knn.pkl')
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
        ret, yaframe = cv2.threshold(yaframe0, 50, 255, 1)
        yaframe = cv2.dilate(yaframe, None, iterations=6)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=5)  # 黑腐蚀
        yaframe = cv2.dilate(yaframe, None, iterations=2 )  # 膨胀
        cv2.imshow('sz1', yaframe)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(gray, 30, 120)
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
                    lentemp = lentempy // 4
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)
                    lentemp = lentempx // 4

                if(lentemp>0):
                    roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                    cv2.imshow('sz', roi)
                    roi = cv2.resize(roi, (28, 28))
                    img = np.array(roi).astype(np.float32)
                    img = img.reshape((1, 784))
                    img = img.tolist()
                    y_train_pred = knn.predict(img)
                    yapre = y_train_pred[0]
                    kcout = kcout + 1
                    l[yapre] = l[yapre] + 1
                    if (kcout == 20):
                        print(l)
                        yapre = l.index(max(l))
                        print(yapre)
                        # (str(y_train_pred[0])).encode('utf-8')
                        kcout = 0
                        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # yapreend = 1
                        # break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
