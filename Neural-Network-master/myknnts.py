import joblib
import cv2
import numpy as np
droid=r'http://192.168.31.244:4747/video'
def yaimg(imgt):
    img=imgt[0]
    yamax=max(img)
    yamin=min(img)
    t1=yamax-yamin
    if(t1==0):
        t1=1
    j=0
    for i in img:
        imgt[0][j]=16-int((i-yamin)*16/t1)
        if(imgt[0][j]<6):
            imgt[0][j]=0
        if (imgt[0][j] > 10):
            imgt[0][j] = 16
        j=j+1
    return imgt
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
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        yaframe0 = gray.copy()

        ret, yaframe1 = cv2.threshold(yaframe0, 100, 255, 1)

        yaframe1= cv2.erode(yaframe1, None, iterations=1)  # 黑腐蚀
        yaframe1 = cv2.dilate(yaframe1, None, iterations=40)  # 膨胀
        binary = cv2.Canny(yaframe1 , 80, 500)

        cv2.imshow('yy', binary)
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
            if (area < 5000 and area <= 30000):
                continue
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            temp = findbox(box)
            if (temp[0] > 0 and temp[1] > 0 and temp[2] > 0 and temp[3] > 0 and  temp[3]-temp[1] >=8 and   temp[2]-temp[0] >=8):
                roi = yaframe0[temp[1]:temp[3], temp[0]:temp[2]]
                lentempx = temp[2] - temp[0]
                lentempy = temp[3] - temp[1]

                lentemp = min(lentempx ,lentempy)//3
                wz=roi[0][0]
                #roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=wz)

                cv2.imshow('sz', roi)
                roi = cv2.resize(roi, (8, 8))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 64))
                img = img.tolist()
                img = yaimg(img)

                y_train_pred = knn.predict(img)
                yapre = y_train_pred[0]
                kcout = kcout + 1
                l[yapre] = l[yapre] + 1
                if (kcout == 40):
                    print(l)
                    yapre = l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    kcout = 0
                    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    # yapreend = 1
                    # break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
            break
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
