import cv2
import numpy as np
url1 = 'http://172.20.10.2:4747/video'
droid=r'http://192.168.31.244:4747/video'
def run(x,y):
    print(x,y)

yacenter=320
yahight =150
yaxw=30

speed = 20
yakp = 0.1
yaki = 0
yakd = 0.1

yalast = 0
yapidasum = 0
yapidcoumt = 1

def yarunlr0(yapy):#pd
    global yalast
    yaturn = yakp * yapy +  yakd * (yapy-yalast)
    yalast = yapy
    yarunl = int(speed + yaturn)
    yarunr = int(speed - yaturn)
    if(yarunl<0):
        yarunl=0
    elif(yarunl>40):
        yarunl=40
    if (yarunr < 0):
        yarunr = 0
    elif (yarunr > 40):
        yarunr = 40
    run(yarunl, yarunr)
def yarunlr1(yapy):#pid
    global yalast,yapidsum,yapidcoumt
    yapidsum = yapidsum + yapy
    yaturn = yakp * yapy +  yakd * (yapy-yalast) + yaki * yapidsum / yapidcoumt
    if(yapidsum / yapidcoumt < 1):
        yapidsum = 0
        yapidcoumt = 1
    else:
        yapidcoumt = yapidcoumt + 1
    yalast = yapy
    yarunl = int(speed + yaturn)
    yarunr = int(speed - yaturn)
    if(yarunl<0):
        yarunl=0
    elif(yarunl>40):
        yarunl=40
    if (yarunr < 0):
        yarunr = 0
    elif (yarunr > 40):
        yarunr = 40
    run(yarunl, yarunr)
while (1):
    f = int(input("fs:"))
    cap = cv2.VideoCapture(droid)
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # huidu
        yaframe0 = gray.copy()  # fuzhi
        retval, dst = cv2.threshold(yaframe0, 0, 255, cv2.THRESH_OTSU)  # er zhi
        dst = cv2.dilate(dst, None, iterations=4)  # 膨胀
        dst = cv2.erode(dst, None, iterations=3)  # 黑腐蚀
        cv2.imshow('xx', dst)
        cv2.imshow("yaxx1", dst[yahight-50:yahight+50, yacenter - 80:yacenter + 80])
        color = dst[yahight]
        white_count = np.sum(color == 0)
        white_index = np.where(color == 0)
        if white_count == 0:
            continue
        if (f):
            center = white_index[0][white_count - 1] - yacenter - yaxw
        else:
            center = white_index[0][0] - yacenter + yaxw
        print(center)
        yarunlr0(center)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()