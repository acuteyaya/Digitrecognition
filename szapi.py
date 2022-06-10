'''
File Name          : api.py
Author             : CUTEYAYA
Version            : V2.0.0
Created on         : 2022/03/09
'''
import requests
import base64
import json
import cv2
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
import threading as thr
import os
import random
import socket
import time
import serial
k=18
swi=0
usart='COM20'
url='172.20.10.20' #my ip socket
url1 = 'http://192.168.31.246:4747/video'
url2 = r"https://aip.baidubce.com/rest/2.0/ocr/v1/numbers"#baiduapi
wz1=(0,0,0,0,0,'',0)
path = r"E:\window\tiaoshi\pycharm\yolo\imgtemp\temp.jpg"#临时
def CatchUsbVideo(window_name, camera_idx):
    global wz1,wz2,swi
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color1 = (c1, c2, c3)
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color2 = (c1, c2, c3)
    cap = cv2.VideoCapture(camera_idx)
    cv2.namedWindow(window_name, 0)
    while cap.isOpened():
        ok, frame = cap.read()
        cv2.rectangle(frame, (0, 0), (640,480), color1,3)
        cv2.rectangle(frame, (0, 0), (320, 240), color1, 1)
        cv2.rectangle(frame, (320, 240), (640, 480), color1, 1)
        if (not ok):
            continue
        img_name = r'%s\temp.jpg' % (r"ima")
        threadLock.acquire()
        cv2.imwrite(img_name, frame)
        threadLock.release()

        swi=1
        if(wz1[0]):
            cv2.rectangle(frame, (wz1[1], wz1[3]), (wz1[1] + wz1[2], wz1[3] + wz1[4]), color2, 2)
            cv2.putText(frame, 'sz:' + wz1[5], (wz1[1], wz1[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
            cv2.rectangle(frame, (wz1[1], wz1[3]), ((wz1[1]*2+wz1[2])//2, (wz1[3]*2+wz1[4])//2), color2, 1)
            cv2.rectangle(frame, ((wz1[1]*2+wz1[2])//2, (wz1[3]*2+wz1[4])//2), (wz1[1] + wz1[2], wz1[3] + wz1[4]), color2, 1)
        cv2.imshow(window_name, frame)
        boardkey = cv2.waitKey(1) & 0xFF
        if boardkey == 32:
            break
    cap.release()
    cv2.destroyAllWindows()
def yaapi():
    global wz1,wz2,url2
    while(not swi):
        pass
    while(1):
        f = open(r"ima\temp.jpg", 'rb')
        img = base64.b64encode(f.read())
        f.close()
        params = {"image": img}
        access_token = ''
        request_url = url2 + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        time.sleep(float(k/10))
        response = requests.post(request_url, data=params, headers=headers)
        n = json.loads(response.text)
        print(n)
        if(list(n.keys())[0]!='words_result'):
            continue
        if(n['words_result']!=[]):
            if (int(n['words_result_num']) == 1):
                x1 = int(n['words_result'][0]['location']['left'])
                w1 = int(n['words_result'][0]['location']['width'])
                y1 = int(n['words_result'][0]['location']['top'])
                h1 = int(n['words_result'][0]['location']['height'])
                if (n['words_result'][0]['words'] == '0'):
                    cla1 = '0'
                    tag1 = 10
                elif (n['words_result'][0]['words'] == '1'):
                    cla1 = '1'
                    tag1 = 1
                elif (n['words_result'][0]['words'] == '2'):
                    cla1 = '2'
                    tag1 = 2
                elif (n['words_result'][0]['words'] == '3'):
                    cla1 = '3'
                    tag1 = 3
                elif (n['words_result'][0]['words'] == '4'):
                    cla1 = '4'
                    tag1 = 4
                elif (n['words_result'][0]['words'] == '5'):
                    cla1 = '5'
                    tag1 = 5
                elif (n['words_result'][0]['words'] == '6'):
                    cla1 = '6'
                    tag1 = 6
                elif (n['words_result'][0]['words'] == '7'):
                    cla1 = '7'
                    tag1 = 7
                elif (n['words_result'][0]['words'] == '8'):
                    cla1 = '8'
                    tag1 = 8
                elif (n['words_result'][0]['words'] == '9'):
                    cla1 = '9'
                    tag1 = 9
                else:
                    cla1 = ''
                    tag1 = 0
                threadLock.acquire()
                wz1 = (1, x1, w1, y1, h1, cla1, tag1)
                threadLock.release()
        else:
            threadLock.acquire()
            wz1 = (0, 0, 0, 0, 0, '', 0)
            threadLock.release()
if __name__ == '__main__':
    threadLock = thr.Lock()
    threads = []
    thread1 = thr.Thread(target=yaapi)
    thread1.start()
    a=input("start")
    thread2 = thr.Thread(target=CatchUsbVideo("ya",url1))
    thread2.start()
    threads.append(thread1)
    threads.append(thread2)
    for t in threads:
        t.join()
    print("Exiting")
