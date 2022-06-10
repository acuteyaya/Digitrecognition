'''
File Name          : yapi.py
Author             : CUTEYAYA
Version            : V1.0.0
Created on         : 2022/04/29
'''
import numpy as np
import sys
from scipy.special import expit
import requests
import base64
import json
import cv2
import threading as thr
import random
import socket
import time
import RPi.GPIO as GPIO
import cv2
import pyzbar.pyzbar as pyzbar
import time
import _thread
import requests
swj= 1#guan
# xml文件路径
cascade_path = '/home/pi/Downloads/haarcascade_frontalface_default.xml'

# 定义白名单
white_list = ['xiaoyi,xiaoer,xiaosan']

# 是否发现有效二维码
find_qrcode = True
# 是否鸣笛警告
waring = False

# 小车电机引脚定义
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13

# RGB三色灯引脚定义
LED_R = 22
LED_G = 27
LED_B = 24

# 小车按键定义
key = 8

# 蜂鸣器引脚定义
buzzer = 8

# 循迹红外引脚定义
# TrackSensorLeftPin1 TrackSensorLeftPin2 TrackSensorRightPin1 TrackSensorRightPin2
#      3                 5                  4                   18
TrackSensorLeftPin1 = 3  # 定义左边第一个循迹红外传感器引脚为3口
TrackSensorLeftPin2 = 5  # 定义左边第二个循迹红外传感器引脚为5口
TrackSensorRightPin1 = 4  # 定义右边第一个循迹红外传感器引脚为4口
TrackSensorRightPin2 = 18  # 定义右边第二个循迹红外传感器引脚为18口

# 超声波引脚定义
EchoPin = 0
TrigPin = 1
# 舵机引脚定义
ServoPin = 11

ServoPin0 = 9


HOST ='192.168.50.138'

PORT = 520
s1='00s'
w='00w'
a='00a'
d='00d'
stop='sto'
all='all'
xx='0xx'
sz='0sz'
ewm='ewm'
MD1='0m1'
data=''
temp=''
sztag = 1
ewmtag = 1
alltag = 1
xxtag = 1
md1tag = 1
k=10
usart='COM20'
droid=r'http://192.168.31.246:4747/video'
yaya=np.zeros((1,784),np.uint8)
url1 = 'http://192.168.31.246:4747/video'
url2 = r"https://aip.baidubce.com/rest/2.0/ocr/v1/numbers"#baiduapi
wz1=(0,0,0,0,0,'',0)
s=0
def yaapi():
    global wz1,wz2,url2
    f = open(r"ima\temp.jpg", 'rb')
    img = base64.b64encode(f.read())
    f.close()
    params = {"image": img}
    access_token = '24.44a761d1e59fe1ff2198575d7d58be9a.2592000.1653986576.282335-26139714'
    request_url = url2 + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    n = json.loads(response.text)
    print(n)
    if(list(n.keys())[0]!='words_result'):
        return
    if(n['words_result']!=[]):
        if (int(n['words_result_num']) == 1):
            x1 = int(n['words_result'][0]['location']['left'])
            w1 = int(n['words_result'][0]['location']['width'])
            y1 = int(n['words_result'][0]['location']['top'])
            h1 = int(n['words_result'][0]['location']['height'])
            if(swj):
                if (n['words_result'][0]['words'] == '0'):
                    cla1 = '0'
                    tag1 = 0
                    s.sendall(b'zero')
                elif (n['words_result'][0]['words'] == '1'):
                    cla1 = '1'
                    tag1 = 1
                    s.sendall(b'one')
                elif (n['words_result'][0]['words'] == '2'):
                    cla1 = '2'
                    tag1 = 2
                    s.sendall(b'two')
                elif (n['words_result'][0]['words'] == '3'):
                    cla1 = '3'
                    tag1 = 3
                    print(3)
                    s.sendall(b'three')
                elif (n['words_result'][0]['words'] == '4'):
                    cla1 = '4'
                    tag1 = 4
                    s.sendall(b'four')
                elif (n['words_result'][0]['words'] == '5'):
                    cla1 = '5'
                    tag1 = 5
                    s.sendall(b'five')
                elif (n['words_result'][0]['words'] == '6'):
                    cla1 = '6'
                    tag1 = 6
                    s.sendall(b'six')
                elif (n['words_result'][0]['words'] == '7'):
                    cla1 = '7'
                    tag1 = 7
                    s.sendall(b'seven')
                elif (n['words_result'][0]['words'] == '8'):
                    cla1 = '8'
                    tag1 = 8
                    s.sendall(b'eight')
                elif (n['words_result'][0]['words'] == '9'):
                    cla1 = '9'
                    tag1 = 9
                    s.sendall(b'nine')
                else:
                    cla1 = ''
                    tag1 = 10
            else:
                if (n['words_result'][0]['words'] == '0'):
                    cla1 = '0'
                    tag1 = 0
                elif (n['words_result'][0]['words'] == '1'):
                    cla1 = '1'
                    tag1 = 1
                elif (n['words_result'][0]['words'] == '2'):
                    cla1 = '2'
                    tag1 = 2
                    
                elif (n['words_result'][0]['words'] == '3'):
                    cla1 = '3'
                    tag1 = 3
                    print(3)
                    
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
                    tag1 = 10


            wz1 = (1, x1, w1, y1, h1, cla1, tag1)
    else:
        wz1 = (0, 0, 0, 0, 0, '', 0)
def CatchUsbVideo(window_name, camera_idx,lxsb):
    global wz1,wz2
    kt=0
    print("entersz")
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color1 = (c1, c2, c3)
    c1 = int(random.random() * 255)
    c2 = int(random.random() * 255)
    c3 = int(random.random() * 255)
    color2 = (c1, c2, c3)
    cap = cv2.VideoCapture(camera_idx)
    #cap = cv2.VideoCapture(camera_idx,cv2.CAP_DSHOW)
    #cv2.namedWindow(window_name, 0)
    while cap.isOpened():
        ok, frame = cap.read()
        print(ok)
        cv2.rectangle(frame, (0, 0), (640,480), color1,3)
        cv2.rectangle(frame, (0, 0), (320, 240), color1, 1)
        cv2.rectangle(frame, (320, 240), (640, 480), color1, 1)
        if (not ok):
            continue

        img_name = r'%s\temp.jpg' % (r"ima")
        cv2.imwrite(img_name, frame)
        kt=kt+1
        if(kt>k):
            kt=0
            yaapi()
            if(wz1[0]):
                cv2.rectangle(frame, (wz1[1], wz1[3]), (wz1[1] + wz1[2], wz1[3] + wz1[4]), color2, 2)
                cv2.putText(frame, 'sz:' + wz1[5], (wz1[1], wz1[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
                cv2.rectangle(frame, (wz1[1], wz1[3]), ((wz1[1]*2+wz1[2])//2, (wz1[3]*2+wz1[4])//2), color2, 1)
                cv2.rectangle(frame, ((wz1[1]*2+wz1[2])//2, (wz1[3]*2+wz1[4])//2), (wz1[1] + wz1[2], wz1[3] + wz1[4]), color2, 1)
                cv2.imshow(window_name, frame)
                boardkey = cv2.waitKey(1) & 0xFF
                if boardkey == 32:
                    break
                if (lxsb == 0):
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
                    break
            else:
                cv2.imshow(window_name, frame)
                boardkey = cv2.waitKey(1) & 0xFF
                if boardkey == 32:
                    break
        else:
            cv2.imshow(window_name, frame)
            boardkey = cv2.waitKey(1) & 0xFF
            if boardkey == 32:
                break

    cap.release()
    cv2.destroyAllWindows()

###########################################
class SMUploader:
    def __init__(self):
        self.session = requests.Session()
        self.api_key = 'wpDVGHpsDSfQpwMFwdpuLuT8WTYYQZHh'  # CHANGE HERE

    def upload_sm(self, file):
        url = 'https://sm.ms/api/v2/upload'
        headers = {}
        headers['Authorization'] = self.api_key
        files = {
            'smfile': open(file, 'rb')
        }

        res = self.session.post(url, headers=headers, files=files)
        if res.status_code > 300:
            return False, None
        else:
            json = res.json()
            code = json.get('code')

            if code == 'image_repeated':
                return True, json.get('images')
            elif code == 'success':
                return True, json['data']['url']

    def close(self):
        self.session.close()


# Push to wechat
class ServerChanPush:
    def __init__(self):
        self.url = 'https://sctapi.ftqq.com/SCT142173T4oHCdNmVu1ann1uLpYQzI7oz.send?title=messagetitle'  # CHANGE HERE

    def send_message(self, title, content):
        data = {
            'text': title,
            'desp': content
        }
        req = requests.post(url=self.url, data=data)
        print(req.text)


# 设置GPIO口为BCM编码方式
GPIO.setmode(GPIO.BCM)

# 忽略警告信息
GPIO.setwarnings(False)

# 记录舵机当前角度
angle = 90
angle1=40



# 电机引脚初始化为输出模式
# 按键引脚初始化为输入模式
# 寻迹引脚初始化为输入模式
def init():
    global pwm_ENA
    global pwm_ENB
    global pwm_servo
    global pwm_servo0
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

    GPIO.setup(key, GPIO.IN)
    GPIO.setup(TrackSensorLeftPin1, GPIO.IN)
    GPIO.setup(TrackSensorLeftPin2, GPIO.IN)
    GPIO.setup(TrackSensorRightPin1, GPIO.IN)
    GPIO.setup(TrackSensorRightPin2, GPIO.IN)
    GPIO.setup(EchoPin, GPIO.IN)
    GPIO.setup(TrigPin, GPIO.OUT)
    GPIO.setup(ServoPin, GPIO.OUT)
    GPIO.setup(ServoPin0, GPIO.OUT)
    # 设置pwm引脚和频率为2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    # 舵机频率50，每个周期20ms
    pwm_servo = GPIO.PWM(ServoPin, 50)
    pwm_servo0 = GPIO.PWM(ServoPin0, 50)
    pwm_ENA.start(0)
    pwm_ENB.start(0)


# 小车前进
def run(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车后退
def back(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车左转
def left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车右转
def right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车原地左转
def spin_left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车原地右转
def spin_right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)


# 小车停止
def brake():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


def servo_spin(next_angle):
    global angle
    global pwm_servo
    pwm_servo.start(0)  # 初始占空比
    print(angle, next_angle)
    if next_angle < angle:
        for i in reversed((next_angle, angle)):
            pwm_servo.ChangeDutyCycle(2.5 + 10 * i / 180)
            time.sleep(0.02)
    elif next_angle > angle:
        for i in (angle, next_angle):
            pwm_servo.ChangeDutyCycle(2.5 + 10 * i / 180)
            time.sleep(0.02)
    angle = next_angle
    time.sleep(1)  # 等待舵机转向指定角度
    pwm_servo.ChangeDutyCycle(0)  # 恢复初始占空比

def servo_spin1(next_angle):
    global angle1
    global pwm_servo0
    pwm_servo0.start(0)  # 初始占空比
    print(angle1, next_angle)
    if next_angle < angle1:
        for i in reversed((next_angle, angle1)):
            pwm_servo0.ChangeDutyCycle(2.5 + 10 * i / 180)
            time.sleep(0.02)
    elif next_angle > angle1:
        for i in (angle1, next_angle):
            pwm_servo0.ChangeDutyCycle(2.5 + 10 * i / 180)
            time.sleep(0.02)
    angle1 = next_angle
    time.sleep(1)  # 等待舵机转向指定角度
    pwm_servo0.ChangeDutyCycle(0)  # 恢复初始占空比

def yasocket():
    global data
    if(swj):
        while 1:
            data = s.recv(1024)
            data = data.decode()  # 第一参数默认utf8，第二参数默认strict
            print(data)
            pd(data)
    else:
        while 1:
            data=input("shuru")
            if(pd(data)):
                break
    print("t1 over")
yacenter=320
yahight =330
yaxw=30

speed = 10
yakp = 0.1
yaki = 0
yakd = 0.2

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
    back(yarunr, yarunl)
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
def yaxx1():
    f=int(input("fs:"))
    servo_spin(25)
    servo_spin1(0)#duoji
    cap = cv2.VideoCapture(-1)
    j=1
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        if(j<=5):
            j=j+1
            continue
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
            break
        if (f):
            center = white_index[0][white_count - 1] - yacenter - yaxw
        else:
            center = white_index[0][0] - yacenter + yaxw
        print(center)
        yarunlr0(center)
        cv2.waitKey(1)

    brake()
    cap.release()
    cv2.destroyAllWindows()
#dy lei 
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
            xmax=i[0]
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
    list1.append(xmin)
    list1.append(ymin)
    list1.append(xmax)
    list1.append(ymax)

    return list1
def yamx():
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(-1)
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
        yaframe = cv2.dilate(yaframe, None, iterations=4)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀

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
            if (area < 2000 and area <= 5000):
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

                # lentemp = lentempx // 3
                # roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
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
                    # yapreend = 1
                    # break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
def yaall1():
    servo_spin(100)
    servo_spin1(30)
    kcout=0#shibie jici
    yapreend=-1#shiebie result
    l=[0,0,0,0,0,0,0,0,0,0]
    cap = cv2.VideoCapture(-1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#huidu
        yaframe0 = gray.copy()
        ret, yaframe = cv2.threshold(yaframe0, 100, 255, 1)#erzhi
        yaframe = cv2.dilate(yaframe, None, iterations=3)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀
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
                # print(j,lentempx,lentempy)
                if (lentempx > lentempy):
                    lentemp = lentempx - lentempy
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((lentemp, lentemp), (0, 0)), 'constant', constant_values=0)
                    # print(1, roi.shape)
                    lentemp = lentempx // 4
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)
                    # print(2, roi.shape)
                    lentemp = lentempy // 4
                roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                cv2.imshow('sz', roi)
                # cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,255,255), 2)
                roi = cv2.resize(roi, (28, 28))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 784))
                y_train_pred = nn.predict(img)
                yapre = y_train_pred[0].item()
                kcout=kcout+1
                l[yapre]=l[yapre]+1
                if(kcout==20):
                    print(l)
                    yapre=l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    #s.sendall(bytes(str(yapre), encoding='utf-8'))
                    yapreend=1
                    break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
    #######################################################################################
    print("xxs")
    yapreend = yapre#shibie jieguo
    kcout = 0
    yapreend1 = -1
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if(yapreend%2):#ji left
        f = 0
    else:#ou right
        f = 1
    ######################################
    servo_spin(25)
    servo_spin1(0)#duoji
    j=1
    cap = cv2.VideoCapture(-1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        if(j<=5):
            j=j+1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # huidu
        yaframe0 = gray.copy()  # fuzhi
        retval, dst = cv2.threshold(yaframe0, 0, 255, cv2.THRESH_OTSU)  # er zhi
        dst = cv2.dilate(dst, None, iterations=6)  # 膨胀
        dst = cv2.erode(dst, None, iterations=3)  # 黑腐蚀
        cv2.imshow('xx', dst)
        cv2.imshow("yaxx1", dst[yahight-50:yahight+50, yacenter - 80:yacenter + 80])
        color = dst[yahight]
        white_count = np.sum(color == 0)
        white_index = np.where(color == 0)
        if white_count != 0:
            if (f):
                center = white_index[0][white_count - 1] - yacenter - yaxw
            else:
                center = white_index[0][0] - yacenter + yaxw
            #print(center)
            yarunlr0(center)
        else:
            brake()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, yaframe = cv2.threshold(gray, 100, 255, 1)
        yaframe = cv2.dilate(yaframe, None, iterations=3)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀
        gray1 = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(gray1, 30, 120)
        contours, hierarchy = cv2.findContours(binary,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        yapre = -1
        for i in range(len(contours)):
            # 筛掉面积过小的轮廓
            area = cv2.contourArea(contours[i])
            if (area < 5000 and area <= 10000):
                continue
           
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box)
            temp = findbox(box)
            # print(temp)
            if (temp[0] >= 0 and temp[1] >= 0 and temp[2] >= 0 and temp[3] >= 0):
                roi = yaframe[temp[1]:temp[3], temp[0]:temp[2]]
                lentempx = temp[2] - temp[0]
                lentempy = temp[3] - temp[1]
                # print(j,lentempx,lentempy)
                if (lentempx > lentempy):
                    lentemp = lentempx - lentempy
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((lentemp, lentemp), (0, 0)), 'constant', constant_values=0)
                    # print(1, roi.shape)
                    lentemp = lentempx // 4
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)
                    # print(2, roi.shape)
                    lentemp = lentempy // 4
                roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                # cv2.imshow('sz', roi)
                # cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,255,255), 2)
                roi = cv2.resize(roi, (28, 28))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 784))
                y_train_pred = nn.predict(img)
                yapre = y_train_pred[0].item()

                kcout = kcout + 1
                l[yapre] = l[yapre] + 1
                if (kcout == 3):
                    print(l)
                    
                    yapre = l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    #s.sendall(bytes(str(yapre), encoding='utf-8'))
                    if (yapre == yapreend):
                        yapreend1 = 1
                        break

        if (yapreend1>=0):
            brake()
            break
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    run(20, 20)
    time.sleep(0.2)
    brake()
    print("allover")
def yaall0():
    servo_spin(100)
    servo_spin1(30)
    kcout=0#shibie jici
    yapreend=-1#shiebie result
    l=[0,0,0,0,0,0,0,0,0,0]
    cap = cv2.VideoCapture(-1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#huidu
        yaframe0 = gray.copy()
        ret, yaframe = cv2.threshold(yaframe0, 100, 255, 1)#erzhi
        yaframe = cv2.dilate(yaframe, None, iterations=3)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀
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
                # print(j,lentempx,lentempy)
                if (lentempx > lentempy):
                    lentemp = lentempx - lentempy
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((lentemp, lentemp), (0, 0)), 'constant', constant_values=0)
                    # print(1, roi.shape)
                    lentemp = lentempx // 4
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)
                    # print(2, roi.shape)
                    lentemp = lentempy // 4
                roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                cv2.imshow('sz', roi)
                # cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,255,255), 2)
                roi = cv2.resize(roi, (28, 28))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 784))
                y_train_pred = nn.predict(img)
                yapre = y_train_pred[0].item()
                kcout=kcout+1
                l[yapre]=l[yapre]+1
                if(kcout==20):
                    print(l)
                    yapre=l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    s.sendall(bytes(str(yapre), encoding='utf-8'))
                    yapreend=1
                    break
            cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 5)
        cv2.imshow('origin with contours', draw_img)
        cv2.waitKey(1)
        if (yapreend >= 0):
            break
    cap.release()
    cv2.destroyAllWindows()
    #######################################################################################
    print("xxs")
    yapreend = yapre#shibie jieguo
    kcout = 0
    yapreend1 = -1
    l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if(yapreend%2):#ji left
        f = 0
    else:#ou right
        f = 1
    ######################################
    servo_spin(25)
    servo_spin1(0)#duoji
    j=1
    cap = cv2.VideoCapture(-1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        while (not ret):
            print('no')
        if(j<=5):
            j=j+1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # huidu
        yaframe0 = gray.copy()  # fuzhi
        retval, dst = cv2.threshold(yaframe0, 0, 255, cv2.THRESH_OTSU)  # er zhi
        dst = cv2.dilate(dst, None, iterations=6)  # 膨胀
        dst = cv2.erode(dst, None, iterations=3)  # 黑腐蚀
        cv2.imshow('xx', dst)
        cv2.imshow("yaxx1", dst[yahight-50:yahight+50, yacenter - 80:yacenter + 80])
        color = dst[yahight]
        white_count = np.sum(color == 0)
        white_index = np.where(color == 0)
        if white_count != 0:
            if (f):
                center = white_index[0][white_count - 1] - yacenter - yaxw
            else:
                center = white_index[0][0] - yacenter + yaxw
            #print(center)
            yarunlr0(center)
        else:
            brake()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, yaframe = cv2.threshold(gray, 100, 255, 1)
        yaframe = cv2.dilate(yaframe, None, iterations=3)  # 膨胀
        yaframe = cv2.erode(yaframe, None, iterations=4)  # 黑腐蚀
        gray1 = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.Canny(gray1, 30, 120)
        contours, hierarchy = cv2.findContours(binary,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        yapre = -1
        for i in range(len(contours)):
            # 筛掉面积过小的轮廓
            area = cv2.contourArea(contours[i])
            if (area < 5000 and area <= 10000):
                continue
           
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(box)
            temp = findbox(box)
            # print(temp)
            if (temp[0] >= 0 and temp[1] >= 0 and temp[2] >= 0 and temp[3] >= 0):
                roi = yaframe[temp[1]:temp[3], temp[0]:temp[2]]
                lentempx = temp[2] - temp[0]
                lentempy = temp[3] - temp[1]
                # print(j,lentempx,lentempy)
                if (lentempx > lentempy):
                    lentemp = lentempx - lentempy
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((lentemp, lentemp), (0, 0)), 'constant', constant_values=0)
                    # print(1, roi.shape)
                    lentemp = lentempx // 4
                elif (lentempx < lentempy):
                    lentemp = lentempy - lentempx
                    lentemp = lentemp // 2
                    roi = np.pad(roi, ((0, 0), (lentemp, lentemp)), 'constant', constant_values=0)
                    # print(2, roi.shape)
                    lentemp = lentempy // 4
                roi = np.pad(roi, ((lentemp, lentemp), (lentemp, lentemp)), 'constant', constant_values=0)
                # cv2.imshow('sz', roi)
                # cv2.rectangle(frame, (temp[0], temp[1]), (temp[2], temp[3]), (255,255,255), 2)
                roi = cv2.resize(roi, (28, 28))
                img = np.array(roi).astype(np.float32)
                img = img.reshape((1, 784))
                y_train_pred = nn.predict(img)
                yapre = y_train_pred[0].item()

                kcout = kcout + 1
                l[yapre] = l[yapre] + 1
                if (kcout == 3):
                    print(l)
                    
                    yapre = l.index(max(l))
                    print(yapre)
                    # (str(y_train_pred[0])).encode('utf-8')
                    s.sendall(bytes(str(yapre), encoding='utf-8'))
                    if (yapre == yapreend):
                        yapreend1 = 1
                        break

        if (yapreend1>=0):
            brake()
            break
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    run(20, 20)
    time.sleep(0.2)
    brake()
    print("allover")

def pd(dt):
    global temp
    if (dt == w):
        temp = w
        print("w")
        run(40, 40)
    elif (dt == s1):
        temp = s1
        print("s")
        back(40, 40)
    elif (dt == a):
        temp = a
        print("a")
        spin_left(0, 40)
    elif (dt == d):
        temp = d
        print("d")
        spin_right(30, 0)
    elif (dt == stop):
        temp = stop
        print("stop")
        brake()
        
    else:
        if(dt == all):
            print("all")
            yaall0()
        elif (dt == xx):
            print("xxs")
            yaxx1()
        elif (dt == sz):
            print("szapi")
            CatchUsbVideo("ya", -1, 1)
        elif (dt == ewm):
            print("szlo")
            yamx()
        return 1
    return 0

if __name__ == "__main__":
    init()
    servo_spin(30)#left
    servo_spin1(75)#up
    
    #csh moxin
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
    nn.loadya()#load mx w1.npy w2.npy
    
    if(swj):
        print("kslj")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        print("ljcg")
    
   
    while(1):
        data = s.recv(1024)
        data = data.decode()  # 第一参数默认utf8，第二参数默认strict
        pd(data)
        
            
       



