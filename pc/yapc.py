'''
File Name          : yapc.py
Author             : CUTEYAYA
Version            : V1.0.0
Created on         : 2022/04/29
'''
import threading as thr
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from pynput import keyboard
import socket
import serial
stats=''
yasocketswi=1 #usart socket按钮
yaall=1
model=0
usart='COM20'
url='172.20.10.2' #my ip socket
sendtemp=''
def yarev():
    while(1):
        data = conn.recv(1024)
        print("recv:", data)
        data = data.decode()
        stats.ui.textEdit_8.setPlaceholderText(data)

def usartsend(str):
    a1 = str+ "\n"
    serial.write((a1).encode("gbk"))
def on_press(key):
    '按下按键时执行。'
    global sendtemp
    if (key.char=='s'):
        if(sendtemp != 's'):
            print("s")
            if (yasocketswi):
                conn.sendall(b'00s')
            else:
                usartsend("*00s#")
            sendtemp = 's'
    elif (key.char=='w'):
        if (sendtemp != 'w'):
            print("w")
            if (yasocketswi):
                conn.sendall(b'00w')
            else:
                usartsend("*00w#")
            sendtemp = 'w'
    elif (key.char == 'd'):
        if (sendtemp != 'd'):
            print("d")
            if (yasocketswi):
                conn.sendall(b'00d')
            else:
                usartsend("*00d#")
            sendtemp = 'd'
    elif (key.char == 'a'):
        if (sendtemp != 'a'):
            print("a")
            if (yasocketswi):
                conn.sendall(b'00a')
            else:
                usartsend("*00a#")
            sendtemp = 'a'
def on_release(key):
    print("stop")
    global sendtemp
    if (yasocketswi):
        conn.sendall(b'sto')
    else:
        usartsend("*sto#")
    sendtemp = 'sto'
    if key == keyboard.Key.esc:
        return False
def ya1():
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

class Stats:
    def __init__(self):
        self.ui = QUiLoader().load("D:\\cat.ui")
        self.ui.pushButton.clicked.connect(self.start)
        self.ui.pushButton_3.clicked.connect(self.all)
        self.ui.pushButton_2.clicked.connect(self.xx)
        self.ui.pushButton_7.clicked.connect(self.sz)
        self.ui.pushButton_8.clicked.connect(self.ewm)
        self.ui.pushButton_4.clicked.connect(self.ccall)

        if (yasocketswi):
            self.ui.textEdit_7.setPlaceholderText("WIFI")
        else:
            self.ui.textEdit_7.setPlaceholderText("BL")
    def start(self):
        global yaall,model
        yaall = 0
        st = self.ui.lineEdit.text()  # 写
        st = int(st)
        model=st
        if (st == 1):
            self.ui.textEdit_5.setPlaceholderText("模式1运行中")
        elif (st == 2):
            self.ui.textEdit_6.setPlaceholderText("模式2运行中")
    def ccall(self):
        global yasocketswi
        if (yasocketswi == 0):
            yasocketswi = 1
        else:
            yasocketswi = 0
        if (yasocketswi):
            self.ui.textEdit_7.setPlaceholderText("WIFI")
        else:
            self.ui.textEdit_7.setPlaceholderText("BL")
    def all(self):
        if (yasocketswi):
            conn.sendall(b'all')
        else:
            usartsend("*all#")
        print("ok")
    def xx(self):
        if (yasocketswi):
            conn.sendall(b'0xx')
        else:
            usartsend("*0xx#")
        print("ok")
    def sz(self):
        if (yasocketswi):
            conn.sendall(b'0sz')
        else:
            usartsend("*0sz#")
        print("ok")
    def ewm(self):
        if (yasocketswi):
            conn.sendall(b'ewm')
        else:
            usartsend("*ewm#")
        print("ok")
def yaui():
    global stats
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
if __name__ == "__main__":
    threadLock = thr.Lock()
    threads = []
    thread1 = thr.Thread(target=yaui)
    thread2 = 0
    thread1.start()
    while(yaall):
        pass
    if(yasocketswi):
        HOST = url
        PORT = 520
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(5)
        conn, addr = s.accept()  # 接受连接
        yasocketover = 1
        print("socket open success")
        stats.ui.textEdit_9.setPlaceholderText("网络连接成功")
        if (model == 1):
            ya1()
        else:
            thread2 = thr.Thread(target=yarev())
            thread2.start()
        yaya1 = 0
    else:
        serial = serial.Serial(usart, 9600, timeout=2)  # /dev/ttyUSB0
        if serial.isOpen():
            yasocketover = 1
            print("usart open success")
            stats.ui.textEdit_9.setPlaceholderText("串口连接成功")
            yaya1 = 0
        if (model == 1):
            ya1()
        #thread2 = thr.Thread(target=yayolo())
        #thread2.start()

    threads.append(thread1)
    threads.append(thread2)
    for t in threads:
        t.join()
    print("Exiting")
