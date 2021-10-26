from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from tkinter import messagebox
import datetime
import sys
from tkinter import filedialog as fd
# import winsound
import playsound
from tkinter import ttk
import wmi
import cv2
from PIL import ImageTk, Image
from tkinter_custom_button import TkinterCustomButton
import os
import argparse
import json
from tinydb import TinyDB
from win32api import GetSystemMetrics
import time
import cv2
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import numpy as np
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from object_tracking.application_util import preprocessing
from object_tracking.deep_sort import nn_matching
from object_tracking.deep_sort.detection import Detection
from object_tracking.deep_sort.tracker import Tracker
from object_tracking.application_util import generate_detections as gdet
from utils.bbox import draw_box_with_id

import warnings
warnings.filterwarnings("ignore")

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
adl=['Select Camera']
ad={}
daf=TinyDB('cam_in.json')
def camerachecker(detclic):
    
    camnumber = -2
    list = []
    for i in range(8):
        detclic.set('Checking Camera')
        cap = cv2.VideoCapture(camnumber)
        ret, frame = cap.read()
        if(ret == 1):
            list.append(camnumber)
            cap.release()
        camnumber += 1
    if list:
        print('listofcamers'+str(list))
        detclic.set('Camera Found')
        detclic.set('Camer Found->'+str(list[0]))    
        return list[0]
    else:
        return 0
def mainpro():
    try:
            root=Tk()
            root.geometry('300x500')
            root.title('Helmet Detection')
            # p1=PhotoImage(file='hplogo.png')
            # root.iconphoto(False,p1)
            
            menubar=Menu(root)
            file = Menu(menubar,tearoff=0)
            file.add_command(label="change camera",command=lambda:mainpage(root))
            file.add_command(label='Quit',command=lambda:applicationquit(cap,root))
            menubar.add_cascade(label="File", menu=file)
            menubar.add_command(label='Help')
            menubar.add_command(label='About Us')
            clicked=tk.StringVar()
            clicked.set("Audio:English")
            detclic=StringVar()
            detclic.set("To Start Click 'Detect Employee'")
            camd=daf.get(doc_id=1)
            camc=camd['cam_name']
            print("cam name"+str(camc))
            cap1 = cv2.VideoCapture(camc)


            

            def chagetext():
                if clicked.get()=='Audio:English':
                    # queue1.deque()
                    clicked.set('Audio:Tamil')
                else:
                    
                    clicked.set("Audio:English")
                    

            img = ImageTk.PhotoImage(Image.open("homepagepic.png"))
            img2 = ImageTk.PhotoImage(Image.open("loadingimg.png"))
            panel = Label(root, image = img)
            panel.place(relx=.07,rely=.05)
            # panel2 = Label(root, image = img2)
            frameCnt = 10
            frames = [PhotoImage(file='loadinggif.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]
            frameCnt2 = 10
            frames2 = [PhotoImage(file='wearhelmet.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt2)]
            frameCnt3 = 10
            frames3 = [PhotoImage(file='welldone.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt3)]
            def update(ind):

                frame = frames[ind]
                ind += 1
                if ind == frameCnt:
                    ind = 0
                panel2.configure(image=frame)
                root.after(100, update, ind)
            def update1(ind):

                try:
                    frame = frames2[ind]
                    ind += 1
                    if ind == frameCnt2:
                            ind = 0
                    panel3.configure(image=frame)
                    root.after(100, update1, ind)
                except Exception as E:
                    print(E)
            def update2(ind):

                try:
                    frame = frames3[ind]
                    ind += 1
                    if ind == frameCnt3:
                            ind = 0
                    panel4.configure(image=frame)
                    root.after(100, update2, ind)
                except Exception as E:
                    print(E)        
            panel2=Label(root)
            panel3=Label(root)
            panel4=Label(root)
            
            lmain = Label(root)

            style = ttk.Style()
            style.configure("BW.TLabel", foreground="White", background="#2874a6")
            disnam=ttk.Label(root,textvariable=detclic,style="BW.TLabel",width=30,font=('TimesNewRoman',10))
            disnam.place(relx=.16,rely=.8)
            button_1 = TkinterCustomButton(text="Detect Employee", corner_radius=10,command=lambda:threading.Thread(target=detemp,args=(detclic,clicked,panel3,)).start())
            button_1.place(relx=.02,rely=.9)
            button_2 = TkinterCustomButton(text="Back", corner_radius=10,command=lambda:threading.Thread(target=backtomain,args=()).start())
            lang=tk.Button(textvariable=clicked,command=lambda:chagetext(),fg="#ffffff",bg="#2874a6")
            # lang.place(relx=.38,rely=.9)
            # lang=MyOptionMenu(root,"English","English","Tamil")
            lang.place(relx=.68,rely=.01)
            qut=TkinterCustomButton(text="Quit Applicaton", corner_radius=10,command=lambda:threading.Thread(target=applicationquit,args=()).start())
            qut.place(relx=.58,rely=.9)

            canvas = Canvas(root, width = 10, height = 10)      
            canvas.place()
            def applicationquit():
                    que.enque('Comeback')   
                    root.destroy()
                    cv2.destroyAllWindows()
                    
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # detclic.set('exiting...')
                    # if que.deque()=='exiting':
                    #         root.destroy()
                    # else:
                    #         que.enque('Comeback')
                            # applicationquit()
                    # root.destroy()
            def backtomain():
                    que.enque('Comeback')
                    root.geometry('300x500')
                    panel.place(relx=.07,rely=.1)
                    disnam.place(relx=.16,rely=.8)
                    button_1.place(relx=.02,rely=.9)
                    lang.place(relx=.68,rely=.01)
                    qut.place(relx=.58,rely=.9)
                    button_2.place_forget()
                    panel2.place_forget()
                    panel3.place_forget()
                    panel4.place_forget()
                    lmain.place_forget()
            def detemp(detclic,clicked,panel3):
                    width=GetSystemMetrics(0)
                    height=GetSystemMetrics(1)
                    geo=str(width)+'x'+str(height)+'+0+0'
                    root.geometry(geo)
                    panel.place(relx=.02,rely=.05)
                    disnam.place(relx=.04,rely=.8)
                    button_1.place_forget()
                    button_2.place(relx=.01,rely=.9)
                    # lang.place(relx=.38,rely=.9)
                    lang.place(relx=.18,rely=.01)
                    qut.place(relx=.15,rely=.9)
                    panel2.place(relx=.4,rely=.3)
                    root.after(0, update, 0)
                    lmain.place(relx=.28,rely=.01)
                    # detemp2(detclic,clicked,panel3)
                    threading.Thread(target=detemp2,args=(detclic,clicked,panel3,)).start() 
            def plays(audio):
                    t=5
                    que1.enque('Playing')
                    playsound.playsound(audio)
                    while t:
                            mins, secs = divmod(t, 60)
                            timer = '{:02d}:{:02d}'.format(mins, secs)
                            print(timer, end="\r")
                            time.sleep(1)
                            t -= 1
                    que1.deque()
            def detemp2(detclic,clicked,panel3):
                    try:
                            # camc=camerachecker()
                            # print('camc+++'+str(camc))
                            config_path = 'config.json'
                            # num_cam = int(camc+1)

                            with open(config_path) as config_buffer:
                                config = json.load(config_buffer)
                            net_h, net_w = 416, 416  
                            obj_thresh, nms_thresh = 0.5, 0.45




                            os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
                            infer_model = load_model(config['train']['saved_weights_name'])






                            max_cosine_distance = 0.3
                            nn_budget = None
                            nms_max_overlap = 1.0


                            model_filename = 'mars-small128.pb'
                            encoder = gdet.create_box_encoder(model_filename, batch_size=1)

                            metrics = []
                            trackers = []
                            # for i in range(num_cam):
                            #       if camc>0:
                            #               if i==0:
                            #                       continue
                            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
                            tracker = Tracker(metric)
                            trackers.append(tracker)







                            video_readers = []
                            # for i in range(num_cam):
                            #       if camc>0:
                            #               if i==0:
                            #                       continue
                            try:
                                    video_reader = cv2.VideoCapture(camc)
                            except Exception as E:
                                    messagebox.showerror(title='Camera Error',message='No Camera Module')
                                    que.enque('Comeback')
                            video_readers.append(video_reader)
                            print("dfafas"+str(len(video_readers)))

                            # if camc>0:
                            #       batch_size = camc
                            # else:
                            #       batch_size=camc+1       
                            # images = []
                            while True:
                                    try:
                                            ret_val,image=video_readers[0].read()               
                                            if ret_val:
                                                    batch_boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh,
                                                                 nms_thresh)

                                    
                                                    boxs = [[box1.xmin,box1.ymin,box1.xmax-box1.xmin, box1.ymax-box1.ymin] for box1 in batch_boxes[0]]
                                                    features = encoder(image, boxs)
                                                    detections = []
                                                    for j in range(len(boxs)):
                                                            label = batch_boxes[0][j].label
                                                            detections.append(Detection(boxs[j], batch_boxes[0][j].c, features[j],label))
                                                    trackers[0].predict()
                                                    trackers[0].update(detections)   
                                                    n_without_helmet = 0
                                                    n_with_helmet = 0
                                                    for track in trackers[0].tracks:
                                                            if not track.is_confirmed() or track.time_since_update > 1:
                                                                    continue
                                                            if track.label == 2:
                                                                    n_without_helmet += 1
                                                            if track.label == 1:
                                                                    n_with_helmet += 1
                                                            bbox = track.to_tlbr()
                                                            draw_box_with_id(image, bbox, track.track_id, track.label, config['model']['labels'])
                                                    print("CAM "+str(0))
                                                    print("Persons without helmet = " + str(n_without_helmet))
                                                    print("Persons with helmet = " + str(n_with_helmet))
                                                    cv2image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                                                    cv2image=cv2.resize(cv2image1,(900,700))
                                                    img = Image.fromarray(cv2image)
                                                    imgtk = ImageTk.PhotoImage(image=img)
                                                    lmain.imgtk = imgtk
                                                    lmain.configure(image=imgtk)
                                                    panel2.place_forget()
                                                    print('ntg.......')
                                                    if (n_without_helmet >0) and (n_with_helmet>0):
                                                        detclic.set('Someone is without helmet')
                                                        if clicked.get()=='Audio:Tamil':
                                                                if que1.deque()==None:
                                                                        # panel3.place(relx=.02,rely=.05)
                                                                        root.after(0, update1, 0)
                                                                        tamhel=threading.Thread(target=plays,args=('./audio/Tamil/tamilhelw_hel.mp3',))
                                                                        tamhel.start()
                                                                else:
                                                                        que1.enque('playing')
                                                        else:
                                                                    if que1.deque()==None:
                                                                            # panel3.place(relx=.02,rely=.05)
                                                                            root.after(0, update1, 0)
                                                                            enghel=threading.Thread(target=plays,args=('./audio/English/englishhelw_hel.mp3',))
                                                                            enghel.start()
                                                                    else:
                                                                            que1.enque('playing')
                                                    else:
                                                        if n_without_helmet>0:
                                                                detclic.set('Persons without Helmet Found')
                                                                if clicked.get()=='Audio:Tamil':
                                                                        if que1.deque()==None:
                                                                                # panel3.place(relx=.02,rely=.05)
                                                                                root.after(0, update1, 0)
                                                                                tamhel=threading.Thread(target=plays,args=('./audio/Tamil/tamilhel.mp3',))
                                                                                tamhel.start()
                                                                        else:
                                                                                que1.enque('playing')

                                                                else:
                                                                        if que1.deque()==None:
                                                                                # panel3.place(relx=.02,rely=.05)
                                                                                root.after(0, update1, 0)
                                                                                enghel=threading.Thread(target=plays,args=('./audio/English/englishhel.mp3',))
                                                                                enghel.start()
                                                                        else:
                                                                                que1.enque('playing')        
                                                        elif(n_with_helmet>0):
                                                                detclic.set('Persons with Helmet Found')
                                                                panel3.place(relx=1,rely=1)
                                                                if clicked.get()=='Audio:Tamil':
                                                                        if que1.deque()==None:
                                                                                # panel4.place(relx=0,rely=.05)
                                                                                root.after(0, update2, 0)
                                                                                tamhelw=threading.Thread(target=plays,args=('./audio/Tamil/tamilhelw.mp3',))
                                                                                tamhelw.start()
                                                                        else:
                                                                                que1.enque('playing')


                                                                else:
                                                                        if que1.deque()==None:
                                                                                # panel4.place(relx=0,rely=.05)
                                                                                root.after(0, update2, 0)
                                                                                enghelw=threading.Thread(target=plays,args=('./audio/English/englishhelw.mp3',))
                                                                                enghelw.start()
                                                                        else:
                                                                                que1.enque('playing')
                                                        else:
                                                                panel4.place(relx=1,rely=1)
                                                                panel3.place(relx=1,rely=1)
                                                                detclic.set('Wearing Helmet Is Mandatory')

                                                                # cv2.imshow('Cam'+str(i), images[i])
                                                                
                                                                # lmain.after(10, _main_)
                                                
                                            if que.deque()=='Comeback':
                                                    # que.enque('exiting')
                                                    print('entered')
                                                    break

                                                    

                                    except Exception as E:
                                            print("Line Number:",sys.exc_info()[-1].tb_lineno)
                                            if str(E) =='Too early to create image':
                                                    root.destroy()
                                                    sys.exit()
                                            if str(E) =='main thread is not in main loop':
                                                    root.destroy()
                                                    sys.exit()
                                            messagebox.showerror(title="Error-Wle",message="Error:\n"+str(E)+'LN='+str(sys.exc_info()[-1].tb_lineno))
                                            print(E)      
                                            
                                # for i in range(num_cam):
                                #   if camc>0:
                                #           if i==0:
                                #                   continue
                                #           print('sdsad'+str(i))   
                                #           ret_val, image = video_readers[i-1].read()
                                #           if ret_val == True: images += [image]
                                #           print('len(images)'+str(len(images)))
                                #   else:
                                #           print('sdsad25'+str(i)) 
                                #           ret_val, image = video_readers[i].read()
                                #           if ret_val == True: images += [image]   
                                #           print('len(images)'+str(len(images)))
                                
                                                       
                            
                    except Exception as E:
                            messagebox.showerror(title="Error-dte",message="Error:\n"+str(E)+'LN='+str(sys.exc_info()[-1].tb_lineno))
                            print(E)      
            class Queue(object):
                    def __init__(self):
                        super(Queue, self).__init__()
                        self.item = []
                        
                    def enque(self,add):
                        self.item.insert(0,add)
                        return True
                    def size(self):
                        return len(self.item)
                    def isempty(self):
                        if self.size()==0:
                            return True
                        else:
                            return False
                    def deque(self):
                        if self.size()==0:
                            return None
                        else:
                            return self.item.pop()                  
            class Queue1(object):
                    def __init__(self):
                        super(Queue1, self).__init__()
                        self.item = []
                        
                    def enque(self,add):
                        self.item.insert(0,add)
                        return True
                    def size(self):
                        return len(self.item)
                    def isempty(self):
                        if self.size()==0:
                            return True
                        else:
                            return False
                    def deque(self):
                        if self.size()==0:
                            return None
                        else:
                            return self.item.pop()
            que1=Queue1()
            que=Queue()                
            root.config(menu=menubar) 
            root.mainloop()
    except Exception as E:
            print(E)
            messagebox.showerror(title='Error',message='Mian Block Error: \n'+str(E))
            print(traceback.format_exc())


def storecam(a,croot,ul):
    croot.destroy()
    if len(daf)==0:
        if ul=='u':
            cam=ad[a]
            print('cam in'+str(cam))
            daf.insert({'cam_name':cam})
            mainpro()
        elif (ul=='l'):
            cam=a
            daf.insert({'cam_name',cam})
            mainpro()
    else:
        if ul=='u':
            cam=ad[a]
            print('cam in'+str(cam))
            daf.update({'cam_name':cam})
            mainpro()
        elif (ul=='l'):
            cam=a
            print('Lan cam'+str(cam))
            daf.update({'cam_name':cam})
            mainpro()
    
    

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 10 # Interval in ms to get the latest frame
        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=600, height=400)
        self.canvas.pack()
        # Update image on canvas
        window.after(self.interval, self.update_image)
        self.button = tk.Button()

    def update_image(self):    
        # Get the latest frame and convert image format
        self.OGimage = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        self.OGimage = Image.fromarray(self.OGimage) # to PIL format
        self.image = self.OGimage.resize((600, 400), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)
def list_cam1():
    
    count=0
    c = wmi.WMI()
    wql = "Select * From Win32_USBControllerDevice"
    for item in c.query(wql):
        q = item.Dependent.Caption
        if re.findall("Camera",q):
            if q in ad.keys():
                pass
            else:
                ad[q]=count
                adl.append(q)
                print(ad)
                count+=1
    UCAM2(ad)



def UCAM2(ad):
    root1=Tk()
    sBut=tk.Button(root1,text='YES',fg="#ffffff",bg="#2874a6",command=lambda:storecam(camname.get(),root1,'u'))
    nBut=tk.Button(root1,text='NO',fg="#ffffff",bg="#2874a6",command=lambda:UCAM(root1))
    def show(ad):
        cam=camname.get()
        if cam=='select Camera':
            pass
        else:
            MainWindow(root1, cv2.VideoCapture(ad[cam]))
            cbut.pack_forget()
            sBut.pack(fill=BOTH,expand=True)
            nBut.pack(fill=BOTH,expand=True)
    camname=StringVar()
    nob=len(ad)
    print(adl)
    drop = OptionMenu( root1 , camname , *adl )
    drop.pack()
    cbut = Button( root1 , text = "click Me" , command =lambda:show(ad))
    cbut.pack()
    label = Label(root1,text="")
    label.pack()
    root1.mainloop()    

def UCAM(froot):
    froot.destroy()
    print('enterd..........')
    list_cam1()
    print("WASS"+str(ad))
    print("sdad"+str(adl))
# rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov
def LCAM(froot):
    froot.destroy()
    root2=Tk()
    def printInput():
        inp = inputtxt.get(1.0, "end-1c")
        inp1=""+str(inp)
        lbl.config(text = "Provided Input: "+inp)
        sBut=tk.Button(root2,text='YES',fg="#ffffff",bg="#2874a6",command=lambda:storecam(inp,root2,'l'))
        nBut=tk.Button(root2,text='NO',fg="#ffffff",bg="#2874a6",command=lambda:LCAM(root2))
        if 'rtsp' in inp:
            lbl.config(text = "correct Int")
            MainWindow(root2, cv2.VideoCapture(inp))
            But2.pack_forget()
            sBut.pack(fill=BOTH,expand=True)
            nBut.pack(fill=BOTH,expand=True)

    Txt=Label(root2,text='Enter URL (rtsp)')
    Txt.pack()
    inputtxt = tk.Text(root2,
                   height = 1,
                   width = 50)
  
    inputtxt.pack()
    But2=Button(root2,text='open',command=lambda:printInput())
    But2.pack()
    lbl = Label(root2, text = "")
    lbl.pack()
    root2.mainloop()
    

def mainpage(root):
    root.destroy()
    camroot=Tk()
    camroot.title('Select CAM MODE')
    
    # camroot.resizable(0,0)
    but1=TkinterCustomButton(master=camroot,text='USB CAM',corner_radius=10,command=lambda:UCAM(camroot))
    but1.pack(side=LEFT,padx=10,expand=True,fill=BOTH)
    but2=TkinterCustomButton(master=camroot,text='LAN CAM',corner_radius=10,command=lambda:LCAM(camroot))
    but2.pack(side=RIGHT,padx=10,expand=True,fill=BOTH)
    but3=Button(camroot,text='BACK',command=lambda:LCAM(camroot))
    but3.pack(side=RIGHT,padx=10,expand=True,fill=BOTH)
    camroot.mainloop()


mainpro()

