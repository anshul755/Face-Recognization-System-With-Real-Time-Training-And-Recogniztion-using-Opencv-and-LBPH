import cv2
import numpy as np
import os
from PIL import Image


def addFace(fid,name):
    video=cv2.VideoCapture(0)
    video.set(5,640)
    video.set(6,480)

    faces=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    facedir='dataset'
    facename=name
    path=os.path.join(facedir,facename)
    
    if not os.path.isdir(path):
        os.makedirs(path,exist_ok=True)

    c=0
    while True:
        _,f=video.read()
        img=cv2.flip(f,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        multifaces=faces.detectMultiScale(gray,1.5,5)

        for(x,y,w,h) in multifaces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            text = f'x: {x} y: {y}'
            cv2.putText(f,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),3)
            c+=1

            cv2.imwrite("{}/{}.{}.{}{}".format(path,name,fid,c,".jpg"),gray[y:y+h,x:x+w])
            cv2.imshow('Image',img)

        k=cv2.waitKey(10)

        if k==27:
            break
        elif c>=40:
            break

    video.release()
    cv2.destroyAllWindows()


def train():
    database='dataset'
    img_dir=[a[0] for a in os.walk(database)][1::]
    
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faceSample=[]
    ids=[]

    for path in img_dir:
        path=str(path)
        ipaths=[os.path.join(path,f) for f in os.listdir(path)]

        for ipath in ipaths:
            Pilimg=Image.open(ipath).convert('L')
            img_numpy=np.array(Pilimg,'uint8')

            id=int(os.path.split(ipath)[-1].split('.')[1])
            faces=detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSample.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

    recognizer.train(faceSample,np.array(ids))
    recognizer.write('trainer.yml')
    print('\n {0} faces trained,Exiting Program'.format(len(np.unique(ids))))
    return len(np.unique(ids))

def recognizer(names):
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    cascadepath="haarcascade_frontalface_default.xml"
    facecascade=cv2.CascadeClassifier(cascadepath)

    font=cv2.FONT_HERSHEY_COMPLEX

    id=0
    name=""
    face_count=0

    cam=cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)

    minw=0.1*cam.get(3)
    minh=0.1*cam.get(4)

    while True:
        _,f=cam.read()
        img=cv2.flip(f,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces=facecascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(int(minw),int(minh)))

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            id,confidence=recognizer.predict(gray[y:y+h,x:x+h])
            if confidence<70:
                id=names[id]
            else:
                id='unknown'
                roll=None
                confidence="{0}%".format(round(100-confidence))

            if name==id:
                face_count+=1

                if(face_count>21):
                    face_count=-100
            else:
                name=id
                face_count=0
            
            cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(img,str(confidence),(x+5,y+h-5),font,1,(255,255,255),2)

        cv2.imshow("Face Detector",img)
        k=cv2.waitKey(10)
        if k==27:
            break

    cam.release()
    cv2.destroyAllWindows()


def use():
    while True:
        x=input("Enter 1: Add a Face, 2: Recognize a face, 3: Train Your Face, 4: Exit\n")

        if(x==str(1)):
            id=input("Enter id:")
            name=input("Enter name:")
            addFace(id,name)
        elif(x==str(2)):
            recognizer({1:input("Enter name you want to recognize:")})
        elif(x==str(3)):
            train()
        elif(x==str(4)):
            exit()
        else:
            print("Invalid Input")

use()