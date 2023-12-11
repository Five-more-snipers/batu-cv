import os
import cv2
import numpy as np

face_csd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list = []
class_id = []

train_path = 'images/train'
pName = os.listdir(train_path)

for i, nm in enumerate(pName):
    full_path = train_path+'/'+nm

    for ign in os.listdir(full_path):
        ig_fp = full_path +'/'+ ign
        img = cv2.imread(ig_fp,0)

        det_face = face_csd.detectMultiScale(img,scaleFactor=1.2, minNeighbors=5)

        if len(det_face) < 1:
            continue

        for fr in det_face:
            x,y,h,w = fr
            face_img = img[y:y+h, x:x+w]

            face_list.append(face_img)
            class_id.append(i)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_id))

#Test

test_path = 'images/test'

for lin in os.listdir(test_path):
    fip = test_path+'/'+lin
    img_gry = cv2.imread(fip,0)
    img_bgr = cv2.imread(fip)

    det_face = face_csd.detectMultiScale(img_gry,
                                         scaleFactor=1.2,
                                         minNeighbors=5)
    if len(det_face) < 1:
        continue

    for fr in det_face:
        x,y,h,w = fr
        face_img = img_gry[y:y+h,x:x+w]

        res, conf = face_recognizer.predict(face_img)

        cv2.rectangle(img_bgr,(x,y),
                      (x+w,y+h),(255,0,0,),1)
        text = pName[res]+' : '+str(conf)
        cv2.putText(img_bgr,text,(x,y-10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,(0,255,0),2)
        cv2.imshow('Result: ',  img_bgr)
        cv2.waitKey(0)
