#Face recognition
import cv2
import pandas as pd
import face_recognition as fr
fname = 'features1.csv'
import numpy as np
try:
    df = pd.read_csv(fname)
except:
    print('Face Database not found. Halt')
else:
    fd = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    vid = cv2.VideoCapture(0)
    while True:
        ack, img = vid.read()
        if ack:
            faces = fd.detectMultiScale(img, 1.2,10,minSize=(150,150))
            if len(faces) == 1:
                x,y,w,h = faces[0]
                face_img = img[y:y+h,x:x+w,:].copy()
                face_enc = fr.face_encodings(face_img)
                if len(face_enc) == 1:
                    feats_data = df['enc'].apply(lambda x:eval(x)).values.tolist()
                    matches = fr.compare_faces(face_enc, np.array(feats_data))
                    if True in matches:
                        match_ind = matches.index(True)
                        name = df['name'][match_ind]
                    else:
                        name = 'Unknown'
                    cv2.putText(img, name,(150,150),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255),5
                    )                    

            cv2.imshow('Preview', img)  # depends on requirement
            key = cv2.waitKey(1)

            if key == ord('x'):
                break
    cv2.destroyAllWindows();
    
    vid.release()