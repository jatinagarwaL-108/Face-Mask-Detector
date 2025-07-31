import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("face_detect_model1.h5")

haar = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")
def detect_face(img):
    coods=haar.detectMultiScale(img)
    return coods


def detect_image(img):
    y_pred=model.predict(img.reshape(1,224,224,3))
    return y_pred[0][0]

def draw_label(img,text,pos,bg_color):
    text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)

    end_x=pos[0]+text_size[0][0]+2
    end_y=pos[0]+text_size[0][1]-2
    cv2.rectangle(img,pos,(end_x,end_y),color=bg_color,thickness=-1)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    img= cv2.resize(frame,(224,224))

    y_pred=detect_image(img)
    coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for x, y, w, h in coods:
        cv2.rectangle(frame, (x, y), (x + w, w + h), color=(255, 0, 0), thickness=3)
    if y_pred<10e-3:
        draw_label(frame,"with_Mask",(30,30),(0,255,0))
    else:
        draw_label(frame,"without_mask",(30,30),(0,0,255))

    cv2.imshow("Window",frame)
    if cv2.waitKey(1) & 0xFF==ord('x'):
        break

cv2.destroyAllWindows()
