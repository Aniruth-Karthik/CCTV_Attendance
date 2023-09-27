#import pafy
import cv2
import os

from ultralytics import YOLO
from collections import defaultdict

from deepface import DeepFace
import matplotlib.pyplot as plt
from math import atan2, degrees

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#url = "https://www.youtube.com/watch?v=6r3TiKlaOgk"
#url = "https://www.youtube.com/watch?v=pgk-719mTxM"
#video = pafy.new(url)
#best = video.getbest(preftype="mp4")

border = 40 #image cropping

cap = cv2.VideoCapture("D:\Proglint\Final\production_id_5198159 (2160p).mp4")

model = YOLO("D:\Proglint\Final\yolov8n-face.pt")

track_history = defaultdict(lambda: [])
foundFaces = []
def find_face(img, trackId):
    global foundFaces
    path = f"{trackId}.png"
    cv2.imwrite(path, img)
    #plt.imshow(img[:,:,::-1])
    #plt.show()
    m = 0.25
    f = "UNK"
    for face in os.listdir("register"):
        try:
            prob = DeepFace.verify(img1_path = "register\\"+face, img2_path = path)['distance']
            print(face[:-4],':',prob)
            if prob < m and (face[:-4]not in foundFaces):
                m = prob
                f = face[:-4]
        except:
            pass
    print("final:",f)
    if f!="UNK":foundFaces.append(f)
    return f

def faceDown(img, centerX, centerY):
    pitch = degrees(atan2(centerY - img.shape[0] / 2, 80))
    print(pitch)
    if abs(pitch)<56:
        return True
    return False
    
frame = 0
while True:
    success, img = cap.read()
    img1 = img.copy()
    frame += 1
    results = model.track(img, persist=True, tracker="botsort.yaml")

    for r in results:
        boxes = r.boxes
        for ind in range(len(boxes)):
            box = boxes[ind]
            trackId = int(boxes.id[ind])
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            centerY, centerX, _, _ = box.xywh[0]
            
            if faceDown(img, centerX, centerY):
                print("Face is DOWN")
            
            track = track_history[trackId]
            if track==[]:
                im = img1[y1-border:y2+border, x1-border:x2+border].copy()
                track.append(im)
                track.append(find_face(im, trackId))
            elif track[1]=="UNK":
                im = img1[y1-border:y2+border, x1-border:x2+border].copy()
                track[0] = im
                track[1] = find_face(im, trackId)
            track.append(frame)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.putText(img, track[1], [x1,y1], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            
    cv2.imshow('Webcam', cv2.resize(img, (768, 440)))
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#ATTENDANC
for student in track_history.values():
    plt.imshow(student[0])
    print("name:",student[1])
    att = len(student[2:])/frame
    print("enter frame:",student[2])
    print("attendance:",att*100)
    if att<0.75:
        print("ABSENT")
    else:
        print("PRESENT")