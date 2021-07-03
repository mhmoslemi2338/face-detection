import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from my_func import  readvar , NMS , sliding_window ,show



def get_classifier():
    try:
        classifier=readvar('classifier')
    except:
        print('we should train classifier first: ')
        with open('classifier.py',"r") as rnf:
            exec(rnf.read())
            classifier=readvar('classifier')
    return classifier   
    
  
def FaceDetector(name,scale=1.1,th1=0.85,th2=0.2,repeat=12,method=1):
    classifier=get_classifier()
    try:
        img1_color=cv2.imread(name+".jpg",cv2.IMREAD_COLOR)
        img1=cv2.cvtColor(img1_color,cv2.COLOR_BGR2GRAY)
    except:
        print("the image not found")
        return
    print("1-run sliding window")
    windows=sliding_window(img1,repeat)
    feature=[]
    
    print("2-run feature extracting")
    for row in windows:
        resized_img = resize(row[0], (64,64))
        fd = hog(resized_img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3))
        feature.append(fd)
    
    X=[]
    for row in feature:
        X.append(row)
    X=np.array(X)
    print("3-scoring windows")
    print(" ")
    score=classifier.decision_function(X)
    
    boxes=[]
    for i in range(len(score)):
        if score[i]>th1:
            cnt=windows[i][2]
            (x,y)=windows[i][1]
            step=int(64*scale**cnt)
            boxes.append([x,y,step])
    boxes=np.array(boxes)        
    boxes_NMS=NMS(boxes,th2,method)
    
    
    im2=img1_color.copy()
    for row in boxes_NMS:
        [x,y,step]=row
        im2=cv2.rectangle(im2,(x,y),(int(x+step),int(y+step)),(0,255,0),2)
        
    return [im2,boxes,score]



    
[Melli,boxes1,score1]=FaceDetector('Melli',th1=0.75,th2=0.15)
[Persepolis,boxes2,score2]=FaceDetector('Persepolis',th1=0.65,th2=0.1)
[Esteghlal,boxes3,score3]=FaceDetector('Esteghlal',th1=0.47,th2=0.3,repeat=15,method=2)
    
    
    
cv2.imwrite('res4.jpg',Melli)
cv2.imwrite('res5.jpg',Persepolis)
cv2.imwrite('res6.jpg',Esteghlal)
