

import cv2
import numpy as np
import os , tarfile , pickle , shutil
from skimage.transform import resize
from skimage.feature import hog


def show(im4,height=400):
    (h4,w4)=np.shape(im4)[0:2]
    scale=height/h4
    dim=(int(scale * w4) , height)
    im4_resize=cv2.resize(im4.copy(),dim)
    cv2.imshow('tmp', im4_resize)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def saveVar(myvar,name):
    path=os.path.join(os.getcwd(),'variables')
    try:
        os.mkdir(path)
    except:
        pass    
    name='variables/'+name+'.pckl'  
    f = open(name, 'wb')
    pickle.dump(myvar, f)
    f.close()
    return


def readvar(name):
    name='variables/'+name+'.pckl'  
    f = open(name, 'rb')
    myvar = pickle.load(f)
    f.close()
    return myvar

def import_data():       
    ##### import all data from face dataset ####
    f=tarfile.open('lfw.tgz', 'r')
    path = os.path.join(os.getcwd(), 'data')
    try:
        os.mkdir(path)
    except:
        pass  
    for i,row in enumerate(f):
        f.extract(row,'data')
        
    path="data/lfw"
    names=os.listdir(path)
    img_face=[]
    for row in names:
        dir_=os.path.join(path,row)    
        img_path=[]
        for j in os.listdir(dir_):
            img_path.append(os.path.join(path,row,j))
        for j in img_path:
            img_=cv2.imread(j)
            img_=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
            img_face.append(img_)
    shutil.rmtree('data')
        
    ##### import all data from non face dataset ####
    f=tarfile.open('nonface.tgz', 'r')
    path = os.path.join(os.getcwd(), 'data')
    try:
        os.mkdir(path)
    except:
        pass  
    for i,row in enumerate(f):
        f.extract(row,'data')
    path="data/nonface"
    names=os.listdir(path)
    img_nonface=[]
    for i,row in enumerate(names):
        dir_=os.path.join(path,row)
        img_=cv2.imread(dir_)
        img_=cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        img_nonface.append(img_)
    shutil.rmtree('data')
    img_face=np.array(img_face,dtype=object)
    img_nonface=np.array(img_nonface,dtype=object)
    
    return [img_face,img_nonface]
    
    
def extract_features(data ,feature, label , cell_size , block_size):
    for row in data:
        row=np.uint8(row)
        resized_img = resize(row.copy(), (64,64))
        fd = hog(resized_img, orientations=9, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size))
        feature.append([label,fd])
    return feature
    
    
    

def NMS(boxes, th,method=1):
    result = []
    (x1,y1)=(boxes[:,0],boxes[:,1])
    (x2,y2)=(boxes[:,2]+x1,boxes[:,2]+y1)
    area = (x2 - x1 ) * (y2 - y1)
    index = np.argsort(y2)
    while (True):   
        if len(index) <=0:
            break
        i = index[len(index)-1]
        result.append(i)
        tmp = [len(index)-1]
        for row in range(len(index)):
            j = index[row]
            (x1_,y1_) =( max(x1[i], x1[j]),max(y1[i], y1[j]))
            (x2_,y2_) =( min(x2[i], x2[j]),min(y2[i], y2[j]))
            (w ,h) = (max(0, x2_ - x1_ ),max(0, y2_ - y1_ ))
            area_mutual = (w * h)
            m= area[i] 
            if method==2:
                m=area[j]
            if (area_mutual/m) > th:
                tmp.append(row)
        index = np.delete(index, tmp)
    return boxes[result]



def sliding_window(img1,repeat=12):
    img2=img1.copy()
    scale=1.1
    win_size,step=64,11
    windows,windows2,cnt=[],[],0
    while(cnt<repeat):  
        (h,w)=np.shape(img2)
        x,y=0,0
        while(True):
            if x+win_size>=w:
                x=0
                y+=step
            if y+win_size>=h:
                break
            tmp=np.uint8(img2[y:y+win_size,x:x+win_size])
            x+=step
            windows.append([tmp,(x,y),cnt])   
        cnt+=1
        win_size=int(64*scale**cnt)
        
    for row in windows:
        if np.mean(row[0])<=55 or np.sum(row[0])<180000 or np.mean(row[0])>=200:
            continue
        windows2.append(row)
    return windows2

    
    

