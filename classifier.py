
import numpy as np
import sklearn.svm
import matplotlib.pyplot as plt
import sklearn.metrics 
from my_func import import_data, saveVar ,readvar, extract_features


##### import all data from face dataset ####
print('\t1-import all train and test data')


[img_face,img_nonface]=import_data()



#### select pos/neg train/test/validation data randomly #####
np.random.shuffle(img_face)
train_pos=img_face[0:10000]
validation_pos=img_face[10000:11000]
test_pos=img_face[11000:12000]

np.random.shuffle(img_nonface)
train_neg=img_nonface[0:10000]
validation_neg=img_nonface[10000:11000]
test_neg=img_nonface[11000:12000]





####### extract feature vectors from train data ###### 

print("\t2-extracting feature vector")
cell_=6
block_=3

   
feature_train=extract_features(train_pos, [] ,label=1, cell_size=cell_, block_size=block_)
feature_train=extract_features(train_neg, feature_train ,label=0, cell_size=cell_, block_size=block_)
X_train=[]
Y_train=[]
for row in feature_train:
    X_train.append(row[1])
    Y_train.append(row[0])       
X_train=np.array(X_train)


##### train SVM with feature vectors ######
print("\t3-train SVM classifier")
kernel_my='poly'

classifier=sklearn.svm.SVC(C=1,kernel=kernel_my)
classifier.fit(X_train,Y_train);
saveVar(classifier,'classifier')


##### calculate final precision with test data ######





feature_test=extract_features(test_pos, [] ,label=1, cell_size=cell_, block_size=block_)
feature_test=extract_features(test_neg, feature_test ,label=0, cell_size=cell_, block_size=block_)

    

X_test=[]
Y_test=[]
for row in feature_test:
    X_test.append(row[1])   
    Y_test.append(row[0])       
X_test=np.array(X_test)

cnt=0
for row in feature_test:
    tmp=classifier.predict(row[1].flatten().reshape(1, -1)).tolist()
    if tmp[0]==row[0]:
        cnt+=1
 
precision_=cnt/len(feature_test)*100





###### calc ROC and AP ######

print("\t4-calculae ROC and AP")

y_score = classifier.decision_function(X_test)

AP = sklearn.metrics.average_precision_score(Y_test, y_score)
precision, recall, _ = sklearn.metrics.precision_recall_curve(Y_test,y_score)

false_pos, true_pos, _ = sklearn.metrics.roc_curve(Y_test, y_score)
ROC = sklearn.metrics.auc(false_pos, true_pos)

######  draw curves #####

fig=plt.figure()
plt.plot(false_pos, true_pos, color='darkorange',label='ROC curve (area = %0.6f)' % ROC)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.01]); plt.ylim([-0.01, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
fig.savefig('res1.jpg', dpi=4*fig.dpi)
plt.close(fig)


fig=plt.figure()
plt.step(recall, precision, linewidth=0.8, where='post',label=' (AP = %0.6f)' % AP)
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.ylim([0.0, 1.05]); plt.xlim([-0.01, 1.01])
plt.title('precision-recall curve')
plt.legend(loc="lower right")
fig.savefig('res2.jpg', dpi=4*fig.dpi)
plt.close(fig)



print('\n\nfinal precision is : ' ,round(precision_,3) , ' %')






