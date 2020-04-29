import os  
import random   
  
xmlfilepath=r'VOC2007/Annotations'                            
trainval_percent=0.9
train_percent=0.9
total_xml = os.listdir(xmlfilepath)  
num=len(total_xml)    
list=range(num)    
tv=int(num*trainval_percent)    
tr=int(tv*train_percent)    
trainval= random.sample(list,tv)    
train=random.sample(trainval,tr)    
  
print("train and val size",tv)  
print("train size",tr)  
ftrainval = open('VOC2007/ImageSets/Main/trainval.txt', 'w')    
ftest = open('VOC2007/ImageSets/Main/test.txt', 'w')    
ftrain = open('VOC2007/ImageSets/Main/train.txt', 'w')    
fval = open('VOC2007/ImageSets/Main/val.txt', 'w')    
  
for i  in list:    
    name=total_xml[i][:-4]+'\n'    
    if i in trainval:    
        ftrainval.write(name)    
        if i in train:    
            ftrain.write(name)    
        else:    
            fval.write(name)    
    else:    
        ftest.write(name)    
    
ftrainval.close()    
ftrain.close()    
fval.close()    
ftest .close()  
