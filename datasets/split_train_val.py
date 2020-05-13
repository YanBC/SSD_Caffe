import os  
import random
import argparse
# import pathlib

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='path to source directory')
    p.add_argument('des', help='path to destination directory')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    srcDir = args.src
    desDir = args.des

    xmlfilepath = os.path.join(srcDir, 'Annotations')
    desDir = os.path.join(desDir, 'ImageSets/Main/')
    os.makedirs(desDir, exist_ok=True)
    ftrainvalPath = os.path.join(desDir, 'trainval.txt')
    ftestPath = os.path.join(desDir, 'test.txt')
    ftrainPath = os.path.join(desDir, 'train.txt')
    fvalPath = os.path.join(desDir, 'val.txt')

    # xmlfilepath=r'VOC2007/Annotations'                            
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
    ftrainval = open(ftrainvalPath, 'w')    
    ftest = open(ftestPath, 'w')    
    ftrain = open(ftrainPath, 'w')    
    fval = open(fvalPath, 'w')    
      
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
