from xml.dom.minidom import parse

import os,shutil
import argparse
import cv2

from get_files_path import *

def readXML(data_dir):
  
  file_list = []
  dir_list = []
  get_file_path(data_dir + 'Annotations/', file_list, dir_list)

  for file in file_list:
    name = file.rsplit('/', 1)[1]
    name = name.rsplit('.', 1)[0]
    if len(name) < 4:
      continue
    print(name)
    img = cv2.imread(data_dir + 'JPEGImages/' + name + '.jpg')

    domTree = parse(file)
    rootNode = domTree.documentElement

    objects = rootNode.getElementsByTagName('object')
    for obj in objects:
      xmin = int(obj.getElementsByTagName('xmin')[0].firstChild.data)
      xmax = int(obj.getElementsByTagName('xmax')[0].firstChild.data)
      ymin = int(obj.getElementsByTagName('ymin')[0].firstChild.data)
      ymax = int(obj.getElementsByTagName('ymax')[0].firstChild.data)
      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    cv2.imshow('jj', img)
    key = cv2.waitKey()
    if(key == 114):
      shutil.move(file, data_dir + "bak/" + name + '.xml')
      shutil.move(data_dir + 'JPEGImages/' + name + '.jpg', data_dir + "bak/" + name + '.jpg')

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--datas_path', default='VOC2007')
    return parser.parse_args()

if __name__ == '__main__':
    readXML(parse_args().datas_path + '/')
