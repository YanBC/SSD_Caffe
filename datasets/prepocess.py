from xml.dom.minidom import parse

import cv2

from get_files_path import *

def align_class_name(rootNode):
    objects = rootNode.getElementsByTagName('object')
    for obj in objects:
      name = obj.getElementsByTagName('name')[0].firstChild.data
      index = name.find('hand')
      if index == -1:
        name = 'hand'
        obj.getElementsByTagName('name')[0].childNodes[0].data = name

def gene_flip_data(file, src_rootNode, name):
    img_path = file.replace('.xml', '.jpg')
    img_path = img_path.replace('Annotations', 'JPEGImages')
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.flip(img, -1)
    index = img_path.find('.jpg')
    img_path = list(img_path)
    img_path.insert(index, '_flip')
    img_path = "".join(img_path)
    print(img_path)
    cv2.imwrite(img_path, img)
    height = img.shape[0]
    width = img.shape[1]
    depth = img.shape[2]

    xml_path = str(file);
    index = xml_path.find('.xml')
    xml_path = list(xml_path)
    xml_path.insert(index, '_flip')
    xml_path = "".join(xml_path)
    xml_file = open(xml_path, 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>hand_detect</folder>\n')
    xml_file.write('    <filename>' + str(name) + '_flip.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>' + str(depth) + '</depth>\n')
    xml_file.write('    </size>\n')
    
    objects = src_rootNode.getElementsByTagName('object')
    for obj in objects:
      name = obj.getElementsByTagName('name')[0].firstChild.data
      xmin = int(obj.getElementsByTagName('xmin')[0].firstChild.data)
      xmax = int(obj.getElementsByTagName('xmax')[0].firstChild.data)
      ymin = int(obj.getElementsByTagName('ymin')[0].firstChild.data)
      ymax = int(obj.getElementsByTagName('ymax')[0].firstChild.data)
      n_xmin = width - xmax
      n_ymin = height - ymax
      n_xmax = width - xmin
      n_ymax = height - ymin

      xml_file.write('    <object>\n')
      xml_file.write('        <name>hand</name>\n')
      xml_file.write('        <pose>Unspecified</pose>\n')
      xml_file.write('        <truncated>0</truncated>\n')
      xml_file.write('        <difficult>0</difficult>\n')
      xml_file.write('        <bndbox>\n')
      xml_file.write('            <xmin>' + str(n_xmin) + '</xmin>\n')
      xml_file.write('            <ymin>' + str(n_ymin) + '</ymin>\n')
      xml_file.write('            <xmax>' + str(n_xmax) + '</xmax>\n')
      xml_file.write('            <ymax>' + str(n_ymax) + '</ymax>\n')
      xml_file.write('        </bndbox>\n')
      xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()

data_dir = 'VOC2007/'
file_list = []
dir_list = []
get_file_path(data_dir + 'Annotations/', file_list, dir_list)

for file in file_list:
  name = file.rsplit('/', 1)[1]
  name = name.rsplit('.', 1)[0]
  if len(name) < 3:
    continue
  print(file)
  domTree = parse(file)
  rootNode = domTree.documentElement
  align_class_name(rootNode)

  #gene_flip_data(file, rootNode, name)

  f = open(file, 'w')
  domTree.writexml(f)
  f.close()
