from get_files_path import *

def saveXML(file, name, img_w, img_h, img_c, bbox_list):
    xml_file = open(file, 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>hand_detect</folder>\n')
    xml_file.write('    <filename>' + str(name) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(img_w) + '</width>\n')
    xml_file.write('        <height>' + str(img_h) + '</height>\n')
    xml_file.write('        <depth>' + str(img_c) + '</depth>\n')
    xml_file.write('    </size>\n')
    
    for bbox in bbox_list:
      xml_file.write('    <object>\n')
      xml_file.write('        <name>hand</name>\n')
      xml_file.write('        <pose>Unspecified</pose>\n')
      xml_file.write('        <truncated>0</truncated>\n')
      xml_file.write('        <difficult>0</difficult>\n')
      xml_file.write('        <bndbox>\n')
      xml_file.write('            <xmin>' + str(bbox[0]) + '</xmin>\n')
      xml_file.write('            <ymin>' + str(bbox[1]) + '</ymin>\n')
      xml_file.write('            <xmax>' + str(bbox[2]) + '</xmax>\n')
      xml_file.write('            <ymax>' + str(bbox[3]) + '</ymax>\n')
      xml_file.write('        </bndbox>\n')
      xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
    xml_file.close()
