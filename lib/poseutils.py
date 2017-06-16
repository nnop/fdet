import os.path as osp
from xml.etree import ElementTree

def image_to_xml_path(image_path):
    _, image_fn = osp.split(image_path)
    upper_dir = osp.split(_)[0]
    name = osp.splitext(image_fn)[0]
    xml_dir = osp.join(upper_dir, 'xml')
    xml_path = osp.join(xml_dir, name + '.xml')
    return xml_path

def image_to_json_path(image_path, data_root):
    """
    image_path should be a relative path under the data dir
    """
    _, image_fn = osp.split(image_path)
    upper_dir = osp.split(_)[0]
    rel_dir = osp.relpath(upper_dir, data_root)
    if len(rel_dir.split('/')) == 1:
        rel_dir = ''
    else:
        rel_dir = '/'.join(rel_dir.split('/')[1:])
    name = osp.splitext(image_fn)[0]
    json_path = osp.join(data_root, 'json', rel_dir, name + '.json')
    return json_path

def load_image_info(xml_path, class_names):
    """
    class_names: a list contain class names
    """
    xml_obj = ElementTree.parse(xml_path)
    width = int(xml_obj.find('size').find('width').text)
    height = int(xml_obj.find('size').find('height').text)
    annotations = []
    objects = xml_obj.findall('object')
    for obj in objects:
        anno = {}
        name = obj.find('name').text
        cls_label = class_names.index(name)
        anno['aziLabel'] = cls_label
        anno['aziLabelFlip'] = cls_label
        bbobj = obj.find('bndbox')
        xmin = int(float(bbobj.find('xmin').text))
        xmax = int(float(bbobj.find('xmax').text))
        ymin = int(float(bbobj.find('ymin').text))
        ymax = int(float(bbobj.find('ymax').text))
        # NOTE bbox format [x1, y1, wid, hei]
        anno['bbox'] = [xmin, ymin, xmax-xmin+1, ymax-ymin+1]
        anno['category_id'] = 'person'
        anno['difficult'] = False
        anno['iscrowd'] = 0
        annotations.append(anno)
    # convert json
    image_info = {
        'image': { 'height': height, 'width': width },
        'annotation': annotations
    }
    return image_info
