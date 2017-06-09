from os import path

_image_ext = '.jpg'

def xml_to_image_path(xml_path):
    # get ds
    up_dir, fn = path.split(xml_path)
    name = path.splitext(fn)[0]
    ds_dir = path.split(up_dir)[0]
    # get image dir
    image_dir = path.join(path.split(up_dir)[0], 'Frame')
    # get image path
    image_path = path.join(image_dir, name+_image_ext)
    return image_path

def image_to_xml_path(image_path):
    # get ds
    up_dir, fn = path.split(image_path)
    name = path.splitext(fn)[0]
    ds_dir = path.split(up_dir)[0]
    # get image dir
    xml_dir = path.join(path.split(up_dir)[0], 'xml')
    # get image path
    xml_path = path.join(xml_dir, name+'.xml')
    return xml_path

def sample_id_to_path(data_root, sample_id):
    image_dir = path.join(data_root, 'Frame')
    xml_dir = path.join(data_root, 'xml')
    assert path.isdir(image_dir), image_dir
    assert path.isdir(xml_dir), xml_dir
    image_path = path.join(image_dir, sample_id+_image_ext)
    xml_path = path.join(xml_dir, sample_id+'.xml')
    return image_path, xml_path
