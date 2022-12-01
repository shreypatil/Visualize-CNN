import json
import shutil
import os

json_path = './imagenet_class_index.json'
src_path = '/ssd_scratch/cvit/shreya/Imagenet2012/Imagenet-orig/train/'
dest_path = '/ssd_scratch/cvit/shreya/Imagenet2012/Imagenet-smol/train/'

f = open(json_path, 'r')

data = json.load(f)


    
    

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                

for val in range(151, 251) :
    src_name = src_path + data[str(val)][0]
    
    dest_name = dest_path + data[str(val)][0]
    
    copytree(src_name, dest_name)