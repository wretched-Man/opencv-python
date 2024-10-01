# Merge images and feature maps from different places
# load all the images
# copy them to a new folder renamed as P0000_number... P0001_number
# save this list
import os
import shutil

# where the images are
mypath = "./images/"

# prefix before each name, must be unique
person_prefix = "P0002_"

# Read in the images
f = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(mypath)\
    for f in filenames]

# number of images ...
total_size = len(str(len(f)))

write_location = "../complete/images/"

for counted, image in enumerate(f):
    renamed = person_prefix + str(counted + 1).rjust(total_size, '0') + '.' + f[counted].split(".")[-1]
    shutil.copy2(image, write_location + renamed)

