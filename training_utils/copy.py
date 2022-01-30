import os
import shutil

# copied the positive cases three times
path = 'images'
node_img = os.listdir(path)
for i in node_img:
    if i[0] == 'n':
        file = os.path.join(path, i)
        file1 = os.path.join(path, '1_' + i)
        file2 = os.path.join(path, '2_' + i)
        file3 = os.path.join(path, '3_' + i)
        shutil.copy(file, file1)
        shutil.copy(file, file2)
        shutil.copy(file, file3)