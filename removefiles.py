import os
import shutil

for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name == "dataset":
            print("Deleting " + os.path.join(root, name))
            shutil.rmtree(os.path.join(root, name))
        # if name == "main":
        #     print("Deleting " + os.path.join(root, name))
        #     shutil.rmtree(os.path.join(root, name))