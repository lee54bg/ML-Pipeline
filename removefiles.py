import os

print("Initiating the loop")

for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name == "dataset":
            print("Dataset dir exists")
            print(os.path.join(root, name))

            os.rmdir(os.path.join(root, name))