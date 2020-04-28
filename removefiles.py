import os

print("Initiating the loop")

for root, dirs, files in os.walk(".", topdown=False):
    # for name in files:
    #    print(os.path.join(root, name))
    for name in dirs:
        # if name == "dataset":
        #     print("Dataset dir exists")
        print(os.path.join(root, name))
        print(name)

# directory_name="sampleenv"

# for d in */ ; do
#     echo "$d"
#     if [ -d $directory_name ]
#     then
#         echo "Directory already exists"
#     else
#         mkdir $directory_name
#     fi
# done