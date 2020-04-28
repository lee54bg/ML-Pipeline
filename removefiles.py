import os

print("Initiating the loop")

for filename in os.listdir("."):
    # if filename.endswith(".asm") or filename.endswith(".py"): 
    if filename == "dataset":
        print("Dataset directory exists")
    else:
        print(filename)

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