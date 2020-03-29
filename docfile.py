import os

path="/home/hoangntbn/Desktop/20192/project2/20news-bydate/20news-bydate-train"

# print(os.listdir(path))



# import re 
# string = "    4534341234aaaabc12:  " 
# print(re.sub(r'[^a-z]', '', string) )


FJoin = os.path.join
dirs = [FJoin(path, f) for f in os.listdir(path)]
print(len(dirs))

# luu cac file bang mang 2 chieu
path_file=[]
for d in dirs:
    files=[FJoin(d,f) for f in os.listdir(d)]
    # print(files)
    path_file.append(files)
print(len(path_file))
