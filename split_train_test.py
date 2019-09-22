import os

dir = 'data_text_combine'

list_file = []
for file in os.listdir(os.path.join(dir, "images")):
    list_file.append(os.path.join(dir, "images") + "/" + file)

number_train = int(len(list_file) * 0.8)

with open("train.lst", "w") as f:
    for i in range(number_train):
        f.write(list_file[i] + "\n")

with open("val.lst", "w") as f:
    for i in range(number_train, len(list_file), 1):
        f.write(list_file[i] + "\n")
