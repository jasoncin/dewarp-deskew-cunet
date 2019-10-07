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

# import numpy as np
# import tensorflow as tf
#
# print("Hi there")
#
# pred = np.array([[31, 23,  4, 24, 27, 34],
#                 [18,  3, 25,  0,  6, 35],
#                 [28, 14, 33, 22, 20,  8],
#                 [13, 30, 21, 19,  7,  9],
#                 [16,  1, 26, 32,  2, 29],
#                 [17, 12,  5, 11, 10, 15]])
#
# print("Hello")
# print(tf.argmax(pred, 1))