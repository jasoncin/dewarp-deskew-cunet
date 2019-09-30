import os

dir = 'data_text_combine/labels'

for file in os.listdir(dir):
    # if 'GT2' in file:
        os.rename(os.path.join(dir, file), os.path.join(dir, file.replace('_GT','_')))
    # os.rename(os.path.join(dir, file), os.path.join(dir, file.replace("_0.jpg", ".jpg")))
    # if 'GT2' not in file :
    #     os.remove(os.path.join(dir, file))