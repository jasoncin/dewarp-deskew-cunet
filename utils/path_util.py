import os
import glob 
from random import shuffle
import tensorflow as tf

def read_image_list(pathToList, prefix=None):
    '''
    :param pathToList:
    :return:
    '''
    f = open(pathToList, 'r')
    filenames = []
    for line in f:
        if line[0] != '#':
            if line[-1] == '\n':
                filename = line[:-1]
            else:
                filename = line
            if prefix is not None:
                filename = prefix + filename
            filenames.append(filename)
        else:
            if len(line) < 3:
                break
    f.close()
    return filenames

def write_file_list(path_to_files, val_ratio=0.2, shuffled=True):
    '''
    :param path_to_files: folder that contains image files 
    :param val_ratio: the ratio of files belong to validation set 
    :return:
    '''

    path_to_files = os.path.abspath(path_to_files)
    all_files = glob.glob(os.path.join(path_to_files, '*.png'))
    dir_file, file_name = os.path.split(path_to_files)
    
    if shuffled:
        shuffle(all_files)
    num_files = len(all_files)

    with open(os.path.join(dir_file, 'training_data.lst'), 'w') as train_fp:
        for tr_file in all_files[: int((1-val_ratio)*num_files)]:
            print(tr_file)
            train_fp.write('{}\n'.format(tr_file))

    with open(os.path.join(dir_file, 'validation_data.lst'), 'w') as val_fp:
        for val_file in all_files[: int(val_ratio * num_files)]:
            print(val_file)
            val_fp.write('{}\n'.format(val_file))


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == "__main__":
    dir_path = 'train_data/images'
    write_file_list(dir_path)