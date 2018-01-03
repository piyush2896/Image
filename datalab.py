import os
from random import shuffle
from util import Config, combine_files_with_labels


def rename_files_to_ints(path):
    """
    Helper function to convert the image names to ints.
    @params
        path: Path to all images.
    """
    if path[-1] != '/':
        path += '/'
    filenames = os.listdir(path)
    for i, filename in enumerate(filenames):
        new_name = str(i)+'.'+filename.split(".")[-1]
        os.rename(path+filename, path+new_name)


def train_dev_test_split(path, target_dir, other_dir):
    """
    Split files to train, dev and test, set.
    @params
        path: Path to all the dataset
        target_dir: Directory Containing the positive targets
        other_dir: Directory Containing the negative targets
    @returns
        train_set, dev_set and test_set. All three are list of tuples.
        Each tuple contains 2 things index 0= Label and index 1= Filename
    """
    if path[-1] != '/':
        path += '/'

    target_filenames = os.listdir(path+target_dir)
    other_filenames = os.listdir(path+other_dir)
    all_files = (combine_files_with_labels(target_filenames, 1) +
                 combine_files_with_labels(other_filenames, 0))

    shuffle(all_files)

    total = len(all_files)
    train_dev_split = int(0.8 * total)
    dev_test_split = int(0.9 * total)

    train_set = all_files[:train_dev_split]
    dev_set = all_files[train_dev_split:dev_test_split]
    test_set = all_files[dev_test_split:]

    return train_set, dev_set, test_set

def move_files_to_label_dirs(filenames_w_label, original_path, new_path,
                             target_dir, other_dir):
    """
    Move files to label directories
    @params
        filenames_w_label: List of tuples derived from train_dev_test_split function
        original_path: Current Position of dataset files
        new_path: New Position of dataset files
        target_dir: Directory Containing the positive targets
        other_dir: Directory Containing the negative targets
    """
    if original_path[-1] != '/':
        original_path += '/'

    if new_path[-1] != '/':
        new_path += '/'

    for label, filename in filenames_w_label:
        if label == 0:
            o_path = original_path + other_dir + '/'
            n_path = new_path + other_dir + '/'
        else:
            o_path = original_path + target_dir + '/'
            n_path = new_path + target_dir + '/'
        os.rename(o_path + filename, n_path + filename)


def run():
    # originally all images are present in apples and others
    # sub-directories of TRAIN_PATH

    # renaiming them to ints
    rename_files_to_ints(Config.TRAIN_PATH + '/' + Config.TARGET_DIR)
    rename_files_to_ints(Config.TRAIN_PATH + '/' + Config.OTHER_DIR)

    # Spliting files to Train, Dev and Test
    train_set, dev_set, test_set = train_dev_test_split(Config.TRAIN_PATH, Config.TARGET_DIR, Config.OTHER_DIR)

    # Moving files to Dev and Test directories
    move_files_to_label_dirs(dev_set, Config.TRAIN_PATH, Config.DEV_PATH, Config.TARGET_DIR, Config.OTHER_DIR)
    move_files_to_label_dirs(test_set, Config.TRAIN_PATH, Config.TEST_PATH, Config.TARGET_DIR, Config.OTHER_DIR)

if __name__ == '__main__':
    run()
