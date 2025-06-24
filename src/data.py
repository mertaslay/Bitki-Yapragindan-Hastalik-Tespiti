"""
Data.py works for two components:
1 -> Preparing data directions
2 -> Preparing labels of the data
"""

from constant import *
import os
import glob
import shutil
import random


def parse_raw_dataset():

    DATASET_DIR = DATA_DIR
    if not os.path.exists(DATASET_DIR):
        print(f"Creating dataset directory at {DATASET_DIR}")
        os.makedirs(DATASET_DIR, exist_ok=True)
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    VALID_DIR = os.path.join(DATASET_DIR, "valid")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    val_split = 0.15
    test_split = 0.10

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    raw_dataset_path = "dataset"
    for class_folder in os.listdir(raw_dataset_path):
        class_path = os.path.join(raw_dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        val_count = int(len(images) * val_split)
        test_count = int(len(images) * test_split)

        val_images = images[:val_count]
        test_images = images[val_count:val_count + test_count]
        train_images = images[val_count + test_count:]

        # Hedef klasörleri oluştur
        os.makedirs(os.path.join(TRAIN_DIR, class_folder), exist_ok=True)
        os.makedirs(os.path.join(VALID_DIR, class_folder), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_folder), exist_ok=True)

        # train klasörüne taşı
        for img in train_images:
            shutil.move(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_folder, img))

        # valid klasörüne taşı
        for img in val_images:
            shutil.move(os.path.join(class_path, img), os.path.join(VALID_DIR, class_folder, img))

        # test klasörüne taşı
        for img in test_images:
            shutil.move(os.path.join(class_path, img), os.path.join(TEST_DIR, class_folder, img))



def getPathListAndLabelsOfPlants(directionOfDataset:str):
    """
    This functions returns path lists and labels of the images.
    :param directionOfDataset: direction of dataset that we want to process.
    :return:
    """

    file_path_list = glob.glob(os.path.join(directionOfDataset, "*"))

    path_list = []
    labeled_list = []
    for iter, file in enumerate(file_path_list):
        for path in glob.glob(os.path.join(file, "*")):
            path_list.append(path)
            labeled_list.append(iter) if directionOfDataset == TRAIN_DIR else []

    return path_list, labeled_list if directionOfDataset == TRAIN_DIR else []

def getLabeledListAsDictionary(directionOfDataset: str):
    """
    directionOfDataset içindeki klasörleri alfabetik sırada alıp,
    0,1,2... index ile eşleyen bir dict döner.
    """
    import glob, os
    # Klasörleri alfabetik sırada oku
    folders = sorted(glob.glob(os.path.join(directionOfDataset, "*")))
    # Her klasörün basename’ini al
    folder_names = [os.path.basename(f) for f in folders]
    # Index eşlemesini yap
    return {i: folder_names[i] for i in range(len(folder_names))}

def findEquivalentOfLabels(path_list:list,
                           labeledListAsDictionary:dict,
                           directionOfDataset:str):
    """
    Finds Equivalent of labels of folder names that is given.
    For example, in the valid folder, plants and their disease may not be the same as train folder.
    This function provides to know which category valid or test folder names belong to.
    :param path_list: list of str / list of every file in the folders
    :param labeledListAsDictionary: dict / Main label for finding the equivalent of file names
    :param directionOfDataset : str / is it a valid folder or test folder?
    :return:
    list of labels that one-to-one with list of images
    """
    # Finds the length of direction -> "..\\dataset\\Parsed Dataset\\train" = len(dir) is 4
    # We find this variable to find the plant and disease name in below.
    wordsToPlantName = len(directionOfDataset.split("\\"))

    labelOfFolder = []
    for iter, path in enumerate(path_list):
        plantAndDiseaseName = path.split("\\")[wordsToPlantName]
        labelOfFolder.append([key for key, value in labeledListAsDictionary.items() if value == plantAndDiseaseName].pop())
    return labelOfFolder




if __name__ == '__main__':

    parse_raw_dataset()

    path_list, label = getPathListAndLabelsOfPlants(TRAIN_DIR)

    splitted = getLabeledListAsDictionary(TRAIN_DIR)

    # Valid folder
    valid_path_list, _ = getPathListAndLabelsOfPlants(VALID_DIR)
    test_path_list, _ = getPathListAndLabelsOfPlants(TEST_DIR)
    train_path_list, _ = getPathListAndLabelsOfPlants(TRAIN_DIR)
    print("len of train path list..: ", len(train_path_list))
    print("len of valid path list..: ", len(valid_path_list))
    print("len of test path list..: ", len(test_path_list))

    # Labels of valid folder
    labelsOfValid = findEquivalentOfLabels(path_list=valid_path_list, labeledListAsDictionary=splitted, directionOfDataset=VALID_DIR)
    #print(labelsOfValid)







