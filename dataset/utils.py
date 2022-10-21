import os
import os.path
import shutil
from typing import Tuple


def split_data(directory: str) -> Tuple[str, str]:
    """
    Splits the files to train and validation sets, since they are seperated by different folders
    :param directory: Directory to find the folder containing the data
    :return: New paths containing train and validation set split
    """
    source_files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    new_image_path_train = None
    new_image_path_val = None
    assert os.path.exists(directory), "Directory does not exist"

    if len(source_files) == 0:
        new_image_path_train, new_image_path_val = os.path.join(
            directory, "Train"
        ), os.path.join(directory, "Val")

    for file in source_files:
        folder_name = "Val" if file.startswith("V") else "Train"
        new_path = os.path.join(directory, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_image_path = os.path.join(directory, file)
        new_image_path_val = os.path.join(new_path, file)
        shutil.move(old_image_path, new_image_path_val)

    return new_image_path_train or "", new_image_path_val or ""
