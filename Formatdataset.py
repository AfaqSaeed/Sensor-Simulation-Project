import cv2
import os
import matplotlib.pyplot as plt 
data_path = r"C:\Study\Sensor-Simulation-Project\dataset"
image_names = os.listdir(data_path)
import os
import shutil
from sklearn.model_selection import train_test_split

def split_pix2pix_dataset_folders(input_folder, target_folder,  test_size=0.1, eval_size=0.2, random_state=None):
    """
    Splits the Pix2Pix dataset stored in separate folders into training, testing, and evaluation sets.

    Parameters:
    - input_folder: Path to the folder containing input images.
    - target_folder: Path to the folder containing target images.
    - output_folder: Path to the output folder where the split dataset will be stored.
    - test_size: Size of the testing set (default is 0.2).
    - eval_size: Size of the evaluation set (default is 0.2).
    - random_state: Random seed for reproducibility (default is None).

    Returns:
    - None. The split dataset will be stored in the output_folder.
    """

    # List all input and target image files
    input_files = os.listdir(input_folder)
    target_files = os.listdir(target_folder)

    # Remove any files starting with '.'
    input_files = [f for f in input_files if ".png" in f    ]
    target_files = [f for f in target_files if ".png" in f]

    # Ensure that the number of input and target images match
    assert len(input_files) == len(target_files), "Number of input and target images must match."

    # Splitting the dataset into train, test, and eval
    train_input_files, remaining_input_files, train_target_files, remaining_target_files = train_test_split(input_files, target_files, test_size=(test_size + eval_size), random_state=random_state)
    test_input_files, eval_input_files, test_target_files, eval_target_files = train_test_split(remaining_input_files, remaining_target_files, test_size=(eval_size / (test_size + eval_size)), random_state=random_state)

    # Function to copy files from one folder to another
    def copy_files(src_files, src_folder, dest_folder):
        for file_name in src_files:
            src_path = os.path.join(src_folder, file_name)
            dest_path = os.path.join(dest_folder, file_name)
            print(src_path,dest_path)
            shutil.copy(src_path, dest_path)

    # Copy files to output folders
    copy_files(train_input_files, input_folder, os.path.join(input_folder, 'train'))
    copy_files(train_target_files, target_folder, os.path.join(target_folder, 'train'))
    copy_files(test_input_files, input_folder, os.path.join(input_folder, 'test'))
    copy_files(test_target_files, target_folder, os.path.join(target_folder, 'test'))
    copy_files(eval_input_files, input_folder, os.path.join(input_folder, 'eval'))
    copy_files(eval_target_files, target_folder, os.path.join(target_folder, 'eval'))

    print("Dataset split and saved successfully.")

ambient_path = os.path.join(data_path,"Ambient")
lidar_path = os.path.join(data_path,"Lidar")


split_pix2pix_dataset_folders(ambient_path,lidar_path)


# for img_name in image_names:
#     file_path = os.path.join(data_path , img_name) 
#     image = cv2.imread(file_path)
#     print("Image Shape",image.shape) ###########Split between LIDAR and Ambient Images 
#     ambient = image[:,0:128]
#     lidar = image[:,128:]

#     ######## Saving them in corresponding folders #####################################
#     lidar_path = os.path.join(data_path ,"Lidar",img_name)
#     ambient_path = os.path.join(data_path ,"Ambient",img_name)

    
#     cv2.imwrite(lidar_path,lidar)
#     cv2.imwrite(ambient_path,ambient)

