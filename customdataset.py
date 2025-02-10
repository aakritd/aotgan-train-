from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from dataset import image_train_set, image_val_set, mask_train_set, mask_val_set
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data, image_train_set, image_val_set, mask_train_set, mask_val_set):
        self.data = data
        self.img_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((512,512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation = transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        if self.data == "train":
            self.imgpath = image_train_set
            self.maskpath = mask_train_set
        if self.data == "val":
            self.imgpath = image_val_set
            self.maskpath = mask_val_set

    def __len__(self):
        if (len(self.imgpath) > len(self.maskpath)):
            return len(self.maskpath)
        else:
            return len(self.imgpath)
    
    def __getitem__(self, index):
        img = Image.open(self.imgpath[index])
        mask = Image.open(self.maskpath[index])

        mask = self.mask_transform(mask.convert('L'))
        img = self.img_transform(img.convert('RGB'))* 2.0 - 1.0

        return (img, mask)


def create_dataset(data_path, mask_path, training_data, val_data):
    import os
    import matplotlib.pyplot as plt

    training_data_mask = int(training_data/4)
    val_data_mask = int(val_data/4)




    DATA_DIR = data_path

    # Define the size of the subset of the dataset to use
    subset_size =len(DATA_DIR)  # Change this value as needed

    # List and sort the files
    datasetList = sorted(os.listdir(DATA_DIR))

    # print(f"Number of images in the subset: {len(datasetList)}")

    # Define split ratios
    train_ratio = 0.9
    val_ratio = 0.1


    # Calculate the split indices
    total_images = len(datasetList)
    train_count = int(train_ratio * total_images)
    val_count = int(val_ratio * total_images)


    # Split the dataset
    image_train_set = datasetList[:training_data]
    image_val_set = datasetList[training_data:training_data+val_data]


    # Append the directory path to each filename
    image_train_set = [os.path.join(DATA_DIR, file) for file in image_train_set]
    image_val_set = [os.path.join(DATA_DIR, file) for file in image_val_set]




    # # Testing
    # print(f"Example file path in training set: {train_set[0]}")
    # plt.imshow(plt.imread(train_set[0]))
    # plt.axis('off')
    # plt.show()

    # Print the number of images in each set
    print(f"Total images: {total_images}")
    print(f"Training set: {len(image_train_set)} images")
    print(f"Validation set: {len(image_val_set)} images")

    # MASK_DIR = r'C:\Aakrit\College\8th Sem\Major Project\AOTGAN-github\AOT-GAN-for-Inpainting\maskDataset\maskDatasetAOTGAN\masks'
    # datasetList = sorted(os.listdir(MASK_DIR))


    # # Calculate the split indices
    # total_images = len(datasetList)
    # train_count = int(train_ratio * total_images)
    # val_count = int(val_ratio * total_images)

    # # Split the dataset
    # mask_train_set = datasetList[:train_count]
    # mask_val_set = datasetList[train_count:]

    # # Append the directory path to each filename
    # mask_train_set = [os.path.join(MASK_DIR, file) for file in mask_train_set]
    # mask_val_set = [os.path.join(MASK_DIR, file) for file in mask_val_set]

    # Print the number of images in each set
    # print(f"Total masks: {total_images}")
    # print(f"Training set: {len(mask_train_set)} masks")
    # print(f"Validation set: {len(mask_val_set)} masks"


    import os
    import cv2


    DATA_DIR_mask = mask_path
    DATA_DIR_class1 = os.path.join(DATA_DIR_mask, '1-8')
    DATA_DIR_class2 = os.path.join(DATA_DIR_mask, '8-16')
    DATA_DIR_class3 = os.path.join(DATA_DIR_mask, '16-24')
    DATA_DIR_class4 = os.path.join(DATA_DIR_mask, '24-32')

    comp_datasetListClass1 = []
    comp_datasetListClass2 = []
    comp_datasetListClass3 = []
    comp_datasetListClass4 = []

    # List and sort the files for each class
    datasetListClass1 = sorted(os.listdir(DATA_DIR_class1))

    for file in datasetListClass1:
        comp_datasetListClass1.append(os.path.join(DATA_DIR_class1, file))
        
    mask_train_set_1 = comp_datasetListClass1[val_data_mask:training_data_mask + val_data_mask]  # All images except first 125 for training
    mask_val_set_1 = comp_datasetListClass1[:val_data_mask]  # First 125 for validation

    datasetListClass2 = sorted(os.listdir(DATA_DIR_class2))

    for file in datasetListClass2:
        comp_datasetListClass2.append(os.path.join(DATA_DIR_class2, file))

    mask_train_set_2 = comp_datasetListClass2[val_data_mask:training_data_mask + val_data_mask]  # All images except first 125 for training
    mask_val_set_2 = comp_datasetListClass2[:val_data_mask]  # First 125 for validation

    datasetListClass3 = sorted(os.listdir(DATA_DIR_class3))

    for file in datasetListClass3:
        comp_datasetListClass3.append(os.path.join(DATA_DIR_class3, file))


    mask_train_set_3 = comp_datasetListClass3[val_data_mask:training_data_mask + val_data_mask]  # All images except first 125 for training
    mask_val_set_3 = comp_datasetListClass3[:val_data_mask]  # First 125 for validation

    datasetListClass4 = sorted(os.listdir(DATA_DIR_class4))

    for file in datasetListClass4:
        comp_datasetListClass4.append(os.path.join(DATA_DIR_class4, file))


    mask_train_set_4 = comp_datasetListClass4[val_data_mask:training_data_mask + val_data_mask]  # All images except first 125 for training
    mask_val_set_4 = comp_datasetListClass4[:val_data_mask]  # First 125 for validation

    
    print(len(mask_train_set_1), len(mask_train_set_2), len(mask_train_set_3), len(mask_train_set_4))
    # Combine all sets
    mask_train_set = mask_train_set_1 + mask_train_set_2 + mask_train_set_3 + mask_train_set_4
    mask_val_set = mask_val_set_1 + mask_val_set_2 + mask_val_set_3 + mask_val_set_4

    print((mask_train_set[0]))


    print('Before removing : ',len(mask_train_set))
    print(len(mask_val_set))

    def signal_if_no_string(lst):
        if all(isinstance(item, str) for item in lst):
            print("All elements are strings.")
        else:
            print("At least one non-string detected in the list!")

    # Example Usage
    signal_if_no_string(mask_val_set)



    def remove_non_strings(input_list):
        return [item for item in input_list if isinstance(item, str)]

    # Example Usage
    mask_train_set = remove_non_strings(mask_train_set)

    def remove_non_strings(input_list):
        return [item for item in input_list if isinstance(item, str)]

    # Example Usage
    mask_val_set = remove_non_strings(mask_val_set)

    def signal_if_no_string(lst):
        if all(isinstance(item, str) for item in lst):
            print("All elements are strings.")
        else:
            print("At least one non-string detected in the list!")

    # Example Usage
    signal_if_no_string(mask_train_set)

    def signal_if_no_string(lst):
        if all(isinstance(item, str) for item in lst):
            print("All elements are strings.")
        else:
            print("At least one non-string detected in the list!")

    # Example Usage
    signal_if_no_string(mask_val_set)


    print(len(mask_train_set))
    print(len(mask_val_set))

    # ig = cv2.imread(mask_train_set[0])
    # print(ig.shape)

    # plt.imshow(plt.imread(mask_train_set[0]), cmap='gray')
    # plt.axis('off')
    # plt.show()
    traindata = CustomDataset("train", image_train_set, image_val_set, mask_train_set, mask_val_set)


    valdata = CustomDataset("val", image_train_set, image_val_set, mask_train_set, mask_val_set)

    return traindata, valdata





# traindata, valdata = create_dataset(r'C:\Aakrit\College\8th Sem\Major Project\AOTGAN-github\AOT-GAN-for-Inpainting\imageDataset\celebaDatasetAOTGAN\img_align_celeba\img_align_celeba',
#                                      r'C:\Aakrit\College\8th Sem\Major Project\aotgan(scratch)\aotgan-mask', 100, 100)

