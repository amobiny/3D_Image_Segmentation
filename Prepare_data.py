import os

folders = os.listdir('./data/train_data')
folders.sort()
annotation_folders = folders[:len(folders)/2]
input_folders = folders[len(folders)/2:]





print()



