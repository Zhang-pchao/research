import os
import numpy as np
import shutil

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 

def check_npy(loadData):
    print(type(loadData))
    print(loadData.dtype)
    print(loadData.ndim)
    print(loadData.shape)
    print(loadData.size)
    print(loadData[:1])
    #print(loadData[-1:])    

def connect_npy_files(output_file,root,dirs,npyfile):
    file_list = []
    for i in dirs:
        file = os.path.join(root,i,'set.000',npyfile)
        file_list.append(file)

    # Load the first file to get the shape information
    first_file = np.load(file_list[0])
    output_data = first_file

    # Iterate through the rest of the files and concatenate them if shape[1] matches
    for file_name in file_list[1:]:
        data = np.load(file_name)
        if data.shape[1] == output_data.shape[1]:
            output_data = np.concatenate((output_data, data), axis=0)
        else:
            print(f"Ignoring file {file_name} as the shape[1] does not match the first file.")

    # Save the concatenated data into a new .npy file
    np.save(output_file, output_data)

root  = "/data/HOME_BACKUP/pengchao/glycine/deepks/M062X_Dataset/Glycine54H2O_efield/dp_dataset_dpwc_rev_loc"
dirs = [
"001",
"002",
"003",
"004",
"005",
"006",
"007",
"008",
"009",
"010",
"011",
"012",
]

output_dir = os.path.join(root,"concat","set.000")
mkdir(output_dir)
npylist =['box.npy','coord.npy','force.npy','energy.npy','atomic_dipole.npy']
for i in npylist:
    npyfile = i
    output_file = os.path.join(output_dir,npyfile)
    connect_npy_files(output_file,root,dirs,npyfile)
    check_npy(loadData = np.load(output_file))

destination_folder = os.path.join(root,"concat")
rawlist = ['type.raw','type_map.raw']
for i in rawlist:
    source_file = os.path.join(root,dirs[0],i)
    shutil.copy(source_file, destination_folder)

def choose_interval_and_save(input_file, output_file_intervals, output_file_remainder, interval=20):
    # Load the data from the input .npy file
    data = np.load(input_file)

    # Choose data at regular intervals and save them into a new .npy file
    chosen_data = data[::interval]

    # Save the chosen interval data into a new .npy file
    np.save(output_file_intervals, chosen_data)
    
    # Save the remaining data into a new .npy file
    remaining_data = np.delete(data, np.arange(0, len(data), interval), axis=0)
    np.save(output_file_remainder, remaining_data)

data_path  = "/data/HOME_BACKUP/pengchao/glycine/deepks/M062X_Dataset/Glycine54H2O_efield/dp_dataset_dpwc_rev_loc/concat"
train_path = data_path+'_train/set.000'
test_path  = data_path+'_test/set.000'
mkdir(train_path)
mkdir(test_path)
for i in npylist:
    input_file = os.path.join(data_path,"set.000",i)
    output_file_intervals = os.path.join(test_path,i)
    output_file_remainder = os.path.join(train_path,i)
    choose_interval_and_save(input_file, output_file_intervals, output_file_remainder)
    check_npy(loadData = np.load(output_file_intervals))
    check_npy(loadData = np.load(output_file_remainder))
    check_npy(loadData = np.load(input_file)) 

rawlist = ['type.raw','type_map.raw']
for i in rawlist:
    source_file = os.path.join(data_path,i)
    shutil.copy(source_file, data_path+'_train')
    shutil.copy(source_file, data_path+'_test')

