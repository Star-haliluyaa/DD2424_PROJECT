from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import os

# use_baseline
use_baseline = "yes"
# Relative dir. for dataset (input)
dataset_dir = "../data/1D_dataset/"
# Relative dir. for output
result_dir = None
out_dir_with = "../data/3D_dataset/with_base/"
out_dir_without = "../data/3D_dataset/without_base/"

def read_file(file):
    file = sio.loadmat(file)
    trial_data = file['data']
    base_data = file["base_data"]
    return trial_data, base_data, file["arousal_labels"], file["valence_labels"], file["dominance_labels"]


def get_vector_deviation(vector1, vector2):
    return vector1 - vector2


def get_dataset_deviation(trial_data, base_data):
    # trial_data: 2400*128  (videos*seconds)*(frequencies*channels)
    # base_data : 40*128    (videos)*(frequencies*channels)
    # here, we assume the base_data is a constant bias, and we will adjust trial_data accordingly.
    new_dataset = np.empty([0, 128])
    for i in range(0, 2400):
        base_index = i // 60
        # print(base_index)
        base_index = 39 if base_index == 40 else base_index
        new_record = get_vector_deviation(trial_data[i], base_data[base_index]).reshape(1, 128)
        # print(new_record.shape)
        new_dataset = np.vstack([new_dataset, new_record])
    # print("new shape:",new_dataset.shape)
    return new_dataset


def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], 0, data[16], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[1], 0, data[17], 0, 0, 0)
    data_2D[2] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[3] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[4] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[5] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[6] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[7] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:9*9
    return data_2D


def pre_process(path, use_baseline):
    # DE feature vector dimension of each band
    data_3D = np.empty([0, 9, 9])
    sub_vector_len = 32
    trial_data, base_data, arousal_labels, valence_labels, dominance_labels = read_file(path)
    # 60s       # 3s      # label 1       # label 2       # label 3         
    if use_baseline == "yes":
        data = get_dataset_deviation(trial_data, base_data) # consider base_data as bias
        data = preprocessing.scale(data, axis=1, with_mean=True, with_std=True, copy=True)
    else:
        data = preprocessing.scale(trial_data, axis=1, with_mean=True, with_std=True, copy=True)
    # convert 128 (4*32 frequencies*channels) vector ---> 4*9*9 cube (frequencies*new_channels)
    for vector in data:
        for band in range(0, 4):
            data_2D_temp = data_1Dto2D(vector[band * sub_vector_len:(band + 1) * sub_vector_len])
            data_2D_temp = data_2D_temp.reshape(1, 9, 9)
            # print("data_2d_temp shape:",data_2D_temp.shape)
            data_3D = np.vstack([data_3D, data_2D_temp])
    data_3D = data_3D.reshape(-1, 4, 9, 9) # 2400 (videos*seconds) * 4*9*9  (cube, frequencies*new_channels)
    print("final data shape:", data_3D.shape)
    return data_3D, arousal_labels, valence_labels, dominance_labels


if __name__ == '__main__':

    if use_baseline == "yes":
        result_dir = out_dir_with
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = out_dir_without
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)

    for file in os.listdir(dataset_dir):
        print("processing: ", file, "......")
        file_path = os.path.join(dataset_dir, file)
        data, arousal_labels, valence_labels, dominance_labels = pre_process(file_path, use_baseline)
        print("final shape:", data.shape)
        sio.savemat(result_dir + "3D" + file,
                    {"data": data,                          # 2400 (videos*seconds) * 4*9*9  (cube, frequencies*new_channels)
                     "valence_labels": valence_labels,      # 2400      (videos*seconds)
                     "arousal_labels": arousal_labels,      # 2400      (videos*seconds)
                     "dominance_labels": dominance_labels}) # 2400      (videos*seconds)

