import mne
import numpy as np
# tools for plotting confusion matrices kappa
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
# 文件路径和文件名作为可修改变量
d_num=str(3)  # 3 5 7
data_file = "D:/资料/毕设/MyCode/dataset0/BCICIV_2a_gdf/A0"+d_num+"T.gdf"
save_path = "D:/资料/毕设/MyCode/npzdata/"
save_file_name = "A0"+d_num+"Tdata.npz"

raw = mne.io.read_raw_gdf(data_file, stim_channel="auto", verbose='ERROR',
                          exclude=(["EOG-left", "EOG-central", "EOG-right"]))  # 去除眼电通道
raw.plot()
plt.show()
raw.plot_psd(average=True)
plt.show()
print(raw.info)
print(raw.ch_names)
# Find the events time positions
events, _ = mne.events_from_annotations(raw)

# Pre-load the data

raw.load_data()

# Filter the raw signal with a band pass filter in 7-35 Hz
# 因此，47%的研究建议使用6-35 Hz的频率带。然而，使用带通滤波器并不能轻易排除伪迹，因为它们可能会干扰有效的ERD/ERS带。其他一些研究（35%）建议使用比6-35 Hz更宽的频率带，范围在0-40 Hz
raw.filter(6., 35., fir_design='firwin')

# Remove the EOG channels and pick only desired EEG channels

raw.info['bads']

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

# Extracts epochs of 3s time period from the datset into 288 events for all 4 classes

tmin, tmax = 2., 6.
# left_hand = 769,right_hand = 770,foot = 771,tongue = 772
event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
# event_id = dict({'783': 7})
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)

labels = epochs.events[:, -1] - 7

data = epochs.get_data()

# 删除最后一个维度
data = data[:, :, :-1]

# 将数据保存为npz格式，并包含文件路径和文件名
np.savez(save_path + save_file_name, train_data=data, test_label=labels)

# 加载保存的数据
loaded_data = np.load(save_path + save_file_name)
a = loaded_data['train_data']
b = loaded_data['test_label']

print("训练数据形状：", a.shape)
print("测试标签形状：", b.shape)
