import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

full_dataset = []
labels = []

df = pd.read_excel('./data_sheet.xlsx')
for i in range(len(df)):
    try:
        #parse df
        start = '00:'+str(df.start[i])[:-3]
        end = '00:'+str(df.end[i])[:-3]
        link = str(df.link[i])
        name = "video"+str(i)
        label = int(df.severeness_label[i])
        split = str(df.split[i])
        #convert start and end to seconds. take early_start = max of (0, start-30) and set trim = start-early_start
        start = get_sec(start)
        end = get_sec(end)
        diff = str(end-start)
        early_start = max(0, start-30)
        trim = str(start-early_start)
        early_start = str(early_start)
        #save video and audio urls
        video_url, audio_url = os.popen("yt-dlp --youtube-skip-dash-manifest -g "+link).read().split()
        #download crop
        cmd = 'ffmpeg -ss {0} -i "{1}" -ss {0} -i "{2}" -map 0:v -map 1:a -ss {3} -t {4} -c:v libx264 -c:a aac {5}.mp4'.format(early_start, video_url, audio_url, trim, diff, name)
        #os.system(cmd)
        #later center face before resize
        
        #resize video to height of 256
        cmd = 'ffmpeg -i {0}.mp4 -vf scale=-2:256 {0}_256.mp4'.format(name)
        #os.system(cmd)
        #remove original video
        #os.remove(name+".mp4")
        # if label > 0:
        #     label = 1
        # else:
        #     label = 0
        

        bad_vids = [4, 15, 22, 31, 46, 53, 62, 63, 65, 82, 83, 96, 126, 128, 130, 131, 133]

        if i not in bad_vids:
            path = os.path.abspath(name)
            # full_dataset.append((path + '_256.mp4 ' + str(label) + "\n", label))
            full_dataset.append((path, label))

        #create csv files based off split, absolute path, and label
        # path = os.path.abspath(name)
        # if split == 'train':
        #     with open('train.csv', 'a+') as f:
        #         f.write(path + '_256.mp4 ' + str(label) + "\n")
        # elif split == 'val':
        #     with open('val.csv', 'a+') as f:
        #         f.write(path + '_256.mp4 ' + str(label) + "\n")
        # else:
        #     with open('test.csv', 'a+') as f:
        #         f.write(path + '_256.mp4 ' + str(label) + "\n")
    except:
        #keep track of videos that could not be downloaded
        print("video{0} could not be downloaded".format(str(i)))
        with open('error.csv', 'a+') as f:
                f.write("video{0}\n".format(str(i)))
        pass

print(full_dataset)
full_dataset = np.array(full_dataset)
np.random.shuffle(full_dataset)

num_splits = 5
kf = KFold(n_splits=num_splits)

fold_num = 0
for train_index, test_index in kf.split(full_dataset):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_dataset = full_dataset[train_index]
    test_dataset = full_dataset[test_index]
    
    np.random.shuffle(train_dataset)
    val_idx_split = int(len(train_dataset) / (num_splits - 1))
    val_dataset = train_dataset[:val_idx_split]
    train_dataset = train_dataset[val_idx_split:]

    print("Train length: ", len(train_dataset))
    print("Val length: ", len(val_dataset))
    print("Test length: ", len(test_dataset))

    for sample in train_dataset:
        path, label = sample
        with open(f'train_fold{fold_num}.csv', 'a+') as f:
            f.write(path + '_256.mp4 ' + str(label) + "\n")


    for sample in val_dataset:
        path, label = sample
        with open(f'val_fold{fold_num}.csv', 'a+') as f:
            f.write(path + '_256.mp4 ' + str(label) + "\n")

    
    for sample in test_dataset:
        path, label = sample
        with open(f'test_fold{fold_num}.csv', 'a+') as f:
            f.write(path + '_256.mp4 ' + str(label) + "\n")

    fold_num += 1