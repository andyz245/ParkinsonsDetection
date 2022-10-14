import os
import pandas as pd

binary = False

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

df = pd.read_excel('./data_sheet.xlsx')
for i in range(len(df)):
    try:
        #parse df
        errors = [14, 21, 26, 30, 34, 36, 51, 60, 63, 80, 81, 92, 94, 96, 115, 124, 126, 127, 128, 129, 131, 82, 109, 114, 116, 59, 82, 83, 118, 115]
        name = "video"+str(i)
        start = '00:'+str(df.start[i])[:-3]
        end = '00:'+str(df.end[i])[:-3]
        link = str(df.link[i])
        label = int(df.severeness_label[i])
        if binary:
            if label > 0:
                label = 1
            else:
                label = 0
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
        #download time crop
        cmd = 'ffmpeg -ss {0} -i "{1}" -ss {0} -i "{2}" -map 0:v -map 1:a -ss {3} -t {4} -c:v libx264 -c:a aac {5}.mp4'.format(early_start, video_url, audio_url, trim, diff, name)
        os.system(cmd)
        #center face
        cmd = 'python ~/mmpose/demo/zitong_crop.py \
                ~/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                ~/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
                https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
                --video-path /home/swli2/mmpose/data/{0}.mp4 \
                --out-video-root ~/mmpose/data'.format(name)
        os.system(cmd)
        #resample to 20 fps
        cmd = 'ffmpeg -i vis_{0}.mp4 -filter:v fps=20 {0}_20fps.mp4'.format(name)
        os.system(cmd)
        #resize video to height of 256
        cmd = 'ffmpeg -i {0}_20fps.mp4 -vf scale=-1:256 {0}_final.mp4'.format(name)
        os.system(cmd)
        #remove previous video
        os.remove(name+".mp4")
        os.remove("vis_"+name+".mp4")
        os.remove(name+"_20fps.mp4")
        
        #get region coordinates
        cmd = 'python ~/mmpose/demo/top_down_video_demo_with_mmdet.py \
                ~/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                ~/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
                https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
                --video-path /home/swli2/mmpose/data/{0}_final.mp4 \
                --out-video-root vis_results'.format(name)
        os.system(cmd)
        
        
        #create csv files based off split, absolute path, and label
        path = os.path.abspath(name)
        if i not in errors:
            with open(split+'.csv', 'a+') as f:
                   f.write(path + '_final.mp4 ' + str(label) + "\n")
    except:
        #keep track of videos that could not be downloaded
        print("video{0} could not be downloaded".format(str(i)))
        with open('error.csv', 'a+') as f:
            f.write("video{0}\n".format(str(i)))

        pass
        
