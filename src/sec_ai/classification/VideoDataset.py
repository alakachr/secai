from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
from torch.utils.data import Dataset
from sec_ai.PackPathway import PackPathway
from pathlib import Path

import os
import numpy as np
import pandas as pd
import av


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3
clip_duration = (num_frames * sampling_rate)/frames_per_second

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)
def list_folder_paths(path):
    folder_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folder_paths


def list_files_in_directory(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(Path(os.path.join(root, file)))
    return file_paths

class VideoDataset(Dataset):
    def __init__(
        self,
        videos_folder:Path|str = "/data/ubuntu/secai/data/Shoplifting2",
        labels_csv_path:Path |str= "/data/ubuntu/secai/data/Shoplifting2/Shoplifting.csv",
        take_all_videos_in_folder: bool=False,
        index_of_first_video_to_take: int = 0,
        index_of_last_video_to_take:int =20
     
    ):
        super().__init__()
        self.labels_csv_path = labels_csv_path

        self.videos_folder_list = list_folder_paths(videos_folder)
        if not take_all_videos_in_folder:
            self.videos_folder_list =  self.videos_folder_list[index_of_first_video_to_take:index_of_last_video_to_take]
        

        self.videos_list =  []
        for folder in self.videos_folder_list:
            self.videos_list+=list_files_in_directory(folder)
       
        df = pd.read_csv(labels_csv_path, names= ['video_name', 'action_category', 'label'])

        self.labels = []
        for video_path in self.videos_list:
            label = int(df[df.video_name == video_path.name.split(".")[0]].label)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)
    
    def get_duration(self,video_path)->float:
        
        container = av.open(str(video_path))
        # take first video stream
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)


    def __getitem__(self, index: int) :
        start_sec = 0
        end_sec = start_sec + self.get_duration(self.videos_list[index])

       
        label = self.labels[index]
      
        video = EncodedVideo.from_path(self.videos_list [index])
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        video_data = transform(video_data)
        inputs = video_data["video"]
        return inputs, label







def collate_fn(batch: list):
    slow_path_list = [example[0][0] for example in batch]
    fast_path_list = [example[0][1] for example in batch]
    label_list = [example[1] for example in batch]

    slow_paths_tensor = torch.stack(slow_path_list)
    fast_paths_tensor = torch.stack(fast_path_list)
    labels =torch.tensor(label_list)
    return slow_paths_tensor, fast_paths_tensor,labels