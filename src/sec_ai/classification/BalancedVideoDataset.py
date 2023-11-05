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
    UniformCropVideo,
)
from torch.utils.data import Dataset
from sec_ai.PackPathway import PackPathway
from pathlib import Path

import os
import numpy as np
import pandas as pd
import av
import pickle
from sklearn.model_selection import train_test_split


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
clip_duration = (num_frames * sampling_rate) / frames_per_second

transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway(),
        ]
    ),
)


def list_folder_paths(path):
    folder_paths = [
        os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
    ]
    return folder_paths


def list_files_in_directory(directory_path):
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(Path(os.path.join(root, file)))
    return file_paths


class BalancedVideoDataset(Dataset):
    def __init__(
        self,
        videos_folder: Path | str = "/data/ubuntu/secai/data/Shoplifting2",
        labels_csv_path: Path | str = "/data/ubuntu/secai/data/Shoplifting2/Shoplifting.csv",
        take_all_videos_in_folder: bool = False,
        index_of_first_video_to_take: int = 0,
        index_of_last_video_to_take: int = 20,
        normal_paths: Path | str = "/data/ubuntu/secai/normal_train.pickle",
        shoplift_paths: Path | str = "/data/ubuntu/secai/shoplift_train.pickle",
    ):
        super().__init__()
        self.labels_csv_path = labels_csv_path

        with open(normal_paths, "rb") as file:
            self.normal_paths = pickle.load(file)
        with open(shoplift_paths, "rb") as file:
            self.shoplift_paths = pickle.load(file)

        self.videos_list = self.normal_paths + self.shoplift_paths
        self.labels = list(np.zeros(len(self.normal_paths))) + list(
            np.ones(len(self.shoplift_paths))
        )

    def __len__(self) -> int:
        return len(self.labels)

    def get_duration(self, video_path) -> float:
        container = av.open(str(video_path))
        # take first video stream
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)

    def __getitem__(self, index: int):
        start_sec = 0
        # print("Video path", self.videos_list[index])
        end_sec = start_sec + self.get_duration(self.videos_list[index])

        label = self.labels[index]

        video = EncodedVideo.from_path(self.videos_list[index])
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        video_data = transform(video_data)
        inputs = video_data["video"]
        return inputs, int(label)


def collate_fn(batch: list):
    slow_path_list = [example[0][0] for example in batch]
    fast_path_list = [example[0][1] for example in batch]
    label_list = [example[1] for example in batch]

    slow_paths_tensor = torch.stack(slow_path_list)
    fast_paths_tensor = torch.stack(fast_path_list)
    labels = torch.tensor(label_list)
    return slow_paths_tensor, fast_paths_tensor, labels
