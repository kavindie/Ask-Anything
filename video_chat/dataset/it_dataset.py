import logging
import os
import json
import sqlite3
import random
from os.path import basename

import numpy as np
import datetime

from dataset.base_dataset import ImageVideoBaseDataset
from dataset.utils import load_anno
from dataset.video_utils import VIDEO_READER_FUNCS
from utils.distributed import is_main_process

logger = logging.getLogger(__name__)


class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self, ann_file, transform, 
        system="", role=("Human", "Assistant"),
        start_token="<Image>", end_token="</Image>",
    ):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = "###"
        self.end_signal = " "
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        logger.info(f"Random shuffle: {self.random_shuffle}")
        logger.info(f"Individual sentence for prompt: {self.individual}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        if 'start' in self.anno[index]:
            anno['start'] = self.anno[index]['start']
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        conversation = self.system
        # rstrip() for the extra " "
        conversation += (
            self.begin_signal + self.role[0] + ": " + 
            self.start_token + self.end_token + msg.rstrip() + self.end_signal
        )
        for idx, sentence in enumerate(qa):
            q = sentence["q"]
            a = sentence["a"]
            conversation += (self.begin_signal + self.role[0] + ": " + q + self.end_signal)
            conversation += (self.begin_signal + self.role[1] + ": " + a + self.end_signal)
        conversation += self.begin_signal
        return conversation

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"])
            conversation = self.process_qa(ann["qa"])
            return image, conversation, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


class ITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
    ):
        super().__init__(
            ann_file, transform, 
            system=system, role=role,
            start_token=start_token, end_token=end_token,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg
        self.add_movie_start = add_movie_start

        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True)
            # " " should be added in the start and end
            msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            conversation = self.process_qa(ann["qa"], msg)
            return video, conversation, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)