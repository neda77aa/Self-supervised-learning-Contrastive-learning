"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas
import torch
import numpy as np
import skimage.draw
import torchvision
import echonet_dataloader
from echonet_dataloader.deformation.utils import *
from echonet_dataloader.deformation.utils_image import generate_pair

class Echo(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 fixed_length=16, max_length=250,
                 clips=1,
                 mode = "train", channels = 3,
                 padding = None,conf = None):


        super().__init__(root)
        self.root = root
        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.fnames, self.outcome , self.ejection  = [], [] , []
        self.fixed_length = fixed_length
        self.padding = padding
        self.channels = channels

        # Load video-level labels
        # Read csv file
        with open(os.path.join(self.root, "FileList.csv")) as f:
            data = pandas.read_csv(f)
            
        # Make all TRAIN,VAl and TEST upper
        data["Split"].map(lambda x: x.upper())
        
        
        # Split = Train , Val or Test
        if self.split != "ALL":
            
            data = data[data["Split"] == self.split]
            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            # File names with suffix (.avi)
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            
            
            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            
            # Open VolumeTracings.csv
            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    
                    # New frame index for the given filename
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                        
                     # Add volume lines to trace
                    self.trace[filename][frame].append((x1, y1, x2, y2))
                    
                    
            # Changing the format to numpy array
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

                    
            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            
            # Prepare for getitem
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
            self.mode = mode
            self.conf = conf

    def __getitem__(self, index):
        
        # Find filename of video
        path = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet_dataloader.utils.loadvideo(path).astype(np.float32)
        # print(video.shape)


        # Scale pixel values from 0-255 to 0-1
        video -= 32.260647
        video /= 48.50121
        # video = np.moveaxis(video, 0, 1)

        # Set number of frames
        c, f, h, w = video.shape
        
        
        # index of ED and ES 
        key = self.fnames[index]
        samp_size = abs(self.frames[key][0]-self.frames[key][-1])

        large_key = self.frames[key][-1]
        small_key = self.frames[key][0]
            
        # Index of first and last frame with segmentation
        first_poi = min(small_key, large_key)
        last_poi  = max(small_key, large_key)
        dist = abs(small_key-large_key) 
        
        # Label of frames 
        label  = np.zeros(f)
        label[small_key] = 1 # End systole (small)
        label[large_key] = 2 # End diastole (large)

         
            
        # Add padding 
        if self.padding is not None:
            p = self.padding
            video = np.pad(video, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=0)


        # Gather targets
        target = []
        deformframe = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "LargeFrame":
                if self.channels == 3:
                    target.append(video[:, self.frames[key][-1], :, :])
                    if self.mode == "self_supervised":
                        deform = generate_pair(video[:, self.frames[key][-1], :, :],1,self.conf)
                        deformframe.append(deform)
                if self.channels == 1:
                    target.append(np.expand_dims(video[0, self.frames[key][-1], :, :], axis=0))
                    if self.mode == "self_supervised":
                        deform = generate_pair(np.expand_dims(video[0, self.frames[key][-1], :, :], axis=0),1,self.conf)
                        deformframe.append(deform)
                    
            elif t == "SmallFrame":
                if self.channels == 3:
                    target.append(video[:, self.frames[key][0], :, :])
                    if self.mode == "self_supervised":
                        deform = generate_pair(video[:, self.frames[key][0], :, :],1,self.conf)
                        deformframe.append(deform)
                if self.channels == 1:
                    target.append(np.expand_dims(video[0, self.frames[key][0], :, :], axis=0))
                    if self.mode == "self_supervised":
                        deform = generate_pair(np.expand_dims(video[0, self.frames[key][0], :, :], axis=0),1,self.conf)
                        deformframe.append(deform)
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (112, 112))
                mask = np.zeros((112, 112), np.float32)
                mask[r, c] = 1
                if self.padding is not None:
                    p = self.padding
                    mask = np.pad(mask, ((p,p),(p,p)), mode='constant', constant_values=0)
                target.append(torch.tensor(mask))
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(torch.tensor(np.float32(0)))
                else:
                    target.append(torch.tensor(np.float32(self.outcome[index][self.header.index(t)])))
        if self.mode == "video":
            length = self.length
            # Gather Video
            if dist<length:
                # Take random clips from video
                pre_start = int((length - dist)//2)
                start_index = int(max(0, first_poi - pre_start))  
                end_index = start_index + length 
                video = video[:,start_index : end_index, :, :]
                label = label[start_index:end_index]

            else:
                divider     = np.random.random_sample()*5+2
                start_index = first_poi - dist//divider
                start_index = int(max(0, start_index)//2*2)             
                divider     = np.random.random_sample()*5+2
                end_index   = last_poi +1 + dist//divider #+1 to INCLUDE the frame
                end_index   = int(min(f, end_index)//2*2)
                step = int( np.ceil((end_index-start_index)/ length) )
                list_frame = np.arange(first_poi, last_poi , step, dtype=int)
                list_frame = np.append(list_frame, last_poi)
                list_all = np.arange(start_index,end_index)
                while length!=len(list_frame):
                    np.random.shuffle(list_all)
                    list_frame  = np.unique(np.sort(np.append(list_frame,list_all[:(length - len(list_frame))])))
                video = video[:,list_frame, :, :]
                label = label[list_frame]

            label = torch.tensor(label)
            video = torch.tensor(video)
            
            
        if self.mode == "self_supervised_video":
            deform = generate_pair_d(video,1, self.conf)
            #cine_d = torch.as_tensor(np.array(deform).astype('float'))
            return video,deform,label,target
        if self.mode == "self_supervised":
            return target,deformframe
        elif self.mode == "simclr":
             return target
            
        return target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)