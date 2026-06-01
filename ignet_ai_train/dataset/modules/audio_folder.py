from os import PathLike
import librosa
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from ignet_ai_train.utils.download import auto_cache_download_list

class AudioFolder(Dataset):
    def __init__(
            self,
            filedir:PathLike, 
            transform=None, 
            sr = 8000,
            dtype = np.float32,
            use_cache:bool=False):
        """
        Args:
            dir (str): 数据根目录，包含多个子文件夹（每个子文件夹是一个类别）
            transform (callable, optional): 可选的样本变换（如归一化）
            sr (int): 音频采样率
            dtype (np.dtype): 音频数据类型
            use_cache (bool): 是否使用缓存
        """
        self.filedir = Path(filedir)
        self.transform = transform
        self.sr = sr
        self.dtype = dtype
        self.samples = []  # 存储 (文件路径, 类别索引)
        self.classes = sorted([d.name for d in self.filedir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self._caches = {}
        self.use_cache = use_cache
        #
        for class_name in self.classes:
            class_dir = self.filedir / class_name
            formats = ['wav', 'aac', 'mp3', 'ogg']
            filelist = [f for format in formats for f in class_dir.glob('*.{}'.format(format))]
            for audio_file in filelist:
                self.samples.append((str(audio_file), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.use_cache and path in self._caches:
            return self._caches[path],label
        
        y, _ = librosa.load(path, sr=self.sr,dtype=self.dtype,offset=0.5) # offset=self.offset,duration=self.duration
        # S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        # S_dB = librosa.power_to_db(S, ref=np.max)
        
        data = y
        if self.transform:
            data = self.transform(data)
        if self.use_cache and path not in self._caches:
            self._caches[path] = data
        return data, label

def preprocess_audio2spectrogram(waveform:np.ndarray,*args, **kwargs):
    # 1. Load audio (librosa automatically converts to mono by default)
    # waveform, sample_rate = librosa.load(audio_path, sr=target_sr,offset=0.5,duration=3)  # sr=None preserves original sampling rate

    # 2. Resample to target_sr if necessary
    # if sample_rate != target_sr:
    #     waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    #     sample_rate = target_sr

    # 3. Compute short-time Fourier transform (STFT)
    # Default parameters similar to torchaudio.Spectrogram():
    #   n_fft=2048 → freq bins = 1025, but torchaudio defaults to n_fft=400
    # To match torchaudio's default more closely:
    n_fft = 400
    hop_length = 200
    win_length = 400

    stft = librosa.stft(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True  # librosa pads input so frames are centered (like torchaudio)
    )

    # 4. Compute magnitude spectrogram
    spectrogram = np.abs(stft)  # shape: (freq_bins, time_frames)

    # 5. Add channel and batch dimensions to mimic torchaudio output:
    # torchaudio: (batch=1, channel=1, freq, time)
    # Here: we assume mono → 1 channel
    spectrogram = np.expand_dims(spectrogram, axis=0)  # add channel dim → (1, freq, time)
    # spectrogram = np.expand_dims(spectrogram, axis=0)  # add batch dim → (1, 1, freq, time)

    return spectrogram


def preprocess_audio2mels(y:np.ndarray,sr:int=8000,n_mels:int=128,normalize:bool=False):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if normalize:
        max_val = np.max(S_dB)
        min_val = np.min(S_dB)
        S_dB = (S_dB - min_val) / (max_val - min_val)
    x = S_dB # (128, 65)
    x = np.expand_dims(x, axis=0)  # (1, 128, 65)
    return x

def preprocess_random_subsample(arr:np.ndarray, sr:int = 8000,length:int=None,):
    total_length = arr.size
    if length is None:
        length = int(sr * 3.0) # 3 秒
    if length > total_length:
        # 填充 0
        arr = np.concatenate([arr, np.zeros(length - arr.size)])
    max_sample_index = (total_length - length) // sr
    if max_sample_index <= 0: 
        sample_index = 0
    else:
        sample_index = np.random.randint(0,max_sample_index)
    start = sample_index * sr
    return arr[start:start + length]

def add_noise(arr:np.ndarray,factor = 0.005):
    max_val = np.max(arr) * factor * 0.5
    min_val = np.min(arr) * factor * 0.5
    x = np.random.uniform(min_val,max_val,arr.shape)
    return arr + x


def get_dataset(
        folder:str,
        sr:int=16000,
        duration:int=3,
        use_noise:bool=False,
        use_cache:bool=False,
        dtype:str|np.dtype=None,
        **kwargs):
    """
    加载音频数据集
    Args:
        folder (str): 数据集根目录,或URL内容指向的资源列表
        sr (int): 音频采样率
        duration (float): 音频时长
        use_cache (bool): 是否使用缓存
        use_noise (bool): 是否添加噪声
        dtype (np.dtype): 音频数据类型
        kwargs: 其他参数
    """
    # from transformers import ASTFeatureExtractor
    if isinstance(dtype,str):
        dtype = np.dtype(dtype)
    elif dtype is None:
        dtype = np.float32

    if folder.startswith(("http://","https://")):
        folder = auto_cache_download_list(folder)
    
    process_list = []
    import torchvision.transforms as transforms
    process_list.append(transforms.Lambda(lambda arr: preprocess_random_subsample(arr, sr=sr,length=int(sr*duration))))
    if use_noise:
        process_list.append(transforms.Lambda(lambda arr: add_noise(arr)))
    process_list.append(transforms.Lambda(lambda arr: preprocess_audio2mels(arr,sr=sr)))
    process_list.append(transforms.Lambda(lambda arr: arr.astype(dtype)))
    
    # processer = ASTFeatureExtractor()
    trans = transforms.Compose(process_list)
    return AudioFolder(folder,transform=trans,sr=sr,use_cache=use_cache,dtype=dtype,**kwargs)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    
    sr = 16000
    duration = 7
    dataset = get_dataset('D:/work/data/baby_cry/baby_cry',duration=duration)
    (data,label) = dataset[0] # tuple[np.ndarray,int] 

    print(data.shape,label) #(1, 256, 94),7


    S_dB = data[0]
    import librosa.display
    import matplotlib.pyplot as plt
    print(S_dB.shape,label)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()



