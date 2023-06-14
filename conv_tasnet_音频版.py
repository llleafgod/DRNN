#import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from torch.autograd import Variable
import sys
print(sys.path)
sys.path.append('/home/jiayang/yz/DRNN/utility')
sys.path.append('/home/jiayang/yz/DRNN/utility/models')
sys.path.append('/home/jiayang/yz/DRNN/utility/sdr')
from models import TCN

# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=1024, feature_dim=512, sr=44100, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
               self.layer, self.stack, self.kernel, causal=self.causal)


        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output
    
def load_audio_files(file1,file2,target_sr):
    #加载音频文件
    waveform1, _ = torchaudio.load(file1)
    waveform2, _ = torchaudio.load(file2)

    # 创建 Resample 对象
    resampler = torchaudio.transforms.Resample(_, target_sr)

    # 重新采样音频波形
    waveform1 = resampler(waveform1)
    waveform2 = resampler(waveform2)

     #确保两音频长度相同
    length1 = waveform1.size(1)
    length2 = waveform2.size(1)
    min_length = min(length1, length2)
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]
    
    #将两个音频叠放
    stacked_waveforms = torch.cat([waveform1, waveform2], 0)
    print(stacked_waveforms.size)
    
    return stacked_waveforms
    
    #音频导入
audio_file1 = "/home/jiayang/yz/DRNN/DRNN/伴奏提取.wav"
audio_file2 = "/home/jiayang/yz/DRNN/DRNN/人声提取.wav"

def test_conv_tasnet():
    x = load_audio_files(audio_file1, audio_file2,25000)
    #print(x.shape)
    nnet = TasNet() # 初始化 TasNet 模型
    x = nnet(x)
    s1 = x[0]
    s2 = x[1]

    s1_numpy = s1.detach().numpy()
    s2_numpy = s2.detach().numpy()

    m1 = s1_numpy[0]
    m2 = s1_numpy[1]
    m3 = s2_numpy[0]
    m4 = s2_numpy[1]

    audio_signal1 = librosa.util.normalize(m1, norm=np.inf, axis=None) * (2**15 - 1)
    audio_signal2 = librosa.util.normalize(m2, norm=np.inf, axis=None) * (2**15 - 1)
    audio_signal3 = librosa.util.normalize(m3, norm=np.inf, axis=None) * (2**15 - 1)
    audio_signal4 = librosa.util.normalize(m4, norm=np.inf, axis=None) * (2**15 - 1)
    audio_signal1 = audio_signal1.astype(np.int16)
    audio_signal2 = audio_signal2.astype(np.int16)
    audio_signal3 = audio_signal3.astype(np.int16)
    audio_signal4 = audio_signal4.astype(np.int16)
    sf.write('人声-人声.wav', audio_signal1, 25000, subtype='PCM_16')
    sf.write('人声-背景.wav', audio_signal2, 25000, subtype='PCM_16')
    sf.write('背景-人声.wav', audio_signal3, 25000, subtype='PCM_16')
    sf.write('背景-背景.wav', audio_signal4, 25000, subtype='PCM_16')
    #print(s1.shape)
    #print(s2.shape)


if __name__ == "__main__":
    test_conv_tasnet()


