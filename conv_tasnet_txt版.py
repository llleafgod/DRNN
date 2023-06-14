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
def load_audio_files(file1,file2):
    #加载音频文件
    waveform1, _ = torchaudio.load(file1)
    waveform2, _ = torchaudio.load(file2)
  
     # Ensure the two audio files have the same length
    length1 = waveform1.size(1)
    length2 = waveform2.size(1)
    min_length = min(length1, length2)
    waveform1 = waveform1[:, :min_length]
    waveform2 = waveform2[:, :min_length]
    
    # Stack the two audio files together
    stacked_waveforms = torch.cat([waveform1, waveform2], 0)
    print(stacked_waveforms.size)
    
    return stacked_waveforms
#音频导入
audio_file1 = "/home/jiayang/yz/DRNN/DRNN/伴奏提取.wav"
audio_file2 = "/home/jiayang/yz/DRNN/DRNN/人声提取.wav"

def test_conv_tasnet():
    x = load_audio_files(audio_file1, audio_file2)
    #print(x.shape)
    nnet = TasNet() # 初始化 TasNet 模型
    x = nnet(x)
    s1 = x[0]
    s2 = x[1]
    #将产生的音频信号写入.txt文件
    def write_elements(file, tensor):
        for i, item in enumerate(tensor):
            tensor_array = item.detach().numpy()
            for element in tensor_array:
                file.write(str(element) + ", ")
            if (i + 1) % 2 == 0:
                file.write("\n")

    with open("output.txt1", "w") as file:
        write_elements(file, s1)

    with open("output.txt2", "w") as file:
        write_elements(file, s2)
if __name__ == "__main__":
    test_conv_tasnet()
