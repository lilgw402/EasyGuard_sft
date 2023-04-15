# -*- coding: utf-8 -*-
import torchaudio
from .ecapatdnn import ECAPA_TDNN


def processor(sr=16000, filepath=None):
    if filepath is not None:
        audio, sr = torchaudio.load(filepath)

    if sr != self.sampling_rate:
        print(f'source sr: {sr}')
        audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
    audio = audio.squeeze(0).numpy()
    length = self.num_frames * 160 + 240
    if audio.shape[0] <= length:
        padding_length = length - audio.shape[0]
        audio = np.pad(audio, (0, padding_length), 'wrap')
    start_frame = int(random.random() * (audio.shape[0] - length))  # 截断
    audio = audio[start_frame: start_frame + length]
    audio = torch.from_numpy(audio).float()

    return audio


def cos_sim(e1, e2):
    e1 = F.normalize(e1, dim=1)
    e2 = F.normalize(e2, dim=1)
    score = torch.mm(e1, e2.t())
    return score


svenocder = ECAPA_TDNN(C=1024, hidden_dim=192)

if __name__ == '__main__':
    pass
