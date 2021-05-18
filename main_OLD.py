import librosa as librosa
import numpy as np

from PattRecClasses import MarkovChain, HMM
from PattRecClasses.features import get_features

BITRATE = 22050
N = 10
M = 12
# M = 49


def main():
    wav_colours = ["#1954a6", "#c9c8cd", "#da5195"]
    # wav_files = ["test_5.ogg", "melody_2.wav", "melody_3.wav"][:1]
    wav_files = {"A": ["melody_1.wav"], "B": ["melody_3.wav"]}
    hmms = {}
    wav_recordings = {song: [librosa.load(f"Songs/{recording}")[0] for recording in recordings] for song, recordings in wav_files.items()}
    songs_features = {song: [get_features(signal=wav, fs=BITRATE) for wav in wavs] for song, wavs in wav_recordings.items()}

    for song, song_features in songs_features.items():
        if not song in hmms.keys():
            hmms[song] = HMM(mc=MarkovChain(n=N, m=M))

        for features in song_features:
            obs = [int(features[0][i]) for i in range(len(features[0])) if not features[2][i]]
            hmms[song].mc.baum_welch(obs=obs)

    wav = librosa.load(f"Songs/melody_2.wav")[0]
    features = get_features(signal=wav, fs=BITRATE)
    obs = [int(features[0][i]) for i in range(len(features[0])) if not features[2][i]]
    _, c_ts = hmms["A"].mc.forward(obs)
    print("LOG A:", hmms["A"].logprob(c_ts=c_ts))
    _, c_ts = hmms["B"].mc.forward(obs)
    print("LOG B:", hmms["B"].logprob(c_ts=c_ts))


if __name__ == "__main__":
    main()
