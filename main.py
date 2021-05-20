from collections import Counter
from typing import List

import librosa as librosa
import numpy as np
from os import listdir
from os.path import isfile, join

from PattRecClasses import MarkovChain, HMM
from PattRecClasses.HMM import multigaussD, logprob_2, logprob
from PattRecClasses.features import get_features

from matplotlib import pyplot as plt

BITRATE = 22050
N = 10
M = 12
# M = 49


def trans_feature(fs: np.ndarray, f: List[int]) -> np.ndarray:
    # 2: Silences
    # 3: Real silences
    # 6: Semitone difference
    # 7: Unlimited semitone difference

    fs_extracted = [[] for _ in range(len(fs))]
    in_silence = False
    tmp = [[0] for _ in range(len(fs))]
    for i in range(len(fs[0])):
        if not fs[3][i]:
            if fs[2][i]:
                if not in_silence:
                    for fx in f:
                        tmp[fx] = fs[fx][i]
                    in_silence = True
                else:
                    for fx in f:
                        tmp[fx] += fs[fx][i]
            else:
                if in_silence:
                    in_silence = False
                    for fx in f:
                        fs_extracted[fx].append(tmp[fx] + fs[fx][i])
                    tmp = [[0] for _ in range(len(fs))]
                else:
                    for fx in f:
                        fs_extracted[fx].append(fs[fx][i])
    fs_extracted = [f_ex for i, f_ex in enumerate(fs_extracted) if i in f]
    fs_extracted_2 = [[] for _ in range(len(fs_extracted))]
    for i in range(len(fs_extracted[0])):
        for j in range(len(fs_extracted)):
            if fs_extracted[0][i]:
                fs_extracted_2[j].append(fs_extracted[j][i])
    return np.array(fs_extracted_2)


def main():
    wav_colours = ["#1954a6", "#c9c8cd", "#da5195"]

    song_class = ['A','B']
    wav_files = {}
    # wav_files = ["test_5.ogg", "melody_2.wav", "melody_3.wav"][:1]
    # wav_files = {"A": ["CSD_ER_alto_1.wav"]}
    for letter in song_class:
        wav_files_element =[f for f in listdir(f"SongTrial{letter}/") if isfile(join(f"SongTrial{letter}/", f))]
        wav_files[f"{letter}"] = wav_files_element

    hmms = {}
    wav_recordings = {song: [librosa.load(f"Songs/{recording}")[0] for recording in recordings] for song, recordings in wav_files.items()}
    songs_features = {song: [get_features(signal=wav, fs=BITRATE) for wav in wavs] for song, wavs in wav_recordings.items()}

    wav_records = open('wav_records.txt', 'wt')
    wav_records.write(str(wav_recordings))
    wav_records.close()

    songs_feat = open('songs_features.txt', 'wt')
    songs_feat.write(str(songs_features))
    songs_feat.close()


    for song, song_features in songs_features.items():
        if not song in hmms.keys():
            qstar = np.array([0.8, 0.2])
            Astar = np.array([[0.5, 0.5], [0.5, 0.5]])
            meansstar = np.array([[0, 0], [0, 0]])
            covsstar = np.array([[[1, 0], [0, 1]],
                                 [[1, 0], [0, 1]]])
            Bstar = np.array([multigaussD(meansstar[0], covsstar[0]),
                      multigaussD(meansstar[1], covsstar[1])])
            hmms[song] = HMM(qstar, Astar, Bstar)

        for features in song_features:
            obs = np.array([trans_feature(fs=features, f=[6, 7]).T])
            hmms[song].baum_welch(obs=obs, niter=100, prin=1, uselog=False)

    # wav = librosa.load(f"Songs/melody_2.wav")[0]
    # features = get_features(signal=wav, fs=BITRATE)

    # count = sorted(list(Counter(features[6]).items()), key=lambda x: x[0])
    # plt.plot([a for a, _ in count], [a for _, a in count])
    # plt.show()
    #
    # count = sorted(list(Counter(features[7]).items()), key=lambda x: x[0])
    # plt.plot([a for a, _ in count], [a for _, a in count])
    # plt.show()

    # obs = trans_feature(fs=features, f=[6, 7]).T
    # a_1, c_log_1 = logprob(x=obs, B=hmms["A"].B)
    # a_2, c_log_2 = logprob(x=obs, B=hmms["B"].B)
    # print("a")


    wav = librosa.load(f"SongsPattern/CSD_ER_alto_1.wav")[0]
    features = get_features(signal=wav, fs=BITRATE)
    obs = trans_feature(fs=features, f=[6, 7]).T
    _, _, cs_a = hmms["A"].calcabc(obs=obs)
    _, _, cs_b = hmms["B"].calcabc(obs=obs)
    print("Melody 1")
    print(np.nanmean(np.sum(np.log([a for a in cs_a if not any([np.isnan(ax) for ax in a])]))))
    print(np.nanmean(np.sum(np.log([b for b in cs_b if not any([np.isnan(bx) for bx in b])]))))

    wav = librosa.load(f"SongsPattern/CSD_ER_alto_2.wav")[0]
    features = get_features(signal=wav, fs=BITRATE)
    obs = trans_feature(fs=features, f=[6, 7]).T
    _, _, cs_a = hmms["A"].calcabc(obs=obs)
    _, _, cs_b = hmms["B"].calcabc(obs=obs)
    print("Melody 2")
    print(np.nanmean(np.sum(np.log([a for a in cs_a if not any([np.isnan(ax) for ax in a])]))))
    print(np.nanmean(np.sum(np.log([b for b in cs_b if not any([np.isnan(bx) for bx in b])]))))

    wav = librosa.load(f"SongsPattern/CSD_LI_alto_1.wav")[0]
    features = get_features(signal=wav, fs=BITRATE)
    obs = trans_feature(fs=features, f=[6, 7]).T
    _, _, cs_a = hmms["A"].calcabc(obs=obs)
    _, _, cs_b = hmms["B"].calcabc(obs=obs)
    print("Melody 3")
    print(np.nanmean(np.sum(np.log([a for a in cs_a if not any([np.isnan(ax) for ax in a])]))))
    print(np.nanmean(np.sum(np.log([b for b in cs_b if not any([np.isnan(bx) for bx in b])]))))


    print("a")


if __name__ == "__main__":
    main()
