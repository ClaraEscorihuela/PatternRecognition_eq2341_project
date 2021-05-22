import random
from collections import Counter
from typing import List
import pickle

import librosa as librosa
import numpy as np
from os import listdir
from os.path import isfile, join

from PattRecClasses import MarkovChain, HMM, DataAnalysis
from PattRecClasses.HMM import multigaussD, logprob_2, logprob
from PattRecClasses.features import get_features

from matplotlib import pyplot as plt

BITRATE = 22050
N = 5
M = 12
# M = 49
FEATURES = [6]


def trans_one_feature(fs: np.ndarray, f: int) -> np.array:
    fs_extracted = []
    for f in fs[f]:
        if not fs_extracted or fs_extracted[-1] != f and f:
            fs_extracted.append(f)
    return np.array(2 * fs_extracted)


def trans_feature(fs: np.ndarray, f: List[int]) -> np.ndarray:
    # 2: Silences
    # 3: Real silences
    # 6: Semitone difference
    # 7: Unlimited semitone difference

    fs_extracted = [[] for _ in range(len(fs))]
    in_silence = False
    tmp = [[0] for _ in range(len(fs))]
    for i in range(len(fs[0])):
        for fx in f:
            fs_extracted[fx].append(fs[fx][i])
        # if not fs[3][i]:
        #     if fs[2][i]:
        #         if not in_silence:
        #             for fx in f:
        #                 tmp[fx] = fs[fx][i]
        #             in_silence = True
        #         else:
        #             for fx in f:
        #                 tmp[fx] += fs[fx][i]
        #     else:
        #         if in_silence:
        #             in_silence = False
        #             for fx in f:
        #                 fs_extracted[fx].append(tmp[fx] + fs[fx][i])
        #             tmp = [[0] for _ in range(len(fs))]
        #         else:
        #             for fx in f:
        #                 fs_extracted[fx].append(fs[fx][i])
    fs_extracted = [f_ex for i, f_ex in enumerate(fs_extracted) if i in f]
    fs_extracted_2 = [[] for _ in range(len(fs_extracted))]
    for i in range(len(fs_extracted[0])):
        for j in range(len(fs_extracted)):
            # if fs_extracted[0][i]:
            fs_extracted_2[j].append(fs_extracted[j][i])
    return np.array(fs_extracted_2)


def normalize(xs: List[float]) -> List[float]:
    xs_sum = sum(xs)
    return [x / xs_sum for x in xs]


def main():
    wav_colours = ["#1954a6", "#c9c8cd", "#da5195"]

    # wav_files = ["test_5.ogg", "melody_2.wav", "melody_3.wav"][:1]
    # wav_files = {"A": ["CSD_ER_alto_1.wav"]}

    try:
        with open("data_wavs.pickle", "rb") as handle:
            wav_recordings = pickle.load(handle)
        print("Read presaved wav data")
    except FileNotFoundError:
        print("Reading new wav data")
        wav_files = {}
        for letter in listdir("Songs"):
            if not isfile(f"Songs/{letter}"):
                wav_files[letter] = {"Train": [], "Test": []}

        for letter in wav_files.keys():
            for f in listdir(f"Songs/{letter}/"):
                if isfile(join(f"Songs/{letter}/", f)):
                    if f.endswith("_t.wav") and not f.endswith("_c_t.wav"):
                        wav_files[letter]["Test"].append(f)
                    elif (
                        not f.endswith("_r.wav")
                        and not f.endswith("_c.wav")
                        and not f.endswith("_c_r.wav")
                    ):
                        wav_files[letter]["Train"].append(f)

        wav_recordings = {
            song: {
                cat: [
                    librosa.load(f"Songs/{song}/{recording}")[0]
                    for recording in recordings
                ]
                for cat, recordings in cats.items()
            }
            for song, cats in wav_files.items()
        }

        with open("data_wavs.pickle", "wb") as handle:
            pickle.dump(wav_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        with open("data_features.pickle", "rb") as handle:
            songs_features = pickle.load(handle)
        print("Read presaved feature data")
    except FileNotFoundError:
        print("Reading new feature data")
        songs_features = {
            song: {
                cat: [get_features(signal=wav, fs=BITRATE) for wav in wavs]
                for cat, wavs in cats.items()
            }
            for song, cats in wav_recordings.items()
        }

        with open("data_features.pickle", "wb") as handle:
            pickle.dump(songs_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # figure = DataAnalysis.data_analysis(songs_features, features=[6, 7])
    # figure.show()

    try:
        with open("data_hmms.pickle", "rb") as handle:
            hmms = pickle.load(handle)
        print("Read pretrained HMM data")
    except FileNotFoundError:
        print("Training new HMM data")
        hmms = {}
        for song, cats in songs_features.items():
            print(f"Initializing song {song} HMM")
            fs = np.concatenate(cats["Train"], axis=1)
            init_obs = trans_one_feature(fs=fs, f=FEATURES[0])
            count = [c2 for c2,_ in sorted(Counter(init_obs).items(), key=lambda c: c[1])][:5]
            N = len(count)

            qstar = np.array(
                normalize([1 / N + random.uniform(-0.2, 0.2) for _ in range(N)])
            )
            Astar = np.array(
                [
                    np.array(
                        normalize(
                            [1 / N + random.uniform(-0.1, 0.1) for _ in range(N)]
                        )
                    )
                    for _ in range(N)
                ]
            )

            meansstar = np.array([[c] for c in count])
            covsstar = np.array([[[1]], [[1]], [[1]], [[1]], [[1]]])
            Bstar = np.array(
                [
                    multigaussD(meansstar[i], covsstar[i])
                    for i in range(N)
                ]
            )
            hmms[song] = HMM(qstar, Astar, Bstar)


            print(f"\tTraining {len(cats['Train'])} samples")
            obs = np.array([np.array([init_obs]).T])
            hmms[song].baum_welch(obs=obs, niter=20, uselog=True)

        with open("data_hmms.pickle", "wb") as handle:
            pickle.dump(hmms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for song, cats in songs_features.items():
        print(f"Testing song {song}")
        for i, song_features in enumerate(cats["Test"]):
            print(f"\tTesting sample {i}")
            obs = np.array([trans_one_feature(fs=song_features, f=FEATURES[0])]).T
            cs_hmms = {letter: hmm.calcabc(obs=obs)[2] for letter, hmm in hmms.items()}
            for letter, cs in cs_hmms.items():
                cs_mean = np.nanmean(
                    np.sum(
                        np.log([a for a in cs if not any([np.isnan(ax) for ax in a])])
                    )
                )
                print(f"\t\t{letter}: {cs_mean}")
            vit_hmms = {letter: hmm.viterbi(obs=obs) for letter, hmm in hmms.items()}
            for letter, vit in vit_hmms.items():
                print(f"\t\t{letter}: {vit}")

    print("a")




if __name__ == "__main__":
    main()
