from typing import Optional

import numpy as np

from PattRecClasses.GetMusicFeatures import GetMusicFeatures

N_SEMITONES = 12
N_SILENCES = 5
N_MELS = 1024
N_FREQ = 440 / 135  # 3.23
OCTAVE_FREQUENCIES = [0, 220, 440, 880, 1760, 3520, 7040, 14080]

WINDOW_LENGTH = 0.06
WINDOW_TEMPO_LENGTH = 10


class Tone:
    f: float
    i: Optional[float]
    semitone: int
    octave: int

    def __init__(self, f: float, i: Optional[float] = None):
        self.f = f
        self.i = i
        self.semitone, self.octave = self.get_note_and_octave()

    def get_note_and_octave(self):
        octave = 0
        while octave < len(OCTAVE_FREQUENCIES) and self.f >= OCTAVE_FREQUENCIES[octave]:
            octave += 1
        octave -= 1
        f_0 = OCTAVE_FREQUENCIES[octave]
        f_1 = OCTAVE_FREQUENCIES[octave + 1]
        note = int(N_SEMITONES * ((self.f - f_0) / (f_1 - f_0)))
        return note, octave

    def get_frequency(self):
        return (
            OCTAVE_FREQUENCIES[self.octave]
            + self.semitone
            * (OCTAVE_FREQUENCIES[self.octave + 1] - OCTAVE_FREQUENCIES[self.octave])
            / 12
        )

    def is_silent(self):
        return self.get_frequency() > 400 or (self.i < 0.025 if self.i else False)


def get_features(signal: np.array, fs: int) -> np.array:
    fs = GetMusicFeatures(signal=signal, fs=fs, winlength=WINDOW_LENGTH)
    tones = [Tone(f=fs[0][j], i=fs[2][j]) for j in range(len(fs[0]))]
    tone_frequencies = [t.get_frequency() for t in tones]

    tempos = []
    for i in range(len(tone_frequencies) - WINDOW_TEMPO_LENGTH):
        tempos.append(len(set(tone_frequencies[i : i + WINDOW_TEMPO_LENGTH])))
    for i in range(WINDOW_TEMPO_LENGTH):
        tempos.append(tempos[-1])

    intervals_diff = []
    semitones_diff = []
    for i in range(len(tones) - 1):
        t_0 = tones[i]
        t_1 = tones[i + 1]
        intervals_diff.append(t_1.semitone - t_0.semitone)
        semitones_diff.append(
            N_SEMITONES * t_1.octave
            + t_1.semitone
            - (N_SEMITONES * t_0.octave + t_0.semitone)
        )
    intervals_diff.append(intervals_diff[-1])
    semitones_diff.append(semitones_diff[-1])

    silences = [t.is_silent() for t in tones]
    silences_all = []
    for i in range(len(silences) - N_SILENCES):
        silences_all.append(
            len([s for s in silences[max(i - N_SILENCES, 0) : i + N_SILENCES - 1] if s])
            >= N_SILENCES
        )
    for i in range(N_SILENCES):
        silences_all.append(silences_all[-1])

    return np.array(
        [
            np.array(
                [
                    t.semitone,
                    t.octave,
                    silences[i],
                    silences_all[i],
                    fs[2][i],
                    tempos[i],
                    intervals_diff[i],
                    semitones_diff[i],
                ]
            )
            for i, t in enumerate(tones)
        ]
    ).T
