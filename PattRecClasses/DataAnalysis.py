import matplotlib.pyplot as plt
from PattRecClasses.main import trans_feature
import numpy as np

def data_analysis(songs_features, features):
    colors = ["red","green","blue"]

    songs_features_without_zeros = {
        song: {
            cat: [np.array(trans_feature(fs=song_feature, f=features)) for song_feature in wavs]
            for cat, wavs in cats.items()
        }
        for song, cats in songs_features.items()
    }

    cols = ['Song {}'.format(col) for col in ['A']]
    rows = ['Sample {}'.format(row) for row in range(1,2)]


    fig, axes = plt.subplots(nrows=6, ncols=3)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large')


    letter = -1
    for song, cats in songs_features.items():
        letter += 1
        print(f"Testing song {song}...")

        #fig, ax = plt.subplots(len(cats["Train"]))
        for i, song_features in enumerate(cats["Train"][:2]):
            time = np.array(range(0, len(song_features[0])))
            sc = axes[i][letter].scatter(time, song_features[0], c=song_features[1])

    #plt.legend()
    #plt.show()

    #fig.suptitle(f'ScatterPlot{}')
    #plt.tick_params(labelcolor="none", sharex=True, sharey=True, bottom=False, left=False)
    #plt.ylabel('Semitons Difference')
    #plt.xlabel('Note Index')

    return fig