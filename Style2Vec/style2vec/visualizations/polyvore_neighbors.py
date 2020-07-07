import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path

em = np.load('./models/amazon_embeding_colors.npy', allow_pickle=True)
imgs = np.load('./models/amazon_paths_colors.npy', allow_pickle=True)


def plot_figures(figures, nrows=15, ncols=15):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 30), dpi=72)

    for ind,path in enumerate(figures):
        img = mpimg.imread(Path("./data/amazon_images/" + imgs[path]))
        axeslist.ravel()[ind].imshow(img)
        axeslist.ravel()[ind].set_axis_off()

    # plt.tight_layout()  # optional


# target images: key in productid
men_office = ["B00BXEF6JC",
              "B00AHCK4AM",
              "B00BZQ7632",
              "B00BW9YUUY",
              "B0000643Q8",
              "B00CO97ZJY",
              "B000G3NDUK"]

# embedding array only
em = [e[0][0] for e in em]
chosen = []
nr = neighbors.NearestNeighbors(metric='cosine', n_neighbors=15).fit(em)

# office
# fixed = [489, 3400, 5693, 11393, 14301, 22461]

# colors
fixed = [23033, 23034, 23035, 23036, 23037, 23038, 23039, 23040, 23041, 23042, 23043]

for i in fixed:
    chosen.append(em[i])

# for i in range(15):
#     chosen.append(random.choice(em))

dist, kn = nr.kneighbors(np.array(chosen))
df_kn = pd.DataFrame(kn)
df_kn.to_csv('./results/knn_15(color).csv')

plot_figures(kn.flatten(), len(fixed))
r = random.randint(100, 999)
plt.savefig("./results/nr" + str(r) + ".png")

plt.show()
