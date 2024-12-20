from matplotlib import animation
import numpy as np 
import matplotlib.pyplot as plt
from utils.data_utils import is_numpy_event_data, is_pyg_event_data, pyg2numpy_event_convertor

def animate_events(events, transform, fig_size=None):
    
    if is_pyg_event_data(events):
        events = pyg2numpy_event_convertor(events)
    else:
        assert is_numpy_event_data(events), "The input data must be a structured numpy array or a PyG dataset."
    
    frames = transform(events)
    if fig_size is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig, ax = plt.subplots(1,1, figsize=fig_size)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 0, ...] = frames[:, 0, ...] > 0
        rgb[:, 2, ...] = frames[:, 1, ...] > 0
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    im = ax.imshow(frames[0])
    # plt.axis("off")
    # plt.axis("tight")

    def animate(frame):
        im.set_data(frame)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.close(fig)

    return anim
        