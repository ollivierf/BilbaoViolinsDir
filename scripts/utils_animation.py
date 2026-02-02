import numpy as np
import  moviepy.editor as mpy
import io 
from PIL import Image

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)
# No Plotly frames are defined here!! Instead we define moviepy frames by
# converting each Plotly figure to  an array, from which MoviePy creates a clip
# The concatenated clips are saved as a gif file:
def make_frame(t):
    z = f(2*np.pi*t/2)
    fig.update_traces(z=z, intensity=z)  #These are the updates that usually are performed within Plotly go.Frame definition
    return plotly_fig2array(fig)

animation = mpy.VideoClip(make_frame, duration=2)
animation.write_gif("image/my_hat.gif", fps=20)