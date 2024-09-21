import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def interactive_crop(imgpath):
    fig, ax = plt.subplots()
    img = mpimg.imread(imgpath)
    ax.imshow(img)

    ref_points = []
    rect = plt.Rectangle((0, 0), 0, 0, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    def on_press(event):
        nonlocal ref_points
        if event.button == 1:
            ref_points = [(event.xdata, event.ydata)]

    def on_move(event):
        nonlocal ref_points
        if event.button == 1 and len(ref_points) == 1:
            x0, y0 = ref_points[0]
            x1, y1 = event.xdata, event.ydata
            rect.set_width(abs(x1 - x0))
            rect.set_height(abs(y1 - y0))
            rect.set_xy((min(x0, x1), min(y0, y1)))
            fig.canvas.draw()

    def on_release(event):
        nonlocal ref_points
        if event.button == 1 and len(ref_points) == 1:
            ref_points.append((event.xdata, event.ydata))
            fig.canvas.mpl_disconnect(cid_press)
            fig.canvas.mpl_disconnect(cid_move)
            fig.canvas.mpl_disconnect(cid_release)
            plt.close()

    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()

    return ref_points

selected_points = interactive_crop(r"C:\Users\mrtmu\Architect\2 (1).jpg")
print("Seçilen Noktalar:", selected_points)

def select_point(img):
    selected_points = []

    def on_press(event):
        if event.button == 1:  
            selected_points.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')  
            fig.canvas.draw()  

    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()

    return selected_points

img = mpimg.imread(r"C:\Users\mrtmu\Architect\2 (1).jpg")
selected_points = select_point(img)
print("Seçilen Noktalar:", selected_points)