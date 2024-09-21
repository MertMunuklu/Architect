import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



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

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    labels = labels.flatten()
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)
input_point = np.array(select_point(image))
input_label = np.array([1])


plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  


masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)