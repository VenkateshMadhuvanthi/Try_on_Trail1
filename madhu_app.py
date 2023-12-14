import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import random
machine_number = 0
model = os.path.join(os.path.dirname(__file__), "models/eva/Eva_0.png")
def remove_edges(image):
      # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, threshold1=10, threshold2=20)

    # Find the contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the white color
    # Define range of white color in HSV
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([255, 20, 255], dtype=np.uint8)

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert mask to get black as the color to remove
    mask_inv = cv2.bitwise_not(mask)

    # Make the background transparent
    # Convert the image to RGBA (add an alpha channel)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_rgba[:, :, 3] = mask_inv

MODEL_MAP = {
    "AI Model Rouyan_0": 'models/rouyan_new/Rouyan_0.png',
    "AI Model Rouyan_1": 'models/rouyan_new/Rouyan_1.png',
    "AI Model Rouyan_2": 'models/rouyan_new/Rouyan_2.png',
    "AI Model Eva_0": 'models/eva/Eva_0.png',
    "AI Model Eva_1": 'models/eva/Eva_1.png',
    "AI Model Simon_0": 'models/simon_online/Simon_0.png',
    "AI Model Simon_1": 'models/simon_online/Simon_1.png',
    "AI Model Xuanxuan_0": 'models/xiaoxuan_online/Xuanxuan_0.png',
    "AI Model Xuanxuan_1": 'models/xiaoxuan_online/Xuanxuan_1.png',
    "AI Model Xuanxuan_2": 'models/xiaoxuan_online/Xuanxuan_2.png',
    "AI Model Yaqi_0": 'models/yaqi/Yaqi_0.png',
    "AI Model Yaqi_1": 'models/yaqi/Yaqi_1.png',
    "AI Model Yaqi_2": 'models/yaqi/Yaqi_2.png',
    "AI Model Yaqi_3": 'models/yaqi/Yaqi_3.png',
    "AI Model Yifeng_0": 'models/yifeng_online/Yifeng_0.png',
    "AI Model Yifeng_1": 'models/yifeng_online/Yifeng_1.png',
    "AI Model Yifeng_2": 'models/yifeng_online/Yifeng_2.png',
    "AI Model Yifeng_3": 'models/yifeng_online/Yifeng_3.png',
}

def add_waterprint(img):
    h, w, _ = img.shape
    img = cv2.putText(img, 'Powered by MixNMatch', (int(0.3*w), h-20), cv2.FONT_HERSHEY_PLAIN, 2, (128, 128, 128), 2, cv2.LINE_AA)
    return img

def overlay_images(background, overlay, position=(0, 0)):
    # Convert images to appropriate color formats
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)

    # Position of overlay
    x, y = position

    # Resize overlay image to fit the model image
    bg_h, bg_w, bg_channels = background.shape
    ol_h, ol_w, ol_channels = overlay.shape

    # Ensure overlay isn't larger than background
    if ol_w + x > bg_w or ol_h + y > bg_h:
        scale_factor = min((bg_w - x) / ol_w, (bg_h - y) / ol_h)
        new_size = (int(ol_w * scale_factor), int(ol_h * scale_factor))
        overlay = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)
        ol_h, ol_w, _ = overlay.shape

    # Overlaying the images
    for i in range(ol_h):
        for j in range(ol_w):
            if overlay[i, j][3] != 0:  # alpha 0 is transparent
                background[y + i, x + j] = overlay[i, j][:3]

    return cv2.cvtColor(background, cv2.COLOR_RGB2BGR)

def get_tryon_result(model_path, garment_top_path, garment_bottom_path, output_path, seed=1234):
    # Load the images
    model_img = cv2.imread(model_image_path)
    garment_top_img = cv2.imread(garment_top_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(garment_top_image_path) else None
    garment_top_img = remove_edges(garment_top_img)
    garment_bottom_img = cv2.imread(garment_bottom_image_path, cv2.IMREAD_UNCHANGED) if os.path.exists(garment_bottom_image_path) else None
    garment_bottom_img = remove_edges(garment_bottom_img)
    # Get the dimensions of the image
    model_img_height, model_img_width = model_img.shape[:2]
    # Assume these are the dimensions of the area on the model image where the garments should be placed
    shoulder_position = (model_img_width // 2, model_img_height // 4)  # Center of the model's shoulders
    waist_position = (model_img_width // 2, model_img_height // 2)     # Center of the model's waist

    if garment_top_img is not None:
        # Calculate top garment position, assuming we want to center it horizontally on the model
        # and position it a certain distance from the top of the model image
        top_x = (model_img_width - garment_top_img.shape[1]) // 2
        top_y = int(model_img_height * 0.25)  # Adjust this factor based on where the garment should sit
        model_img = overlay_images(model_img, garment_top_img, position=(top_x, top_y))

    if garment_bottom_img is not None:
        # Calculate bottom garment position, assuming we want to center it horizontally
        # and place it a certain distance from the top, perhaps around the waist area
        bottom_x = (model_img_width - garment_bottom_img.shape[1]) // 2
        bottom_y = int(model_img_height * 0.5)  # Adjust this factor based on where the pants should sit
        model_img = overlay_images(model_img, garment_bottom_img, position=(bottom_x, bottom_y))

    final_img = add_waterprint(model_img)
    cv2.imwrite(output_path, final_img)
   

# Paths to the images
output_path ="/content/OutfitAnyone/madhu_test.png"
model_image_path = "/content/OutfitAnyone/models/eva/Eva_0.png"
garment_top_image_path = "/content/OutfitAnyone/garments/top111.png"
garment_bottom_image_path = "/content/OutfitAnyone/garments/bottom5.png" # This can be optional


# Gradio interface setup
with gr.Blocks() as demo:
    # Run button
    run_button = gr.Button("Try On")

    # Display the result
    result_label = gr.Label("Your virtual try-on result")
    get_tryon_result(model_image_path, garment_top_image_path, garment_bottom_image_path,output_path, seed=1234)
    gallery = gr.Image(label="Result")
if __name__ == "__main__":
    demo.launch()
