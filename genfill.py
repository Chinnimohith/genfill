import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

image_url = "https://www.shutterstock.com/image-photo/guy-standing-on-dock-by-260nw-2340765645.jpg"
image_response = requests.get(image_url)
image = Image.open(BytesIO(image_response.content))
    

if not isinstance(image, Image.Image):
    raise ValueError("The image is not in the correct format. Please ensure it is a PIL.Image.Image.")

image = image.convert("RGB")

width, height = image.size
mask = Image.new("L", (width, height), 0)
mask.paste(255, (width // 4, height // 4, 3 * width // 4, 3 * height // 4)) 

image_np = np.array(image)
mask_np = np.array(mask)

inpainted_image = cv2.inpaint(image_np, mask_np, 3, cv2.INPAINT_TELEA)

inpainted_image_pil = Image.fromarray(inpainted_image)

inpainted_image_pil.show()  
