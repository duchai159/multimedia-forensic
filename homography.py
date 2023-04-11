import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir='.', title='Select an image file', filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])
try:
    img = cv2.imread(file_path)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        raise Exception('Invalid image size')
except Exception as e:
    print(f'Error loading image: {str(e)}')

def mouse_callback(event, x, y, flags, param):
    global plate_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(plate_pts) < 4:
        plate_pts.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Original Image', img)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Original Image', mouse_callback)
plate_pts = []
while len(plate_pts) < 4:
    cv2.imshow('Original Image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

warp_pts = np.array([[0, 0], [400, 0], [400, 100], [0, 100]])

H, _ = cv2.findHomography(np.array(plate_pts), warp_pts)

warp_img = cv2.warpPerspective(img, H, (400, 100))

gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)
warp_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
warp_img = cv2.resize(warp_img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

cv2.imshow('Warped Image', warp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

with cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL):
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
