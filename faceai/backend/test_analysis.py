import os
import json
import base64
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import cv2

client = TestClient(app)

# Create dummy front and side images
front_img = np.zeros((500, 500, 3), dtype=np.uint8)
side_img = np.zeros((500, 500, 3), dtype=np.uint8)

cv2.imwrite("test_front.jpg", front_img)
cv2.imwrite("test_side.jpg", side_img)

with open("test_face.jpg", "rb") as f:
    front_bytes = f.read()

with open("test_face.jpg", "rb") as f:
    side_bytes = f.read()

response = client.post(
    "/api/analyze",
    files={
        "front_image": ("test_front.jpg", front_bytes, "image/jpeg"),
        "side_image": ("test_side.jpg", side_bytes, "image/jpeg"),
    },
    data={"gender": "male"},
)

print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("Forehead:", data.get("forehead"))
    print("Nose:", data.get("nose"))
    print("Mouth:", data.get("mouth"))
    print("Jaw:", data.get("jaw"))
    print("Cheek:", data.get("cheek"))
    print("Eyebrows:", data.get("eyebrows"))
    print("Ears:", data.get("ears"))
else:
    print(response.text)


