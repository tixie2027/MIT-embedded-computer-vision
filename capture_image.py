import subprocess
import torch
import cv2
import time
import os

def capture_images(num=8, secs_bw_cap=0.1):
    images = []

    for i in range(num):
        filename = f"/tmp/frame_{i}.jpg"
        subprocess.run([
            "libcamera-still",
            "-n",  # no preview
            "-o", filename,
            "--width", "640", "--height", "480",
            "--timeout", "100"
        ])

        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pad to multiple of 16 for CompressAI
        h, w, _ = frame.shape
        new_h = (h + 15) // 16 * 16
        new_w = (w + 15) // 16 * 16
        frame = cv2.resize(frame, (new_w, new_h))

        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        images.append(tensor.unsqueeze(0))

        time.sleep(secs_bw_cap)

    return torch.cat(images, dim=0)  # [N, C, H, W]






