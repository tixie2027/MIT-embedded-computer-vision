import cv2
import torch
import matplotlib.pyplot as plt
import time

def capture_images(num=8, warmup_frames=5, secs_bw_cap=0.1):
    cap = cv2.VideoCapture(0)
    
    # warm-up the camera
    for _ in range(warmup_frames):
        cap.read()
    
    images = []
    for i in range(num):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame {i}")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # needs to be divisible by 16 for compressai model
        h, w, _ = frame.shape
        new_h = (h + 15) // 16 * 16 
        new_w = (w + 15) // 16 * 16 
        frame = cv2.resize(frame, (new_w, new_h))

        # convert to tensor: [C, H, W]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        images.append(tensor.unsqueeze(0))  # [1, C, H, W]

        time.sleep(secs_bw_cap)
    
    cap.release()
    #print(images)
    return torch.cat(images, dim=0)  # [N, C, H, W]





