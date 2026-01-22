import os
import shutil

# print(os.listdir('./data/plantvillage dataset/color/Apple___Apple_scab'))

for i in os.listdir('./data/plantvillage dataset/color'):
    for j in os.listdir(f'./data/plantvillage dataset/color/{i}'):
        dir = i
        for k in os.listdir('./data/plantvillage dataset/combined'):
            if dir == k:
                shutil.copy(f'./data/plantvillage dataset/color/{i}/{j}', f'./data/plantvillage dataset/combined/{k}')
    print(f"folder {i} completed")
print("all folders completed")