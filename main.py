import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()  # Libera o cache da GPU
from ultralytics import YOLO

if __name__ == '__main__':
    # Verifica se a GPU está disponível
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Carrega o modelo
    model = YOLO("yolov8n.pt")  # ou outro modelo desejado

    # Move o modelo para o dispositivo correto
    device = 'cpu'
    model.to(device)

    # Treina o modelo
    results = model.train(data="config.yaml", epochs=500, device=device, batch=2,patience=50)



# import torch
# from ultralytics import YOLO
#
# # Verifica se a GPU está disponível
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')
#
# # Carrega o modelo
# model = YOLO("yolov8n-seg.yaml")  # ou outro modelo desejado
#
# # Move o modelo para o dispositivo correto
# model.to(device)
#
# # Treina o modelo
# results = model.train(data="config.yaml", epochs=300, device=device)

#       Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
#     300/500     0.499G     0.5039      1.075     0.5184     0.9706          3        640: 100%|██████████| 70/70 [00:16<00:00,  4.31it/s]
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 20/20 [00:02<00:00,  6.93it/s]
#                    all         77        105      0.889      0.793      0.894      0.737      0.889      0.793      0.888      0.646
#   0%|          | 0/70 [00:00<?, ?it/s]
#       Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
#     301/500     0.497G     0.5362      1.054     0.5068     0.9814          1        640: 100%|██████████| 70/70 [00:16<00:00,  4.33it/s]
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 20/20 [00:02<00:00,  6.90it/s]
#                    all         77        105        0.9      0.802        0.9      0.746        0.9      0.802      0.896      0.656
# Stopping training early as no improvement observed in last 150 epochs. Best results observed at epoch 151, best model saved as best.pt.
# To update EarlyStopping(patience=150) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
#
# 301 epochs completed in 1.673 hours.
# Optimizer stripped from runs\segment\train30\weights\last.pt, 6.8MB
# Optimizer stripped from runs\segment\train30\weights\best.pt, 6.8MB
#
# Validating runs\segment\train30\weights\best.pt...
# Ultralytics YOLOv8.1.11 🚀 Python-3.9.2 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce GTX 1650, 4096MiB)
# YOLOv8n-seg summary (fused): 195 layers, 3258454 parameters, 0 gradients, 12.0 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 20/20 [00:02<00:00,  7.42it/s]
#                    all         77        105      0.835      0.883      0.944      0.784      0.835      0.883      0.943      0.698
#                class_0         77         55      0.772      0.945      0.944      0.796      0.772      0.945      0.942      0.683
#                class_1         77         50      0.897       0.82      0.945      0.772      0.897       0.82      0.945      0.713
# Speed: 0.4ms preprocess, 22.0ms inference, 0.0ms loss, 2.2ms postprocess per image
# Results saved to runs\segment\train30
#
# Process finished with exit code 0


















#      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
#     148/500     0.392G     0.9996     0.6097      1.185         12        640: 100%|██████████| 227/227 [00:40<00:00,  5.58it/s]
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 44/44 [00:04<00:00,  9.43it/s]
#                    all        173        401      0.957      0.855      0.952      0.603
# Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 98, best model saved as best.pt.
# To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
#
# 148 epochs completed in 1.880 hours.
# Optimizer stripped from runs\detect\train44\weights\last.pt, 6.2MB
# Optimizer stripped from runs\detect\train44\weights\best.pt, 6.2MB
#
# Validating runs\detect\train44\weights\best.pt...
# Ultralytics YOLOv8.1.11 🚀 Python-3.9.2 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce GTX 1650, 4096MiB)
# Model summary (fused): 168 layers, 3006038 parameters, 0 gradients, 8.1 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 44/44 [00:04<00:00, 10.98it/s]
#                    all        173        401       0.95      0.848      0.969      0.664
#              Comedouro        173        401       0.95      0.848      0.969      0.664
# Speed: 0.4ms preprocess, 15.6ms inference, 0.0ms loss, 1.7ms postprocess per image
# Results saved to runs\detect\train44
#
# Process finished with exit code 0
