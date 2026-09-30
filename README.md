# Human Action Recognition

A computer vision project for recognizing human actions from video using a pretrained ResNet-34 model trained on the Kinetics dataset.

## 📌 Project Overview

This project uses a pretrained ResNet-34 ONNX model for human action recognition. The model processes video frames and predicts the corresponding human action.

## ✨ Features

- Human action recognition from video
- Pretrained ResNet-34 model
- ONNX model format
- Kinetics dataset-based action recognition

## 📸 Screenshot

![Human Action Recognition](screenshots/human-action-recognition.png)

## 🧠 Model

The project uses the pretrained:

**ResNet-34 Kinetics ONNX model**

Download the pretrained model from the ONNX Model Zoo:

[Download ResNet-34 Kinetics Model](https://github.com/onnx/models/tree/main/vision/action_recognition)

After downloading, place the model in the project directory as:

`resnet-34_kinetics.onnx`

## 🛠️ Technologies

- Python
- ResNet-34
- ONNX
- Computer Vision
- Kinetics Dataset

## 📂 Project Structure

HAR_Project/
│
├── resnet-34_kinetics.onnx
├── screenshots/
│   └── human-action-recognition.png
└── README.md