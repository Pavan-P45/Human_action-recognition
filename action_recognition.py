from collections import deque
import numpy as np
import cv2

# -----------------------------
# Parameters
# -----------------------------
class Parameters:
    def __init__(self):

        # Path to action labels
        self.CLASSES = open("Actions.txt").read().strip().split("\n")

        # Path to ResNet model
        self.ACTION_MODEL = "resnet-34_kinetics.onnx"

        # Number of frames required for prediction
        self.SAMPLE_DURATION = 16

        # Input size for model
        self.SAMPLE_SIZE = 112


param = Parameters()

# Frame buffer
captures = deque(maxlen=param.SAMPLE_DURATION)

print("[INFO] Loading Human Action Recognition Model...")
net = cv2.dnn.readNet(param.ACTION_MODEL)

print("[INFO] Starting Webcam...")
vs = cv2.VideoCapture(0)

while True:

    grabbed, frame = vs.read()

    if not grabbed:
        break

    frame = cv2.resize(frame, (550, 400))
    captures.append(frame)

    # Wait until we have 16 frames
    if len(captures) < param.SAMPLE_DURATION:

        cv2.putText(frame, "Collecting Frames...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Human Action Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # Convert frames to blob
    blob = cv2.dnn.blobFromImages(
        captures,
        1.0,
        (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
        (114.7748, 107.7354, 99.4750),
        swapRB=True,
        crop=True
    )

    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    net.setInput(blob)
    outputs = net.forward()

    label = param.CLASSES[np.argmax(outputs)]

    # Draw label on frame
    cv2.rectangle(frame, (0, 0), (350, 40), (255,255,255), -1)
    cv2.putText(frame, label, (10,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,0),
                2)

    cv2.imshow("Human Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()