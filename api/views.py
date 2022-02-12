import re
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import numpy as np
import cv2
import torch
import time
# Create your views here.

# load model yolo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path="./model_yolo/best.pt")


@api_view(["POST"])
def index(request):
    if request.method == "POST":
        start_time = time.time()
        img = request.data['file']
        img = img.read()
        img = cv2.imdecode(np.frombuffer(img, np.uint8),
                           cv2.IMREAD_UNCHANGED)[..., ::-1]
        results = model(img)
        data = []
        if len(results.pandas().xyxy[0]) > 0:
            for i in results.pandas().xyxy[0].values:
                data.append(i[6])
        print("Time :", time.time()-start_time)
        return Response({"data": data})
