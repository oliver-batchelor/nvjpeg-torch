
from os import path
import numpy as np

import cv2

import argparse

import torch
from nvjpeg_torch import Jpeg2k



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')


  args = parser.parse_args()
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  jpeg = Jpeg2k()

  data = jpeg.encode(torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda())

  filename = path.join("out", path.splitext(args.filename)[0] + ".jp2")
  with open(filename, "wb") as f:
    f.write(data.cpu().numpy())

  print(f"Wrote {data.shape[0]} bytes to {filename}")

  
