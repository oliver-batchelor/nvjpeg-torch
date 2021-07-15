
from os import path
import numpy as np

import cv2

import argparse

import torch
from nvjpeg_torch import Jpeg, write_file



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')


  args = parser.parse_args()
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)

  jpeg = Jpeg()

  data = jpeg.encode(torch.from_numpy(image).cuda())

  filename = path.join("out", path.basename(args.filename))
  write_file(filename, data)
