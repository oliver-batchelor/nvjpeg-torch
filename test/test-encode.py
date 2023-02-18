import os
from os import path
import numpy as np

import cv2
import argparse

import torch
from nvjpeg_torch import Jpeg2k
from nvjpeg_torch import Jpeg

os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'


def display_rgb(image):
  cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  cv2.waitKey(0)



def load_rgb(filename):
  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')
  parser.add_argument('--psnr', default=50, type=int, help='psnr of jpeg2000 encoder')
  parser.add_argument('--quality', default=90, type=int, help='quality of jpeg encoder')


  args = parser.parse_args()
  image = load_rgb(args.filename)

  def decode_rgb(name, data):
    decoded = cv2.imdecode(data.cpu().numpy(), cv2.IMREAD_UNCHANGED)
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    psnr = cv2.PSNR(image, decoded)
    print(f"Encoded {name} as {data.shape[0]} bytes {psnr:.2f}dB")


  jpeg2k = Jpeg2k()
  data = jpeg2k.encode(torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda(), psnr=args.psnr)

  decode_rgb(f"jpeg2k (psnr={args.psnr})", data)


  jpeg = Jpeg()
  data = jpeg.encode(torch.from_numpy(image).contiguous().cuda(), input_format=Jpeg.RGBI, quality=args.quality)

  decode_rgb(f"jpeg (quality={args.quality})", data)


  
