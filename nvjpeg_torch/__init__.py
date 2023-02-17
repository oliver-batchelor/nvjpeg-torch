import torch
import nvjpeg_cuda
import nvjpeg2k_cuda


Subsampling = nvjpeg_cuda.Jpeg.Subsampling

class Jpeg():
  BGR = nvjpeg_cuda.Jpeg.InputFormat.BGR
  RGB = nvjpeg_cuda.Jpeg.InputFormat.RGB

  BGRI = nvjpeg_cuda.Jpeg.InputFormat.BGRI
  RGBI = nvjpeg_cuda.Jpeg.InputFormat.RGBI

  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  def encode(self, image, quality=90, input_format=BGRI, subsampling=Subsampling.CSS_422):
    return self.jpeg.encode(image, quality, input_format, subsampling)



class Jpeg2k():

  def __init__(self):
    self.jpeg = nvjpeg2k_cuda.Jpeg2k()

  def encode(self, image, psnr=40):
    return self.jpeg.encode(image, psnr)