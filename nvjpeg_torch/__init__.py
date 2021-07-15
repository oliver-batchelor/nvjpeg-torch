import torch
import nvjpeg_cuda

from nvjpeg_cuda import write_file

subsampling = nvjpeg_cuda.Jpeg.Subsampling

class Jpeg():
  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  def encode(self, image, quality=90, subsampling=subsampling.css_422):
    return self.jpeg.encode(image, quality, subsampling)


