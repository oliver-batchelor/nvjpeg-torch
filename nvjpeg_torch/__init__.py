import torch
import nvjpeg_cuda

from nvjpeg_cuda import write_file, JpegException

Subsampling = nvjpeg_cuda.Jpeg.Subsampling

class Jpeg():
  BGR = nvjpeg_cuda.Jpeg.InputFormat.BGR
  RGB = nvjpeg_cuda.Jpeg.InputFormat.RGB

  BGRI = nvjpeg_cuda.Jpeg.InputFormat.BGRI
  RGBI = nvjpeg_cuda.Jpeg.InputFormat.RGBI

  Exception = JpegException

  def __init__(self):
    self.jpeg = nvjpeg_cuda.Jpeg()

  def encode(self, image, quality=90, input_format=BGRI, subsampling=Subsampling.CSS_422):
    return self.jpeg.encode(image, quality, input_format, subsampling)


