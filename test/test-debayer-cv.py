import torch
from kornia import color
import cv2
import argparse
import time

from structs.torch import shape_info, shape
from tqdm import tqdm

from nvjpeg_torch import Jpeg


jpeg = Jpeg()

class CudaArrayInterface:
    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        c = gpu_mat.channels()
        type_map = {
            cv2.CV_8U: "u1", cv2.CV_8S: "i1",
            cv2.CV_16U: "u2", cv2.CV_16S: "i2",
            cv2.CV_32S: "i4",
            cv2.CV_32F: "f4", cv2.CV_64F: "f8",
        }
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": (h, w, c),
            "data": (gpu_mat.cudaPtr(), False),
            "typestr": type_map[gpu_mat.depth()],
            "strides": (gpu_mat.step, gpu_mat.elemSize(), 1),
        }


  

def debayer(image, stream=None):
  return cv2.cuda.demosaicing(image, cv2.cuda.COLOR_BayerRG2RGB_MHT, stream=stream)


def cv_to_torch(image):
  return torch.asarray(CudaArrayInterface(image), device=torch.device("cuda"))


def debayer_encode(image, stream=None):
  rgb = debayer(image, stream=stream)
  return jpeg.encode(cv_to_torch(rgb), quality=95)



class Timer:     
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def bench(name, f, images):
  print("Warmup..")
  stream = cv2.cuda_Stream()

  for image in tqdm(images[:len(images) // 4]):
    f(image, stream=stream)

  stream.waitForCompletion()


  print("Benchmark..")
  with Timer() as t:
    for image in tqdm(images):
      f(image, stream=stream)

    stream.waitForCompletion()
  
  rate = len(images) / t.interval
  print(f"{name}: {rate:.2f} images/sec")



def from_torch(image):
  return image.to(torch.uint8).squeeze(0).cpu().numpy()

def display(image):
  cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
  
  cv2.imshow("image", image)
  cv2.resizeWindow("image", 1024, 768)              # Resize window to specified dimensions

  cv2.waitKey()
  
def bgr_to_bayerRG(image):
  output = image[..., 1:2] # green pixels

  output[..., ::2, ::2,   :] = image[..., ::2, ::2,   0:1]  # red
  output[..., 1::2, 1::2, :] = image[..., 1::2, 1::2, 2:3]  # blue

  return output


def load_as_bayer(filename):
  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  return bgr_to_bayerRG(image)


def main(args):
  bayer = load_as_bayer(args.filename)
  gpu = cv2.cuda_GpuMat(bayer)
  images = [gpu] * args.n
      
  bench("debayer", debayer, images)
  bench("debayer_encode", debayer_encode, images)

  rgb = debayer(gpu).download()

  if args.show:
    print(f"Debayered {shape_info(rgb)}")
    display(rgb)

  data = debayer_encode(gpu)
  
  data = data.cpu().numpy()
  with open("test.jpg", "wb") as f:
    f.write(data.tobytes())

  image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
  if args.show:
    print(f"Debayered encode/decoded {shape(image)}")
    display(image)
  
    
  

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Debayer test.')
  parser.add_argument('filename', type=str, help='filename of image to use')
  parser.add_argument('--n', type=int, default=100, help='number of trials')
  parser.add_argument('--show', default=False, action="store_true", help='number of trials')


  args = parser.parse_args()
  with torch.inference_mode():
    main(args)
 