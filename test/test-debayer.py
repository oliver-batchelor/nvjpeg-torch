import torch
from kornia import color
import cv2
import argparse
import time
from torch import nn

from structs.torch import shape_info
from tqdm import tqdm

from nvjpeg_torch import Jpeg

import tensorrt as trt
from torch2trt import torch2trt


device = 'cuda:0'
dtype = torch.float16


def compile(model, input):
  return torch2trt(model, input, 
        fp16_mode=True,   log_level=trt.Logger.INFO, max_workspace_size=1<<28)

class Debayer(nn.Module):
  def __init__(self, image_size, cfa=color.CFA.RG,  device='cuda:0'):
    super(Debayer, self).__init__()

    m = color.RawToRgb(cfa=cfa)
    w, h = image_size

    self.image_size = image_size
    self.debayer = compile(m, [torch.zeros(1, 1, h, w, 
      dtype=torch.float16, device=device)])

  def forward(self, x):
    x = x.to(dtype=torch.float16)

    out = self.debayer(x)
    return out.to(dtype=torch.uint8)


jpeg = Jpeg()

def with_encode(model, input_format=Jpeg.BGR):
  def f(input):
    out = (model(input).squeeze(0)).to(dtype=torch.uint8) 
    data = jpeg.encode(out, input_format=input_format, quality=95)
    return data

  return f



class Timer:     
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def bench_model(name, model, images):
  print("Warmup..")

  for image in tqdm(images):
    model(image)
  torch.cuda.synchronize()

  print("Benchmark..")
  with Timer() as t:
    for image in tqdm(images):
      model(image)

    torch.cuda.synchronize()
  rate = len(images) / t.interval
  print(f"{name}: {rate:.2f} images/sec")



def from_torch(image):
  return image.to(torch.uint8).squeeze(0).cpu().numpy()

def display(image):
  cv2.namedWindow("image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
  
  cv2.imshow("image", image)
  cv2.resizeWindow("image", 1024, 768)              # Resize window to specified dimensions

  cv2.waitKey()
  
def load_as_bayer(filename, dtype=torch.float16, device='cuda:0'):
  image = cv2.imread(filename, cv2.IMREAD_COLOR)
  bgr = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
  bgr = bgr.to(device=device, dtype=dtype)

  return color.rgb_to_raw(color.rgb_to_bgr(bgr), cfa=color.CFA.BG)

def main(args):
  bayer = load_as_bayer(args.filename, dtype=torch.uint8)


  images = [bayer] * args.n

  models = dict(
    kornia = Debayer(
      image_size=(bayer.shape[3], bayer.shape[2]), 
      cfa = color.CFA.RG)
  )
  
  for k, model in models.items():
    model = model.to(dtype=dtype, device=device)

    if args.compile:
      compile(model, bayer.unsqueeze(0))
    
    if args.encode:
      model = with_encode(model)


    bench_model(k, model, images)

    rgb = model(bayer)
    print(shape_info(rgb))

    if args.encode:
      data = rgb.cpu().numpy()
      
      with open("test.jpg", "wb") as f:
        f. write(data.tobytes())

      image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    else:
      image = display(from_torch(rgb))

    if args.show:
      display(image)
        
    
  

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Debayer test.')
  parser.add_argument('filename', type=str, help='filename of image to use')
  parser.add_argument('--n', type=int, default=100, help='number of trials')
  parser.add_argument('--show', default=False, action="store_true", help='number of trials')
  parser.add_argument('--encode', default=False, action="store_true", help='debayer and encode')
  parser.add_argument('--compile', default=False, action="store_true", help='compile debayer with tensorrt')


  args = parser.parse_args()
  with torch.inference_mode():
    main(args)
 