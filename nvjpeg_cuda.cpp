#include <vector>
#include <cassert>
#include <memory>
#include <iostream>

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <nvjpeg.h>


nvjpegImage_t createImage(torch::Tensor const& image) {
  nvjpegImage_t img; 

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i] = 0;
  }

  img.pitch[0] = (unsigned int)at::stride(image, 0);
  img.channel[0] = (unsigned char*)image.data_ptr();

  return img;
}

inline void check_nvjpeg(std::string const &message, nvjpegStatus_t code) {
  if (NVJPEG_STATUS_SUCCESS != code){
      throw std::runtime_error(message + ", error: " + std::to_string(code));
  }
}


class JpegCoder {
  public:

  JpegCoder() {
    nvjpegCreateSimple(&nv_handle);
    nvjpegJpegStateCreate(nv_handle, &nv_statue);
    nvjpegEncoderStateCreate(nv_handle, &enc_state, NULL);
  }

  ~JpegCoder() {
    nvjpegJpegStateDestroy(nv_statue);
    nvjpegEncoderStateDestroy(enc_state);
    nvjpegDestroy(nv_handle);
  }

  inline nvjpegEncoderParams_t createParams(int quality, nvjpegChromaSubsampling_t subsampling, cudaStream_t stream = nullptr) {
    nvjpegEncoderParams_t params;

    nvjpegEncoderParamsCreate(nv_handle, &params, stream);

    nvjpegEncoderParamsSetQuality(params, quality, stream);
    nvjpegEncoderParamsSetOptimizedHuffman(params, 1, stream);
    nvjpegEncoderParamsSetSamplingFactors(params, subsampling, stream);  

    return params;
  }


  torch::Tensor encode(torch::Tensor const& data, int quality = 90, nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_422) {
    TORCH_CHECK(data.is_cuda(), "Input image should be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kU8, "Input image should be uint8");
    TORCH_CHECK(data.dim() == 3, "Input data should be a 3-dimensional tensor (H, W, C)");

    at::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    nvjpegEncoderParams_t params = createParams(quality, subsampling, stream.stream());
    nvjpegImage_t image = createImage(data);

    check_nvjpeg("nvjpegEncodeImage", 
      nvjpegEncodeImage(nv_handle, enc_state, params, 
        &image, NVJPEG_INPUT_BGRI, data.size(1), data.size(0), stream));

    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, NULL, &length, stream.stream());
    auto buffer = torch::empty({ int(length) }, torch::TensorOptions().dtype(torch::kUInt8));

    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, (unsigned char*)buffer.data_ptr(), &length, stream.stream());
    nvjpegEncoderParamsDestroy(params);

    return buffer;
  }


  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_statue;
  nvjpegEncoderState_t enc_state;
};

  
void write_file(const std::string& filename, torch::Tensor& data) {
  TORCH_CHECK(data.device() == torch::kCPU, "Input tensor should be on CPU");
  TORCH_CHECK(data.dtype() == torch::kU8, "Input tensor dtype should be uint8");
  TORCH_CHECK(data.dim() == 1, "Input data should be a 1-dimensional tensor");

  auto fileBytes = data.data_ptr<uint8_t>();
  auto fileCStr = filename.c_str();
  FILE* outfile = fopen(fileCStr, "wb");

  TORCH_CHECK(outfile != nullptr, "Error opening output file");

  fwrite(fileBytes, sizeof(uint8_t), data.numel(), outfile);
  fclose(outfile);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto jpeg = py::class_<JpegCoder>(m, "Jpeg");

  jpeg.def(py::init<>())
      .def("encode", &JpegCoder::encode)
      .def("__repr__", [](const JpegCoder &a) { return "Jpeg"; });
  
  py::enum_<nvjpegChromaSubsampling_t>(jpeg, "Subsampling")
    .value("css_444", nvjpegChromaSubsampling_t::NVJPEG_CSS_444)
    .value("css_422", nvjpegChromaSubsampling_t::NVJPEG_CSS_422)
    .value("css_gray", nvjpegChromaSubsampling_t::NVJPEG_CSS_GRAY)
    .export_values();
  
  m.def("write_file", &write_file);
}
