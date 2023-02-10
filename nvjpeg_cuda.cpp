#include <vector>
#include <cassert>
#include <memory>
#include <iostream>

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <nvjpeg.h>

nvjpegImage_t interleavedImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, 
    "for interleaved (BGRI, RGBI) expected 3D tensor (H, W, C)");

  nvjpegImage_t img; 

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i] = 0;
  }

  img.pitch[0] = (unsigned int)at::stride(image, 0);
  img.channel[0] = (unsigned char*)image.data_ptr();

  return img;
}


nvjpegImage_t planarImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(0) == 3, 
    "for planar (BGR, RGB) expected 3D tensor (C, H, W)");

  nvjpegImage_t img; 

  for(int i = 0; i < NVJPEG_MAX_COMPONENT; i++){
      img.channel[i] = nullptr;
      img.pitch[i] = 0;
  }

  size_t plane_stride = at::stride(image, 0);

  for(int i = 0; i < 3; i++) {
    img.pitch[i] = (unsigned int)at::stride(image, 1);
    img.channel[i] = (unsigned char*)image.data_ptr() + plane_stride * i;
  }
  
  
  return img;
}


inline const char* error_string(nvjpegStatus_t code) {
  switch(code) {
    case NVJPEG_STATUS_SUCCESS: return "success";
    case NVJPEG_STATUS_NOT_INITIALIZED: return "not initialized";
    case NVJPEG_STATUS_INVALID_PARAMETER: return "invalid parameter";
    case NVJPEG_STATUS_BAD_JPEG: return "bad jpeg";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "not supported";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "allocation failed";
    case NVJPEG_STATUS_EXECUTION_FAILED: return "execution failed";
    case NVJPEG_STATUS_ARCH_MISMATCH: return "arch mismatch";
    case NVJPEG_STATUS_INTERNAL_ERROR: return "internal error";
    default: return "unknown";
  }
}


class JpegException : public std::exception {
  nvjpegStatus_t code;
  std::string context;

  public:
    JpegException(std::string const& _context, nvjpegStatus_t _code) :
      code(_code), context(_context)
    { }
        
    const char * what () const throw () {
      std::stringstream ss;
      ss << context << ", nvjpeg error " << code << ": " << error_string(code);
      return ss.str().c_str();

    }
};

inline void check_nvjpeg(std::string const &message, nvjpegStatus_t code) {
  if (NVJPEG_STATUS_SUCCESS != code){
      throw JpegException(message, code);
  }
}



class JpegCoder {
  public:

  JpegCoder() {
    nvjpegCreateSimple(&nv_handle);
    nvjpegJpegStateCreate(nv_handle, &nv_state);
    nvjpegEncoderStateCreate(nv_handle, &enc_state, NULL);
  }

  ~JpegCoder() {
    nvjpegJpegStateDestroy(nv_state);
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

  nvjpegImage_t createImage(torch::Tensor const& data, nvjpegInputFormat_t input_format, size_t &width, size_t &height) const {
    TORCH_CHECK(data.is_cuda(), "Input image should be on CUDA device");
    TORCH_CHECK(data.dtype() == torch::kU8, "Input image should be uint8");
    TORCH_CHECK(data.is_contiguous(), "Input data should be contiguous");

    bool interleaved = input_format == NVJPEG_INPUT_BGRI || input_format == NVJPEG_INPUT_RGBI;

    if(interleaved) {
      width = data.size(1);
      height = data.size(0);
      return interleavedImage(data);
    } else {
      width = data.size(2);
      height = data.size(1);
      return planarImage(data);
    }
  }


  torch::Tensor encode(torch::Tensor const& data, int quality = 90, nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGRI, nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_422) {
    py::gil_scoped_release release;
    size_t width, height;

    nvjpegEncoderParams_t params = createParams(quality, subsampling);
    nvjpegImage_t image = createImage(data, input_format, width, height);

    check_nvjpeg("nvjpegEncodeImage", 
      nvjpegEncodeImage(nv_handle, enc_state, params, 
        &image, input_format, width, height, nullptr));

    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, NULL, &length, nullptr);
    auto buffer = torch::empty({ int(length) }, torch::TensorOptions().dtype(torch::kUInt8));

    nvjpegEncodeRetrieveBitstream(nv_handle, enc_state, (unsigned char*)buffer.data_ptr(), &length, nullptr);
    nvjpegEncoderParamsDestroy(params);

    return buffer;
  }


  nvjpegHandle_t nv_handle;
  nvjpegJpegState_t nv_state;
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

  py::register_exception<JpegException>(m, "JpegException");

  jpeg.def(py::init<>())
      .def("encode", &JpegCoder::encode)
      .def("__repr__", [](const JpegCoder &a) { return "Jpeg"; });
  
  py::enum_<nvjpegChromaSubsampling_t>(jpeg, "Subsampling")
    .value("CSS_444", nvjpegChromaSubsampling_t::NVJPEG_CSS_444)
    .value("CSS_422", nvjpegChromaSubsampling_t::NVJPEG_CSS_422)
    .value("CSS_GRAY", nvjpegChromaSubsampling_t::NVJPEG_CSS_GRAY)
    .export_values();

  py::enum_<nvjpegInputFormat_t>(jpeg, "InputFormat")
    .value("BGR", nvjpegInputFormat_t::NVJPEG_INPUT_BGR)
    .value("RGB", nvjpegInputFormat_t::NVJPEG_INPUT_RGB)
    .value("BGRI", nvjpegInputFormat_t::NVJPEG_INPUT_BGRI)
    .value("RGBI", nvjpegInputFormat_t::NVJPEG_INPUT_RGBI)
    .export_values();

  
  m.def("write_file", &write_file);
}
