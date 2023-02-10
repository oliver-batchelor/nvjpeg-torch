#include <vector>
#include <cassert>
#include <memory>
#include <iostream>

#include <torch/extension.h>
 
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
 
#include "nvjpeg2k.h" 


class Jpeg2kException : public std::exception {
  nvjpeg2kStatus_t code;
  std::string context;

  public:
    Jpeg2kException(std::string const& _context, nvjpeg2kStatus_t _code) :
      code(_code), context(_context)
    { }
        
    const char * what () const throw () {
      std::stringstream ss;
      ss << context << ", nvjpeg2k error " << code << ": " << error_string(code);
      return ss.str().c_str();

    }
};


nvjpeg2kImage_t createImage(torch::Tensor const& data, nvjpegInputFormat_t input_format, size_t &width, size_t &height) const {
  TORCH_CHECK(data.is_cuda(), "Input image should be on CUDA device");
  TORCH_CHECK(data.dtype() == torch::kI16, "Input image should be uint8 or int16");
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

nvjpeg2kImageType_t dataType(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3, "expected 3D tensor (C, H, W)");

  if image.dtype == torch::kU8:
    return NVJPEG2K_UINT8;
  else if image.dtype == torch::kU16:
    return NVJPEG2K_UINT16;
  else:
    throw Jpeg2kException(NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED, "Unsupported pixel type");
}


nvjpeg2kImageType_t precision(torch::Tensor const& image) {

  if image.dtype == torch::kU8:
    return 8
  else if image.dtype == torch::kU16:
    return 16
  else:
    throw Jpeg2kException(NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED, "Unsupported pixel type");
}


nvjpeg2kImage_t interleavedImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, 
    "for interleaved (BGRI, RGBI) expected 3D tensor (H, W, C)");


  nvjpeg2kImage_t img; 
  nvjpeg2kImageComponentInfo_t image_comp_info[3];
  unsigned char *pixel_data[3];
  size_t *pitch_in_bytes[3];


  precision_bits = precision(image);
  precision_bytes = precision_bits / 8;
  for (int c = 0; c < 3; c++)
  {
    image_comp_info[c].component_width  = image.size(1);
    image_comp_info[c].component_height = image.size(0);
    image_comp_info[c].precision        = precision_bits;
    image_comp_info[c].sgn              = 0;
  }


  size_t plane_stride = at::stride(image, 0) * precision_bytes;

  for(int i = 0; i < 3; i++) {
    pitch_in_bytes[i] = (unsigned int)at::stride(image, 1) * precision_bytes;
    pixel_data[i] = (unsigned char*)image.data_ptr() + (plane_stride * i);
  }


  return img;
}


nvjpeg2kImage_t planarImage(torch::Tensor const& image) {
  TORCH_CHECK(image.dim() == 3 && image.size(0) == 3, 
    "for planar (BGR, RGB) expected 3D tensor (C, H, W)");

  nvjpeg2kImage_t img; 
  unsigned char *pixel_data[NUM_COMPONENTS];
  size_t *pitch_in_bytes[NUM_COMPONENTS];

  precision_bytes = precision(image) / 8;
  size_t plane_stride = at::stride(image, 0) * precision_bytes;

  for(int i = 0; i < 3; i++) {
    pitch_in_bytes[i] = (unsigned int)at::stride(image, 1) * precision_bytes;
    pixel_data[i] = (unsigned char*)image.data_ptr() + (plane_stride * i);
  }

  input_image.pixel_data = pixel_data;
  input_image.pixel_type = dataType(image);
  input_image.pitch_in_bytes = plane_stride * i;

  return img;
}

inline const char* error_string(nvjpeg2kStatus_t code) {
  switch(code) {
    case NVJPEG2K_STATUS_SUCCESS: return "success";
    case NVJPEG2K_STATUS_NOT_INITIALIZED: return "not initialized";
    case NVJPEG2K_STATUS_INVALID_PARAMETER: return "invalid parameter";
    case NVJPEG2K_STATUS_BAD_JPEG: return "bad jpeg";
    case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED: return "not supported";
    case NVJPEG2K_STATUS_ALLOCATOR_FAILURE: return "allocation failed";
    case NVJPEG2K_STATUS_EXECUTION_FAILED: return "execution failed";
    case NVJPEG2K_STATUS_ARCH_MISMATCH: return "arch mismatch";
    case NVJPEG2K_STATUS_INTERNAL_ERROR: return "internal error";
    default: return "unknown";
  }
}



inline void check_nvjpeg2k(std::string const &message, nvjpeg2kStatus_t code) {
  if (NVJPEG2K_STATUS_SUCCESS != code){
      throw Jpeg2kException(message, code);
  }
}



class Jpeg2kCoder {
  public:

  Jpeg2kCoder() {
    nvjpeg2kCreateSimple(&nv_handle);
    nvjpeg2kEncoderStateCreate(&enc_state);
  }

  ~Jpeg2kCoder() {
    nvjpeg2kJpeg2kStateDestroy(nv_statue);
    nvjpeg2kEncoderStateDestroy(enc_state);
    nvjpeg2kDestroy(nv_handle);
  }

  inline nvjpeg2kEncoderParams_t createParams(int width, int height, int psnr) {
    nvjpeg2kEncoderParams_t params;

    nvjpeg2kEncodeParamsCreate(nv_handle, &params);
    nvjpeg2kEncodeParamsSetQuality(params, psnr);

    nvjpeg2kEncodeConfig_t enc_config;
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type      =  NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
    enc_config.color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    enc_config.image_width      =  width;
    enc_config.image_height     =  height;
    enc_config.num_components   =  3;
    enc_config.image_comp_info  =  &image_comp_info;
    enc_config.code_block_w     =  64;
    enc_config.code_block_h     =  64;
    enc_config.irreversible     =  0
    enc_config.mct_mode         =  1;
    enc_config.prog_order       =  NVJPEG2K_LRCP;
    enc_config.num_resolutions  =  1;

    nvjpeg2kStatus_t status = nvjpeg2kEncodeParamsSetEncodeConfig(params, &enc_config);

    return params;
  }



  torch::Tensor encode(torch::Tensor const& data, int psnr = 35, nvjpeg2kInputFormat_t input_format = NVJPEG2K_INPUT_BGRI, nvjpeg2kChromaSubsampling_t subsampling = NVJPEG2K_CSS_422) {
    py::gil_scoped_release release;
    size_t width, height;

    nvjpeg2kImage_t image = createImage(data, input_format, width, height);
    nvjpeg2k2kEncodeParams_t params = createParams(width, height, psnr);


    check_nvjpeg2k("nvjpeg2kEncodeImage", 
      nvjpeg2kEncodeImage(nv_handle, enc_state, params, &image, nullptr));


    size_t length;
    nvjpeg2kEncodeRetrieveBitstream(nv_handle, enc_state, NULL, &length, nullptr);
    auto buffer = torch::empty({ int(length) }, torch::TensorOptions().dtype(torch::kUInt8));

    nvjpeg2kEncodeRetrieveBitstream(nv_handle, enc_state, (unsigned char*)buffer.data_ptr(), &length, nullptr);
    nvjpeg2kEncoderParamsDestroy(params);

    return buffer;
  }


  nvjpeg2k2kEncoder_t nv_handle;
  nvjpeg2k2kEncodeState_t enc_state;
};

  

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto jpeg = py::class_<Jpeg2kCoder>(m, "Jpeg2k");

  py::register_exception<Jpeg2kException>(m, "Jpeg2kException");

  jpeg.def(py::init<>())
      .def("encode", &Jpeg2kCoder::encode)
      .def("__repr__", [](const Jpeg2kCoder &a) { return "Jpeg2k"; });
  

  py::enum_<nvjpeg2kInputFormat_t>(jpeg, "InputFormat")
    .value("BGR", nvjpeg2kInputFormat_t::NVJPEG2K_INPUT_BGR)
    .value("RGB", nvjpeg2kInputFormat_t::NVJPEG2K_INPUT_RGB)
    .value("BGRI", nvjpeg2kInputFormat_t::NVJPEG2K_INPUT_BGRI)
    .value("RGBI", nvjpeg2kInputFormat_t::NVJPEG2K_INPUT_RGBI)
    .export_values();
}
