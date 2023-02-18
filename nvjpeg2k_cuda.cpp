#include <vector>
#include <cassert>
#include <memory>
#include <iostream>

#include <torch/extension.h>
 
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
 
#include "nvjpeg2k.h" 


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

class CudaException : public std::exception {
  cudaError_t code;
  std::string context;

  public:
    CudaException(std::string const& _context, cudaError_t _code) :
      code(_code), context(_context)
    { }
        
    const char * what () const throw () {
      std::stringstream ss;
      ss << context << ", CUDA error " << code;
      return ss.str().c_str();
    }
};


inline void check_nvjpeg2k(std::string const &message, nvjpeg2kStatus_t code) {
  if (NVJPEG2K_STATUS_SUCCESS != code){
      throw Jpeg2kException(message, code);
  }
}

inline void check_cuda(std::string const &message, cudaError_t code) {
  if (code != cudaSuccess) { 
    throw CudaException(message, code);
  }   
}

class Jpeg2kImage {
  public:

  nvjpeg2kImageType_t data_type;
  int width, height;

  nvjpeg2kImage_t image; 
  unsigned char *pixel_data[3];
  size_t pitch_in_bytes[3];
  nvjpeg2kImageComponentInfo_t image_comp_info[3];



  int precision_bytes;


  inline void createParams(int psnr, nvjpeg2kEncodeParams_t *params) {

    check_nvjpeg2k("nvjpeg2kEncodeParamsCreate",
      nvjpeg2kEncodeParamsCreate(params));

    check_nvjpeg2k("nvjpeg2kEncodeParamsSetQuality",
      nvjpeg2kEncodeParamsSetQuality(*params, psnr));

    

    nvjpeg2kEncodeConfig_t enc_config;
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type      =  NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
    enc_config.color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    enc_config.image_width      =  width;
    enc_config.image_height     =  height;
    enc_config.num_components   =  3;
    enc_config.image_comp_info  =  image_comp_info;
    enc_config.code_block_w     =  64;
    enc_config.code_block_h     =  64;
    // enc_config.tile_width       =  256;
    // enc_config.tile_height      =  256;
    // enc_config.enable_tiling    = 1;
    enc_config.irreversible     =  1;
    enc_config.mct_mode         =  1;
    enc_config.prog_order       =  NVJPEG2K_LRCP;
    enc_config.num_resolutions  =  6;
    // enc_config.enable_tiling = 1;


    check_nvjpeg2k("nvjpeg2kEncodeParamsSetEncodeConfig",
      nvjpeg2kEncodeParamsSetEncodeConfig(*params, &enc_config));

  }

  Jpeg2kImage(torch::Tensor const& data) {


    TORCH_CHECK(data.is_cuda(), "Input image should be on CUDA device");
    TORCH_CHECK(data.is_contiguous(), "Input data should be contiguous");

    TORCH_CHECK(data.dim() == 3 && data.size(0) == 3, 
      "for planar (BGR, RGB) expected 3D tensor (C, H, W)");

    width = data.size(2);
    height = data.size(1);


    if (data.dtype() == torch::kU8) {
      data_type = NVJPEG2K_UINT8;
      precision_bytes = 1;
    } else if (data.dtype() == torch::kI16) {

      data_type = NVJPEG2K_UINT16;
      precision_bytes = 2;
    }
    else { 
      throw Jpeg2kException("Unsupported pixel type", NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED);
    }


    size_t plane_stride = at::stride(data, 0) * precision_bytes;

    for(int i = 0; i < 3; i++) {
      pitch_in_bytes[i] = (size_t)at::stride(data, 1) * precision_bytes;
      pixel_data[i] = (unsigned char*)data.data_ptr() + (plane_stride * i);

      image_comp_info[i].component_width  = width;
      image_comp_info[i].component_height = height;
      image_comp_info[i].precision        = precision_bytes * 8;
      image_comp_info[i].sgn              = 0;        
    }

    image.pixel_data = (void**)pixel_data;
    image.pixel_type = data_type;
    image.pitch_in_bytes = pitch_in_bytes;
    image.num_components = 3;

  }


};



class Jpeg2kCoder {
  public:

  nvjpeg2kEncoder_t encoder;
  nvjpeg2kEncodeState_t enc_state;

  cudaStream_t stream;

  Jpeg2kCoder() {
    check_nvjpeg2k("nvjpeg2kEncoderCreateSimple",
      nvjpeg2kEncoderCreateSimple(&encoder));

    check_nvjpeg2k("nvjpeg2kEncodeStateCreate",
      nvjpeg2kEncodeStateCreate(encoder, &enc_state));

    check_cuda("cudaStreamCreateWithFlags", 
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  ~Jpeg2kCoder() {

    check_nvjpeg2k("nvjpeg2kEncodeStateDestroy",
      nvjpeg2kEncodeStateDestroy(enc_state));
    check_nvjpeg2k("nvjpeg2kEncoderDestroy",
      nvjpeg2kEncoderDestroy(encoder));
  }





  torch::Tensor encode(torch::Tensor const& data, int psnr = 35) {
    py::gil_scoped_release release;

    Jpeg2kImage image(data);

    nvjpeg2kEncodeParams_t params;
    image.createParams(psnr, &params);

    // std::cout << "Encoding image " << image.width << "x" << image.height << std::endl;
    // std::cout << "Precision: " << image.precision_bytes * 8 << " bits" << std::endl;
    // std::cout << encoder << " " << enc_state <<  std::endl;

    check_nvjpeg2k("nvjpeg2kEncodeImage", 
      nvjpeg2kEncode(encoder, enc_state, params, &image.image, stream));


    size_t length;
    check_nvjpeg2k("nvjpeg2kEncodeRetrieveBitstream", 
      nvjpeg2kEncodeRetrieveBitstream(encoder, enc_state, NULL, &length, stream));
    auto buffer = torch::empty({ int(length) }, torch::TensorOptions().dtype(torch::kUInt8));

    check_nvjpeg2k("nvjpeg2kEncodeRetrieveBitstream", 
      nvjpeg2kEncodeRetrieveBitstream(encoder, enc_state, (unsigned char*)buffer.data_ptr(), &length, stream));

    
    cudaStreamSynchronize(stream);

    check_nvjpeg2k("nvjpeg2kEncodeParamsDestroy", 
      nvjpeg2kEncodeParamsDestroy(params));

    
    return buffer;
  }



};

  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto jpeg = py::class_<Jpeg2kCoder>(m, "Jpeg2k");

  py::register_exception<Jpeg2kException>(m, "Jpeg2kException");

  jpeg.def(py::init<>())
      .def("encode", &Jpeg2kCoder::encode)
      .def("__repr__", [](const Jpeg2kCoder &a) { return "Jpeg2k"; });
  


}
