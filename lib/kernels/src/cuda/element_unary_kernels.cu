/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/datatype_dispatch.h"
#include "kernels/element_unary_kernels.h"
#include "device.h"
#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace ElementUnary {


template <typename T>
__global__ void elewise_unary_forward_kernel(
    size_t volume, const optional<T> scalar, OperatorType type, T const *in, T *out) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case Op::EXP: {
        out[i] = (T)exp((float)in[i]);
        break;
      }
      case Op::IDENTITY: {
        out[i] = in[i];
        break;
      }
      case Op::SCALAR_MULTIPLY: {
        out[i] = in[i] * scalar;
        break;
      }
      case Op::SCALAR_ADD: {
        out[i] = in[i] + scalar;
        break;
      }
      case Op::SCALAR_SUB: {
        out[i] = in[i] - scalar;
        break;
      }
      case Op::SCALAR_TRUE_DIV: {
        out[i] = in[i] / scalar;
        break;
      }
      case Op::GELU: {
        out[i] = (T)(in[i] * 0.5 * erfc(-in[i] * M_SQRT1_2));
        break;
      }
      case Op::RSQRT: {
        out[i] = (T)(1.0f / sqrt((float)in[i]));
        break;
      }
      case Op::POW: {
        out[i] = (T)(powf(in[i], scalar));
        break;
      }
      case Op::SIN: {
        out[i] = (T)sin((float)in[i]);
        break;
      }
      case Op::COS: {
        out[i] = (T)cos((float)in[i]);
        break;
      }
      default:
        assert(false);
    }
  }
}

template <typename T>
__global__ void elewise_unary_backward_kernel(size_t volume,
                                              const optional<T> scalar,
                                              OperatorType type,
                                              T const *output,
                                              T const *output_grad,
                                              T const *input,
                                              T *input_grad) {
  CUDA_KERNEL_LOOP(i, volume) {
    switch (type) {
      case Op::EXP: {
        // TODO: change to use output instead of recomputing
        input_grad[i] += (T)(output_grad[i] * exp((float)input[i]));
        break;
      }
      case Op::IDENTITY: {
        input_grad[i] += output_grad[i];
        break;
      }
      case Op::SCALAR_MULTIPLY: {
        input_grad[i] += output_grad[i] * scalar;
        break;
      }
      case Op::SCALAR_ADD: {
        input_grad[i] += output_grad[i];
        break;
      }
      case Op::SCALAR_SUB: {
        input_grad[i] += output_grad[i];
        break;
      }
      case Op::SCALAR_TRUE_DIV: {
        input_grad[i] += output_grad[i] / scalar;
        break;
      }
      case Op::GELU: {
        input_grad[i] =
            (T)(output_grad[i] *
                (0.5 * erfc(-input[i] * M_SQRT1_2) -
                 0.5 * M_SQRT1_2 * input[i] * exp(-input[i] * input[i] * 0.5)));
        break;
      }
      case Op::RSQRT: {
        input_grad[i] =
            (T)(-0.5f * output_grad[i] * output[i] * output[i] * output[i]);
        break;
      }
      case Op::POW: {
        input_grad[i] =
            (T)(output_grad[i] * scalar * powf(input[i], scalar - 1));
        break;
      }
      case Op::SIN: {
        input_grad[i] += (T)(output_grad[i] * cos((float)input[i]));
        break;
      }
      case Op::COS: {
        input_grad[i] += (T)(output_grad[i] * -sin((float)input[i]));
        break;
      }
      default:
        assert(false);
    }
  }
}


static bool use_cudnn(OperatorType op_type) {
  switch (op_type) {
    case Op::RELU:
    case Op::SIGMOID:
    case Op::TANH:
    case Op::ELU:
      return true;
    default:
      return false;
  }
}

ElementUnaryPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                       OperatorType op_type,
                                       ArrayShape input_shape,
                                       ArrayShape output_shape) {

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffActivationDescriptor_t actiDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));

  if (use_cudnn(op_type)) {
    cudnnActivationMode_t mode;
    switch (op_type) {
      case Op::SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case Op::RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case Op::TANH:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      case Op::ELU:
        mode = CUDNN_ACTIVATION_ELU;
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(
        actiDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(
        cudnnSetTensorDescriptorFromArrayShape(inputTensor, input_shape));
    checkCUDNN(
        cudnnSetTensorDescriptorFromArrayShape(outputTensor, output_shape));
  }

  ElementUnaryPerDeviceState per_device_state = {handle,
                                                 inputTensor,
                                                 outputTensor,
                                                 actiDesc};
  return per_device_state;
}

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    OperatorType op_type,
                    optional<float> scalar) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));
  if (use_cudnn(op_type)) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(m.handle.dnn,
                                      m.actiDesc,
                                      &alpha,
                                      m.inputTensor,
                                      input.get_float_ptr(),
                                      &beta,
                                      m.outputTensor,
                                      output.get_float_ptr()));
  } else {
    size_t num_elements = input.shape.num_elements();
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements),
                                    CUDA_NUM_THREADS,
                                    0,
                                    stream>>>(num_elements,
                                              scalar,
                                              op_type,
                                              input.get_float_ptr(),
                                              output.get_float_ptr());
  }
}

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &m,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &input_grad,
                     GenericTensorAccessorW const &output,
                     GenericTensorAccessorW const &output_grad,
                     OperatorType op_type,
                     optional<float> scalar) {
  checkCUDNN(cudnnSetStream(m.handle.dnn, stream));
  if (use_cudnn(op_type)) {
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnActivationForward(m.handle.dnn,
                                      m.actiDesc,
                                      &alpha,
                                      m.inputTensor,
                                      input.get_float_ptr(),
                                      &beta,
                                      m.outputTensor,
                                      output.get_float_ptr()));
  } else {
    size_t num_elements = input.shape.num_elements();
    elewise_unary_forward_kernel<<<GET_BLOCKS(num_elements),
                                    CUDA_NUM_THREADS,
                                    0,
                                    stream>>>(num_elements,
                                              scalar,
                                              op_type,
                                              input.get_float_ptr(),
                                              output.get_float_ptr());
  }
}

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow
