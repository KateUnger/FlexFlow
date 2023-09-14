#ifndef _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_ELEMENT_UNARY_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include <cstddef>
#include "op-attrs/op.h"

namespace FlexFlow {

struct ElementUnaryPerDeviceState {
  PerDeviceFFHandle handle;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  req<ffActivationDescriptor_t> actiDesc;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ElementUnaryPerDeviceState,
                                             handle,
                                             inputTensor,
                                             outputTensor,
                                             actiDesc);

namespace Kernels {
namespace ElementUnary {

ElementUnaryPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                       OperatorType op_type,
                                       ArrayShape input_shape,
                                       ArrayShape output_shape);

void forward_kernel(ffStream_t stream,
                    ElementUnaryPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    OperatorType op_type,
                    optional<float> scalar);

void backward_kernel(ffStream_t stream,
                     ElementUnaryPerDeviceState const &m,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &input_grad,
                     GenericTensorAccessorW const &output,
                     GenericTensorAccessorW const &output_grad,
                     OperatorType op_type,
                     optional<float> scalar);

} // namespace ElementUnary
} // namespace Kernels
} // namespace FlexFlow

#endif
