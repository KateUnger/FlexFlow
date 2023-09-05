#ifndef _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H

#include "kernels/device.h"
#include <cstddef>
#include "kernels/ff_handle.h"
#include "kernels/allocation.h"
#include "kernels/array_shape.h"

namespace FlexFlow {

struct DropoutPerDeviceState {
public:
  //Have values when passed in
  PerDeviceFFHandle handler;
  float rate;
  unsigned long long seed;
  ArrayShape output_domain;
  Allocator allocator;

  // Realm::RegionInstance reserveInst;
  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t outputTensor;
  ffDropoutDescriptor_t dropoutDesc;

  void *reserveSpace;
  void *dropoutStates;
  size_t reserveSpaceSize;
  req<size_t> dropoutStateSize;
};

FF_VISITABLE_STRUCT_NO_EQ(DropoutPerDeviceState,
                          handler,
                          rate,
                          seed,
                          output_domain,
                          allocator,
                          reserveInst,
                          inputTensor,
                          outputTensor,
                          dropoutDesc,
                          reserveSpace,
                          dropoutStates,
                          reserveSpaceSize,
                          dropoutStateSize);

namespace Kernels {
namespace Dropout {

DropoutPerDeviceState init_kernel(PerDeviceFFHandle handler,
                                  float rate,
                                  unsigned long long seed,
                                  ArrayShape const &output_domain,
                                  Allocator allocator);

void forward_kernel(ffStream_t stream,
                    DropoutPerDeviceState *m,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(ffStream_t stream,
                     DropoutPerDeviceState *m,
                     float const *output_grad_ptr,
                     float *input_grad_ptr);

} // namespace Dropout
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_DROPOUT_KERNELS_H
