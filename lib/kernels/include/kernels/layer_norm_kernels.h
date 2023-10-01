#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

struct LayerNormPerDeviceState { //todo delete commented out code
  // bool elementwise_affine; //attrs
  // int64_t effective_batch_size, effective_num_elements; //input
  // float eps; //attrs
  float *mean, *rstd, *ds, *db, *scale, *bias;
  // DataType data_type; //input.shape
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(LayerNormPerDeviceState, mean, rstd, ds, db, scale, bias);

namespace Kernels {
namespace LayerNorm {

LayerNormPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                    bool elementwise_affine,
                                    int64_t batch_size,
                                    int64_t num_elements,
                                    bool profiling,
                                    float eps);

void forward_kernel(ffStream_t stream,
                    LayerNormPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta,
                    DataType data_type);

void backward_kernel(ffStream_t stream,
                     LayerNormPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &gamma,
                     GenericTensorAccessorW const &gamma_grad,
                     GenericTensorAccessorW const &beta_grad,
                     DataType data_type);

} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
