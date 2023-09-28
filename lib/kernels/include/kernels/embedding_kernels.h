#ifndef _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow {
namespace Kernels {
namespace Embedding {
void forward_kernel(ffStream_t stream,
                    AggregateOp aggr,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &output,
                    GenericTensorAccessorR const &weight,
                    size_t in_dim,
                    size_t out_dim,
                    int batch_size);
void backward_kernel(ffStream_t stream,
                     AggregateOp aggr,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorR const &weight_grad,
                     size_t in_dim,
                     size_t out_dim,
                     int batch_size);

void rand_generate_int64_wrapper(int64_t *ptr, size_t size, int64_t p);
void rand_generate_int32_wrapper(int32_t *ptr, size_t size, int32_t p);

template <typename TD>
__global__ void rand_generate_int(TD *ptr, size_t size, TD p);

} // namespace Embedding
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_EMBEDDING_KERNELS_H
