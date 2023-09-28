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
#include "kernels/embedding_kernels.h"
#include "kernels/device.h"
#include "device.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow {
namespace Kernels {
namespace Embedding {

__global__ void embed_forward_no_aggr(
    int64_t const *input, int64_t *output, int64_t const *embed, int out_dim, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
    output[i] = embed[wordIdx * out_dim + off];
  }
}

__global__ void embed_forward_with_aggr(int64_t const *input,
                                        int64_t *output,
                                        int64_t const *embed,
                                        int out_dim,
                                        int in_dim,
                                        int batch_size,
                                        AggregateOp aggr) {
  int64_t scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    output[i] = 0;
    int idx = i / out_dim;
    int off = i % out_dim;
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      output[i] = output[i] + embed[wordIdx * out_dim + off];
      if (aggr == AggregateOp::SUM) {
      } else {
        assert(aggr == AggregateOp::AVG);
        output[i] = output[i] * scale;
      }
    }
  }
}

__global__ void embed_backward_no_aggr(
    int64_t const *input, int64_t const *output, int64_t *embed, int out_dim, int batch_size) {
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    int64_t wordIdx = input[idx];
    atomicAdd(int(embed + wordIdx * out_dim + off), int(output[i]));
  }
}

__global__ void embed_backward_with_aggr(int64_t const *input,
                                         int64_t const *output,
                                         int64_t *embed,
                                         int out_dim,
                                         int in_dim,
                                         int batch_size,
                                         AggregateOp aggr) {
  TD scale = 1.0f / in_dim;
  CUDA_KERNEL_LOOP(i, batch_size * out_dim) {
    int idx = i / out_dim;
    int off = i % out_dim;
    TD gradient;
    if (aggr == AggregateOp::SUM) {
      gradient = output[i];
    } else {
      assert(aggr == AggregateOp::AVG);
      gradient = output[i] * scale;
    }
    for (int j = 0; j < in_dim; j++) {
      int64_t wordIdx = input[idx * in_dim + j];
      atomicAdd(embed + wordIdx * out_dim + off, gradient);
    }
  }
}


void forward_kernel(cudaStream_t stream,
                    AggregateOp aggr,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const &weight,
                    int in_dim,
                    int out_dim,
                    int batch_size) {
  assert(input.data_type == DT_INT32 || input.data_type == DT_INT64);
  assert(weight.data_type == DT_HALF || weight.data_type == DT_FLOAT || weight.data_type == DT_DOUBLE);

  if (aggr == AggregateOp::NONE) {
    embed_forward_no_aggr(input.get_int64_ptr(),
                          output.get_int64_ptr(),
                          weight.get_int64_ptr(),
                          out_dim,
                          batch_size);
  } else {
    assert(aggr == AggregateOp::AVG || aggr == AggregateOp::SUM);
    embed_forward_with_aggr(input.get_int64_ptr(),
                            output.get_int64_ptr(),
                            weight.get_int64_ptr(),
                            out_dim,
                            in_dim,
                            batch_size,
                            aggr);
  }
}

void backward_kernel(cudaStream_t stream,
                     AggregateOp aggr,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &weight_grad,
                     int in_dim,
                     int out_dim,
                     int batch_size) {
  assert(input.data_type == DT_INT32 || input.data_type == DT_INT64);
  assert(output.data_type == DT_HALF || output.data_type == DT_FLOAT || output.data_type == DT_DOUBLE);

  if (aggr == AggregateOp::NONE) {
      embed_backward_no_aggr(input.get_int64_ptr(),
                              output.get_int64_ptr(),
                              weight_grad.get_int64_ptr(),
                              out_dim,
                              batch_size);
  } else {
    embed_backward_with_aggr(input.get_int64_ptr(),
                              output.get_int64_ptr(),
                              weight_grad.get_int64_ptr(),
                              out_dim,
                              in_dim,
                              batch_size,
                              aggr);
  }
}

} // namespace Embedding
} // namespace Kernels
} // namespace FlexFlow
