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

#include "embedding.h"
#include "kernels/embedding_kernels.h"
#include "utils/hash-utils.h"
#include "op-attrs/get_output_shapes.h"
#include "op-attrs/ops/embedding.h"

namespace FlexFlow {

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::Embedding;

enum Slots {
  INPUT,
  OUTPUT,
  WEIGHT,
  ATTRS,
  PROFILING
};

OpTaskInvocation forward(EmbeddingAttrs const &attrs) {
  OpTaskBinding b;
  
  b.bind(INPUT, input_tensor(2));
  b.bind(OUTPUT, output_tensor(0));
  b.bind(WEIGHT, weight_tensor(0));

  b.bind_arg(ATTRS, attrs);
  b.bind_arg(PROFILING, profiling_settings());

  return {EMBED_FWD_TASK_ID, b};
}

OpTaskInvocation backward(EmbeddingAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {EMBED_BWD_TASK_ID, binding};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto weight = acc.get_tensor<Permissions::RO>(WEIGHT);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  return profile(forward_kernel,
                 profiling,
                 "[Embedding] forward_time = %.2lfms\n",
                 attrs.aggr,
                 input,
                 output,
                 weight,
                 input.shape.get_dim(),
                 output.shape.get_dim(),
                 attrs.num_entries);
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto weight_grad = acc.get_tensor_grad<Permissions::RO>(WEIGHT);
  EmbeddingAttrs attrs = acc.get_argument<EmbeddingAttrs>(ATTRS);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  return profile(backward_kernel,
                 profiling,
                 "[Embedding] forward_time = %.2lfms\n",
                 attrs.aggr,
                 input,
                 output,
                 weight_grad,
                 input.shape.get_dim(),
                 output.shape.get_dim(),
                 attrs.num_entries);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim,
                                  EmbeddingAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &mv) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);
  TensorShape weight_shape = get_weights_shape(attrs, get_piece_shape(input.shape));

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input.shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind(WEIGHT, weight_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(ATTRS, attrs);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(EMBED_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(EMBED_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<EMBED_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_input_slot(INPUT);
  fwd.add_input_slot(OUTPUT);
  fwd.add_input_slot(WEIGHT);
  
  fwd.add_arg_slot<EmbeddingAttrs>(ATTRS);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  register_task(
      EMBED_FWD_TASK_ID, "Embed Fwd", fwd, forward_task);
}

template <>
void register_task<EMBED_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(ATTENTION_FWD_TASK_ID));

  register_task(
      EMBED_BWD_TASK_ID, "Embed Bwd", bwd, backward_task);
}


}; // namespace FlexFlow

