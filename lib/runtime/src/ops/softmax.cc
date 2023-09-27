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

#include "softmax.h"
#include "kernels/softmax_kernels.h"
#include "utils/exceptions.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Softmax;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE };

/* Params */
bool operator==(SoftmaxParams const &lhs, SoftmaxParams const &rhs) {
  return lhs.dim == rhs.dim;
}

bool SoftmaxParams::is_valid(ParallelTensorShape const &input) const {
  return input.is_valid();
}

SoftmaxParams Softmax::get_params() const {
  SoftmaxParams params;
  params.dim = this->dim;
  return params;
}

OpTaskInvocation init(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(HANDLE, ff_handle());
  return {SOFTMAX_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PER_DEVICE_STATE,
                   per_device_op_state<SoftmaxPerDeviceState>());
  binding.bind_arg(PROFILING, profiling_settings());

  binding.bind(INPUT, input_parallel_tensor_shape(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {SOFTMAX_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(SoftmaxAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {SOFTMAX_BWD_TASK_ID, binding};
}

static DeviceSpecific<SoftmaxPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  DeviceSpecific<SoftmaxPerDeviceState> per_device_state =
      acc.create_device_specific<SoftmaxPerDeviceState>(init_kernel(handle));
  return per_device_state;
}

static DeviceSpecific<SoftmaxPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  auto input = acc.get_tensor<Permissions::RO>(A_INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state =
      acc.get_argument<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[SoftMax] forward_time = %.2lfms\n",
                 per_device_state,
                 input.get_float_ptr(),
                 output.get_float_ptr(), );
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);

  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  assert(input_grad.shape == input.shape);

  auto output_grad = acc.get_tensor<Permissions::RO>(OUTPUT);
  auto output = acc.get_tensor<Permissions::RO>(OUTPUT);
  assert(output_grad.shape == output.shape);

  return profile(
      backward_kernel,
      profiling,
      "[SoftMax] backward_time = %.2lfms\n",
      input_grad.get_float_ptr(),
      output_grad.get_float_ptr(),
      output_grad.shape.volume(), // Note(lambda): get num_elements, maybe wrong
  );
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  SoftmaxAttrs const &attrs,
                                  InputParallelTensorDesc const &input,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input.shape);

  SimTaskBinding init_binding;
  // Note: what should init_binding?
  init_binding.bind(INPUT, input);
  init_binding.bind(OUTPUT, output_shape);
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(PROFILING, settings);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(SOFTMAX_INIT_TASK_ID, init_binding);
  DeviceSpecific<SoftmaxPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;
  fwd_binding.bind(INPUT, input);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(SOFTMAX_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(SOFTMAX_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
}

template <>
void register_task<SOFTMAX_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  // Note: we don't add_input(INPUT) and add_output_slot(OUTPUT) here, because
  // init_task_impl doesn't need input, output, just need PerDeviceFFHandle
  register_task(SOFTMAX_INIT_TASK_ID, "SoftMax Init", init, init_task);
}

template <>
void register_task<SOFTMAX_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<bool>(PROFILING);
  fwd.add_unchecked_arg_slot<SoftmaxPerDeviceState>(PER_DEVICE_STATE);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(SOFTMAX_FWD_TASK_ID, "SoftMax Fwd", fwd, forward_task);
}

template <>
void register_task<SOFTMAX_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(SOFTMAX_FWD_TASK_ID));

  register_task(SOFTMAX_BWD_TASK_ID, "SoftMax Bwd", bwd, backward_task);
}

}; // namespace FlexFlow
