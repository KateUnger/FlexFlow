#include "element_unary.h"
#include "kernels/element_unary_kernels.h"
#include "legion/legion_utilities.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

using Legion::Context;
using Legion::PhysicalRegion;
using Legion::Runtime;
using Legion::Task;

using namespace FlexFlow::Kernels::ElementUnary;

enum Slots {
  INPUT,
  OUTPUT,
  ATTRS,
  PER_DEVICE_STATE,
  HANDLE,
  PROFILING
};

OpTaskInvocation init(ElementUnaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE, ff_handle());

  return {ELEMENTUNARY_INIT_TASK_ID, binding};
}

OpTaskInvocation forward(ElementUnaryAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<ElementUnaryPerDeviceState>());

  return {ELEMENTUNARY_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(ElementUnaryAttrs const &attrs) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {ELEMENTUNARY_BWD_TASK_ID, b};
}

static DeviceSpecific<ElementUnaryPerDeviceState>
    init_task_impl(TaskArgumentAccessor const &acc) {

  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);
  auto const &attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);

  DeviceSpecific<ElementUnaryPerDeviceState> per_device_state =
      acc.create_device_specific<ElementUnaryPerDeviceState>(
          init_kernel(handle,
                      attrs.op,
                      input.shape,
                      output.shape));

  return per_device_state;
}

static DeviceSpecific<ElementUnaryPerDeviceState>
    init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {

  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto const &attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return profile(forward_kernel,
                 profiling,
                 "[ElementUnary] forward_time = %.2lfms\n",
                 per_device_state,
                 input,
                 output,
                 attrs.op,
                 attrs.scalar);
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
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(INPUT);
  auto const &attrs = acc.get_argument<ElementUnaryAttrs>(ATTRS);

  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto per_device_state = acc.get_argument<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);

  return profile(backward_kernel,
                 profiling,
                 "[ElementUnary] backward_time = %.2lfms\n",
                 per_device_state,
                 input,
                 input_grad,
                 output,
                 output_grad,
                 attrs.op_type,
                 attrs.scalar);
}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  ElementUnaryAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
auto env = sim.new_environment();

  ParallelTensorShape output_shape = get_output_shape(attrs, input_shape.shape);

  SimTaskBinding init_binding;
  init_binding.bind_arg(HANDLE, ff_handle());
  init_binding.bind_arg(ATTRS, attrs);

  auto init_accessor =
      env.get_init_accessor(DROPOUT_INIT_TASK_ID, init_binding);
  DeviceSpecific<DropoutPerDeviceState> per_device_state =
      init_task_impl(init_accessor);

  SimTaskBinding fwd_binding;  
  fwd_binding.bind(INPUT, input_shape);
  fwd_binding.bind(OUTPUT, output_shape);
  fwd_binding.bind_arg(PROFILING, settings);
  fwd_binding.bind_arg(PER_DEVICE_STATE, per_device_state);

  SimTaskBinding bwd_binding = infer_bwd_binding(fwd_binding);

  auto fwd_accessor = env.get_fwd_accessor(DROPOUT_FWD_TASK_ID, fwd_binding);
  auto bwd_accessor = env.get_bwd_accessor(DROPOUT_BWD_TASK_ID, bwd_binding);

  float forward_time = forward_task_impl(fwd_accessor).value();
  float backward_time = backward_task_impl(bwd_accessor).value();

  float sync_time = default_estimate_sync_time(env);
  return make_metrics(forward_time, backward_time, sync_time, env);
  return true;
}


template <>
void register_task<ELEMENTUNARY_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_arg_slot<ElementUnaryAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<ElementUnaryPerDeviceState>();

  register_task(ELEMENTUNARY_INIT_TASK_ID, "ElementUnary Init", init, init_task);
}

template <>
void register_task<ELEMENTUNARY_FWD_TASK_ID>() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_unchecked_arg_slot<ElementUnaryPerDeviceState>(PER_DEVICE_STATE);
  fwd.add_arg_slot<ProfilingSettings>(PROFILING);

  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);

  register_task(ELEMENTUNARY_FWD_TASK_ID, "ElementUnary Fwd", fwd, forward_task);
}

template <>
void register_task<ELEMENTUNARY_BWD_TASK_ID>() {
  OpTaskSignature bwd =
      infer_bwd_signature(get_op_signature(ELEMENTUNARY_FWD_TASK_ID));

  register_task(
      ELEMENTUNARY_BWD_TASK_ID, "ELEMENTUNARY Bwd", bwd, backward_task);
}

}; // namespace FlexFlow
