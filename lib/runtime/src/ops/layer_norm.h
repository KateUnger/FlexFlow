#ifndef _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H
#define _FLEXFLOW_RUNTIME_SRC_OPS_LAYER_NORM_H

#include "op-attrs/ops/layer_norm.h"
#include "op_task_invocation.h"
#include "sim_environment.h"

namespace FlexFlow {

template <>
void register_task<LAYERNORM_INIT_TASK_ID>();
template <>
void register_task<LAYERNORM_FWD_TASK_ID>();
template <>
void register_task<LAYERNORM_BWD_TASK_ID>();

OpTaskInvocation init(LayerNormAttrs const &);
OpTaskInvocation forward(LayerNormAttrs const &);
OpTaskInvocation backward(LayerNormAttrs const &);

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  LayerNormAttrs const &,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view);

} // namespace FlexFlow

#endif
