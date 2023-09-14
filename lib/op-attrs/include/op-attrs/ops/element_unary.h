#ifndef _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H
#define _FLEXFLOW_ELEMENTARY_UNARY_ATTRS_H

#include "core.h"
#include "op-attrs/op.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ElementUnaryAttrs {
  optional<float> scalar;
  req<Op> op;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ElementUnaryAttrs, scalar, op);
CHECK_VALID_OP_ATTR(ElementUnaryAttrs);

} // namespace FlexFlow

#endif
