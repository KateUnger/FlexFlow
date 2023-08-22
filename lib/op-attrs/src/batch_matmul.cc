#include "op-attrs/ops/batch_matmul.h"

namespace FlexFlow {

int get_aSeqLengthDim(BatchMatmulAttrs const &attrs) {
  return attrs.a_seq_length_dim;
}

int get_bSeqLengthDim(BatchMatmulAttrs const &attrs) {
  return attrs.b_seq_length_dim;
}

/* bool BatchMatmulAttrs::is_valid( */
/*     ParallelTensorShape const &lhs, ParallelTensorShape const &rhs) const {
 */
/*   if (!lhs.is_valid() || !rhs.is_valid()) { */
/*     return false; */
/*   } */
/*   if (lhs.num_dims() != rhs.num_dims()) { */
/*     return false; */
/*   } */
/*   for (int i = lhs.num_dims() - 1; i >= 2; i--) { */
/*     if (lhs.at(i) != rhs.at(i)) { */
/*       return false; */
/*     } */
/*   } */
/*   if (lhs.at(0) != rhs.at(1)) { */
/*     return false; */
/*   } */

/*   return true; */
/* } */

} // namespace FlexFlow
