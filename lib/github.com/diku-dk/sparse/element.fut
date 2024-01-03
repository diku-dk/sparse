-- | Module type for elements of sparse matrices.
--
-- We need to be able to determine whether and element is zero, and
-- perhaps perform some other operations on it.

-- | Element type and required operations for a sparse matrix. This
-- module type is satisfied by standard arithmetic modules such as
-- `i32`@term and `f64`@term.
module type element = {
  type t
  val i64 : i64 -> t
  val * : t -> t -> t
  val + : t -> t -> t
  val - : t -> t -> t
  val < : t -> t -> bool
}
