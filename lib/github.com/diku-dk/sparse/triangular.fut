-- | Sparse triangular matrices.
--
-- A triangular matrix is a square matrix where all elements above or
-- below the diagonal are zero, called respectively *upper* and
-- *lower* triangular matrixes.  While we can always represent an *n²*
-- triangular matrix as an ordinary *n²* matrix where we store the
-- zeroes, this is wasteful of memory.  Instead, we can use a
-- representation where we store only the possibly nonzero elements.
-- This library supports both upper and lower triangular matrices
-- using the same interface, but different concrete types.

import "../linalg/linalg"

-- | The module type of a triangular matrix.  This module type leaves
-- it unstated whether it is an upper or lower triangular matrix, but
-- specific instantiations make it clear.
module type triangular_mat = {
  -- | The scalar type.
  type t
  -- | The type of `n` times `n` triangular matrices.
  type~ mat[n]
  -- | The zero matrix. Given `n`, the function returns an `n`
  -- times `n` empty sparse matrix (zeros everywhere).
  val zero    : (n:i64) -> mat[n]
  -- | Identity matrix. Given `n`, the function returns an `n`
  -- times `n` sparse matrix with ones in the diagonal and zeros
  -- elsewhere.
  val eye     : (n:i64) -> mat[n]
  -- Constructs triangular matrix from dense array.  All elements on
  -- the zero side of the the diagonal is ignored.
  val triangular [n] : [n][n]t -> mat[n]
  -- | Convert to dense format. Given a sparse matrix, the function
  -- returns a dense representation of the matrix.
  val dense [n] : mat[n] -> [n][n]t
  -- | `idx (i,j) m` produces the element at logical position
  -- `(i,j)` in the triangular matrix `m`, returning zero.
  val idx [n] : (i64,i64) -> mat[n] -> t
  -- | Scale elements. Given a matrix and a scale value `v`, the
  -- function returns a new matrix with the elements scaled by `v`.
  val scale [n] : t -> mat[n] -> mat[n]
  -- | Element-wise addition.
  val +     [n] : mat[n] -> mat[n] -> mat[n]
  -- | Element-wise subtraction.
  val -     [n] : mat[n] -> mat[n] -> mat[n]
  -- | Map a function across the elements of the matrix.
  val map [n] 'a 'b : (t -> t) -> mat[n] -> mat[n]
}

-- The number of nonzero elements for triangular `n` by `n` array.
local def elements (n: i64) =
  (n * (1+n))/2

local module type ranking = {
  val rank : (i64, i64) -> i64
  val unrank : i64 -> (i64,i64)
  val zero : (i64,i64) -> bool
}

local module mk_triangular_mat (T : field) (R: ranking) = {
  type t = T.t

  type~ mat [n] =
    ?[nnz].
     { size: [n][0](),
       data: [nnz]t
     }

  def idx [n] 'a (i,j) (tri: mat[n]) =
    if R.zero (i,j) then T.i64 0 else #[unsafe] tri.data[R.rank (i, j)]

  def triangular [n] (arr: [n][n]t) =
    { size = map (const []) arr,
      data = tabulate (elements n)
                      (\p -> let (i,j) = R.unrank p
                             in #[unsafe] arr[i,j])
    }

  def dense [n] (tri: mat[n]) =
    tabulate_2d n n (\i j -> idx (i,j) tri)

  def zero n =
    { size = replicate n [],
      data = replicate (elements n) (T.i64 0)
    }

  def eye n =
    triangular (tabulate_2d n n (\i j -> T.i64 (i64.bool (i==j))))

  def scale s (tri: mat[]) =
    tri with data = map (T.*s) tri.data

  def (+) [n] (x: mat[n]) (y: mat[n]) =
    -- Complicated by the fact that we cannot statically tell that the
    -- two data arrays have same shape.
    let [nnz] (xdata: [nnz]t) = x.data
    let ydata = y.data :> [nnz]t
    in x with data = map2 (T.+) xdata ydata

  def x - y = x + scale (T.i64 (-1)) y

  def map f (tri: mat[]) =
    tri with data = map f tri.data
}

-- Don't worry about the opaque formula - it's essentially just a
-- solution to a certain second-degree equation (I'll admit it's a
-- bit odd to see square roots in index calculations).
local def row (i: i64) =
  i64.f64 (f64.ceil ((f64.sqrt(f64.i64(9+8*i))-1)/2))-1

local module mk_lower_triangular_mat (T: field) =
  mk_triangular_mat T {

  def rank (i, j) =
    elements i + j

  def unrank (p: i64) =
    let i = row p
    let j = p - elements i
    in (i,j)

  def zero (i: i64, j) =
    j > i
}

local module mk_upper_triangular_mat (T: field) =
  mk_triangular_mat T {
  def rank (i, j) =
    elements j + i

  def unrank (p: i64) =
    let i = row p
    let j = p - elements i
    in (j,i)

  def zero (i: i64, j) =
    i > j
}

-- | The type of modules implementing triangular matrices, with
-- distinct submodules and types for lower and upper triangular
-- matrices.
module type triangular = {
  -- | Matrix element type.
  type t
  -- | An upper triangular matrix.
  type~ upper[n]
  -- | A lower triangular matrix.
  type~ lower[n]
  -- | Operations on upper triangular matrices.
  module upper : {
    include triangular_mat with t = t with mat [n] = upper[n]
    -- | Transpose upper triangular matrix, producing lower triangular
    -- matrix.  O(1).
    val transpose [n] : upper[n] -> lower[n]
  }
  -- | Operations on lower triangular matrices.
  module lower : {
    include triangular_mat with t = t with mat [n] = lower[n]
    -- | Transpose lower triangular matrix, producing upper
    -- triangular matrix. O(1).
    val transpose [n] : lower[n] -> upper[n]
  }
}

-- | Create a module implementing the `triangular`@mtype module type.
-- Usage: `module m = mk_triangular f64`.
module mk_triangular (T: field) : triangular with t = T.t = {
  type t = T.t
  module upper = {
    open (mk_upper_triangular_mat T)
    def transpose [n] (m: mat[n]) = m
  }
  module lower = {
    open (mk_lower_triangular_mat T)
    def transpose [n] (m: mat[n]) = m
  }
  type~ upper[n] = upper.mat[n]
  type~ lower[n] = lower.mat[n]
}