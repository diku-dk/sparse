-- | Sparse trapezoidal matrices.
--
-- A trapezoidal matrix is an `n` times `m` matrix where all elements
-- above or below the diagonal are zero, called respectively *upper*
-- and *lower* trapezoidal matrixes.  While we can always represent an
-- `n` times `m` trapezoidal matrix as an ordinary `n` times `m`
-- matrix where we store the zeroes, a less wasteful representation is
-- possible.  Instead, we can use a representation where we store only
-- the possibly nonzero elements.  This library supports both upper
-- and lower trapezoidal matrices using the same interface, but
-- different concrete types. Notice that trapezoidal matrices, in
-- contrast to triangular matrices, are not required to be square.

import "../linalg/linalg"
import "../segmented/segmented"

-- | The module type of a trapezoidal matrix.  This module type leaves
-- it unstated whether it is an upper or lower trapezoidal matrix, but
-- specific instantiations make it clear.
module type trapezoidal_matrix = {
  -- | The scalar type.
  type t
  -- | The type of `n` times `m` trapezoidal matrices.
  type~ mat[n][m]
  -- | The zero matrix. Given `n` and `m`, the function returns an `n`
  -- times `m` empty sparse matrix (zeros everywhere).
  val zero          : (n:i64) -> (m:i64) -> mat[n][m]
  -- | The eye. Given `n` and `m`, the function returns an `n` times
  -- `m` sparse matrix with ones in the diagonal and zeros elsewhere.
  val eye           : (n:i64) -> (m:i64) -> mat[n][m]
  -- | The diagonal matrix with zeros everywhere but in the diagonal.
  val diag      [n] : [n]t -> mat[n][n]
  -- Constructs trapezoidal matrix from dense array.  Elements on the
  -- zero side of the the diagonal are ignored.
  val trapezoidal [n][m] : [n][m]t -> mat[n][m]
  -- | Convert to dense format. Given a sparse matrix, the function
  -- returns a dense representation of the matrix.
  val dense  [n][m] : mat[n][m] -> [n][m]t
  -- | `idx (i,j) a` produces the element at logical position
  -- `(i,j)` in the trapezoidal matrix `a`.
  val idx    [n][m] : (i64,i64) -> mat[n][m] -> t
  -- | Scale elements. Given a matrix and a scale value `v`, the
  -- function returns a new matrix with the elements scaled by `v`.
  val scale  [n][m] : t -> mat[n][m] -> mat[n][m]
  -- | Element-wise addition.
  val +      [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
  -- | Element-wise subtraction.
  val -      [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
  -- | Map a function across the elements of the matrix.
  val map    [n][m] : (t -> t) -> mat[n][m] -> mat[n][m]
  -- | Number of non-zero elements.
  val nnz    [n][m] : mat[n][m] -> i64
  -- | Matrix multiplication.
  val smm [n][m][k] : mat[n][m] -> mat[m][k] -> mat[n][k]
}

-- Number of nonzero elements for triangular `n` by `n` array.
local def elements (n:i64) =
  (n * (1+n))/2

-- Number of nonzero elements for lower-trapezoidal `n` by `m` arrays.
local def elements_lower (n:i64) (m:i64) =
  let k = i64.min n m
  in (k * (1+k)) / 2 + k * i64.max (n-m) 0

-- Row in a lower-triangular array, given the value index (solution to
-- a second-degree equation)
local def row (i:i64) =
  i64.f64 (f64.ceil ((f64.sqrt(f64.i64(9+8*i))-1)/2))-1

-- lower: row major, upper: column major
local def row_lower (n:i64) (m:i64) (i:i64) =
  let k = i64.min n m
  let e = elements k
  let () = assert (n >= m || i < e) ()
  in if i < e then row i
     else k + (i64.max (i - e) 0) / m

local def col_lower (n:i64) (m:i64) (i:i64) =
  let k = i64.min n m
  let e = elements k
  let () = assert (n >= m || i < e) ()
  in if i <= e
     then i - elements (row_lower n m i)
     else (i - e) i64.% m

local module type ranking = {
  val rank   : (i64,i64) -> (i64, i64) -> i64
  val unrank : (i64,i64) -> i64 -> (i64,i64)
  val zero   : (i64,i64) -> bool
  val datasz : (i64,i64) -> i64
}

local module mk_trapezoidal_matrix (T:field) (R:ranking) = {
  type t = T.t

  type~ mat [n][m] =
    ?[nnz].
     { size: [0][n][m](),
       data: [nnz]t
     }

  def idx [n][m] (i,j) (tra: mat[n][m]) =
    if R.zero (i,j) then T.i64 0 else #[unsafe] tra.data[R.rank (n,m) (i,j)]

  def trapezoidal [n][m] (arr: [n][m]t) : mat[n][m] =
    { size = [],
      data = tabulate (R.datasz (n,m))
                      (\p -> let (i,j) = R.unrank (n,m) p
                             in #[unsafe] arr[i,j])
    }

  def dense [n][m] (tra: mat[n][m]) =
    tabulate_2d n m (\i j -> idx (i,j) tra)

  def zero n m : mat[n][m] =
    { size = [],
      data = replicate (R.datasz (n,m)) (T.i64 0)
    }

  def eye n m =
    trapezoidal (tabulate_2d n m (\i j -> T.i64 (i64.bool (i==j))))

  def diag [n] (v:[n]t) =
    trapezoidal (tabulate_2d n n (\i j ->
				    if i==j then v[i] else T.i64 0))

  def scale [n][m] s (tra:mat[n][m]) : mat[n][m] =
    tra with data = map (T.*s) tra.data

  def smm [n][m][k] (a:mat[n][m]) (b:mat[m][k]) : mat[n][k] =
    let sz (i,j) =
      if j >= m then 1 else i64.min (m-1) i - j + 1 -- lower: i >= j
    let get (i,j) c : T.t =
      if j >= m then T.i64 0
      else a.data[R.rank (n,m) (i,j + c)] T.*
	   b.data[R.rank (m,k) (j + c,j)]
    in { size = [],
	 data =
  	   expand_outer_reduce
  	   sz get (T.+) (T.i64 0)
  	   (iota (elements_lower n k) |> (map (R.unrank (n,k))))
       }

  def (+) [n][m] (x:mat[n][m]) (y:mat[n][m]) =
    let [nnz] (xdata: [nnz]t) = x.data
    let ydata = y.data :> [nnz]t
    in x with data = map2 (T.+) xdata ydata

  def x - y = x + scale (T.i64 (-1)) y

  local def neq x y = x T.< y || y T.< x
  def nnz [n][m] (a:mat[n][m]) : i64 =
    map (neq (T.i64 0) >-> i64.bool) a.data |> reduce (i64.+) 0i64

  def map [n][m] f (tra:mat[n][m]) =
    tra with data = map f tra.data
}

local module rank_lower = {
  def rank (n,m) (i,j) =
    if m > n || i <= m then elements i + j
    else elements m + (i-m) * m + j
  def unrank (n,m) (p:i64) =
    (row_lower n m p,
     col_lower n m p)
  def zero (i: i64, j) =
    j > i
  def datasz (n:i64, m:i64) =
    elements_lower n m
}

local module rank_upper = {
  def rank (n,m) (i,j) =
    rank_lower.rank (m,n) (j,i)
  def unrank (n,m) p =
    rank_lower.unrank (m,n) p |> (\(x,y) -> (y,x))
  def zero (i:i64,j:i64) =
    rank_lower.zero (j,i)
  def datasz (n:i64, m:i64) =
    rank_lower.datasz (m,n)
}

local module mk_lower_trapezoidal_matrix (T: field) =
  mk_trapezoidal_matrix T rank_lower

local module mk_upper_trapezoidal_matrix (T: field) =
  mk_trapezoidal_matrix T rank_upper

-- | The type of modules implementing trapezoidal matrices, with
-- distinct submodules and types for lower and upper trapezoidal
-- matrices.
module type trapezoidal = {
  -- | Matrix element type.
  type t
  -- | A lower trapezoidal matrix.
  type~ lower[n][m]
  -- | An upper trapezoidal matrix.
  type~ upper[n][m]
  -- | Operations on lower trapezoidal matrices.
  module lower : {
    include trapezoidal_matrix with t = t with mat [n][m] = lower[n][m]
    -- | Transpose lower trapezoidal matrix, producing upper
    -- trapezoidal matrix. O(1).
    val transpose [n][m] : lower[n][m] -> upper[m][n]
  }
  -- | Operations on upper trapezoidal matrices.
  module upper : {
    include trapezoidal_matrix with t = t with mat [n][m] = upper[n][m]
    -- | Transpose upper trapezoidal matrix, producing lower trapezoidal
    -- matrix.  O(1).
    val transpose [n][m] : upper[n][m] -> lower[m][n]
  }
}

-- | Create a module implementing the `trapezoidal`@mtype module type.
-- Usage: `module m = mk_trapezoidal f64`.
module mk_trapezoidal (T: field) : trapezoidal with t = T.t = {
  type t = T.t
  module lower = {
    open (mk_lower_trapezoidal_matrix T)
    def transpose [n][m] (a: mat[n][m]) : mat[m][n] =
      a with size = []
  }
  module upper = {
    open (mk_upper_trapezoidal_matrix T)
    def transpose [n][m] (a: mat[n][m]) : mat[m][n] =
      a with size = []
    def smm a b = transpose (lower.smm (transpose b) (transpose a))
  }
  type~ upper[n][m] = upper.mat[n][m]
  type~ lower[n][m] = lower.mat[n][m]
}
