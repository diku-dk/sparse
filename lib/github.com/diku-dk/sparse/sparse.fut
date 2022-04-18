
import "../segmented/segmented"
import "../linalg/linalg"

-- | Module type for sparse matrix operations. The abstract matrix
-- type `mat` is size-lifted to indicate its potential irregular
-- structure. The module type is declared `local` to avoid that
-- outside code makes direct use of the module type.

local module type sparse = {
  type t
  type~ mat [n][m]

  val zero         : (n:i64) -> (m:i64) -> mat[n][m]
  val eye          : (n:i64) -> (m:i64) -> mat[n][m]
  val smvm  [n][m] : mat[n][m] -> [m]t -> [n]t
  val dense [n][m] : mat[n][m] -> [n][m]t
  val scale [n][m] : t -> mat[n][m] -> mat[n][m]
  val sparse [nnz] : (n:i64) -> (m:i64) -> [nnz](i64,i64,t) -> mat[n][m]
  val nnz   [n][m] : mat[n][m] -> i64
  val coo   [n][m] : mat[n][m] -> ?[nnz].[nnz](i64,i64,t)
}

-- | Sparse matrix module based on a compressed sparse row (CSR)
-- representation, parameterised over a field (defined in the linalg
-- package).

module csr (T : field) : sparse with t = T.t = {

  type t = T.t

  type~ mat [n][m] = ?[nnz]. {dummy_n  : [n](),     -- nnz: number of non-zeros
  			      dummy_m  : [m](),
 			      row_off  : [n]i64,    -- prefix 0
			      col_idx  : [nnz]i64,  -- size nnz
			      vals     : [nnz]t}    -- size nnz

  -- | The zero matrix. Given `n` and `m`, the function returns an `n` times
  -- `m` empty sparse matrix (zeros everywhere).
  def zero (n:i64) (m:i64) : mat[n][m] =
    {dummy_n = replicate n (),
     dummy_m = replicate m (),
     row_off=replicate n 0,
     col_idx=[],
     vals=[]
     }

  -- | The eye. Given `n` and `m`, the function returns an `n` times
  -- `m` sparse matrix with ones in the diagonal and zeros elsewhere.
  def eye (n:i64) (m:i64) : mat[n][m] =
    let e = i64.min n m
    let one = T.i64 1
    let row_off =
      (map (+1) (iota e) ++ replicate (i64.max 0 (n-e)) e) :> [n]i64
    in {dummy_n = replicate n (),
	dummy_m = replicate m (),
	row_off=row_off,
	col_idx=iota e,
	vals=replicate e one
	}

  -- | Convert to dense format. Given a sparse matrix, the function
  -- returns a dense representation of the matrix.
  def dense [n][m] (csr: mat[n][m]) : [n][m]t =
    let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
	       dummy_n=_, dummy_m=_} = csr
    let roff0 (i:i64) = if i == 0 then 0 else row_off[i-1]
    let rs = map2 (\i r -> (i,r - roff0 i))
                  (iota n)
                  row_off
    let rss = (expand (\(_,n) -> n) (\(r,_) _ -> r) rs) :> [nnz]i64
    let iss = map2 (\r c -> (r,c)) rss col_idx
    let arr : *[n][m]t = tabulate_2d n m (\ _ _ -> T.i64 0)
    in scatter_2d arr iss vals

  -- | Sparse matrix vector multiplication. Given a sparse `n` times
  -- `m` matrix and a vector of size `m`, the function returns a
  -- vector of size `n`, the result of multiplying the argument matrix
  -- with the argument vector.
  def smvm [n][m] (csr:mat[n][m]) (v:[m]t) : [n]t =
    -- expand each row into an irregular number of multiplications, then
    -- do a segmented reduction with + and 0.
    let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
	       dummy_n=_, dummy_m=_} = csr
    let roff0 (i:i64) = if i == 0 then 0 else row_off[i-1]
    let rows = map2 (\i r -> (i,
  			      roff0 i,
			      r-roff0 i))
                   (iota n) row_off
    let sz r = r.2
    let get r i = (T.*) (vals[r.1+i]) (v[col_idx[r.1+i]])
    in (expand_outer_reduce sz get (T.+) (T.i64 0) rows) :> [n]t

  -- below we assume `row_off` is sorted; we could check that this
  -- invariant holds for the input data...
  def csr [nnz] (n:i64) (m:i64) {row_off:[n]i64, col_idx:[nnz]i64,
				 vals:[nnz]t} : mat[n][m] =
    {dummy_n=replicate n (),
     dummy_m=replicate m (),
     row_off=row_off,
     col_idx=col_idx,
     vals=vals}

  -- | Scale elements. Given a sparse matrix and a scale value `v`,
  -- the function returns a new sparse matrix with the elements scaled
  -- by `v`.
  def scale [n][m] (v:t) (csr:mat[n][m]) : mat[n][m] =
    let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
	       dummy_n, dummy_m} = csr
    in {row_off, col_idx, dummy_n, dummy_m,
	vals = map ((T.*) v) vals}

  -- | Create sparse matrix from coordinate arrays.
  def sparse [nnz] (n:i64) (m:i64) (vs:[nnz](i64,i64,t)) : mat[n][m] =
    -- memo: sort entries and test indices using assert
    let vals = map (.2) vs
    let col_idx = map (.1) vs
    let rs = map (.0) vs
    let rows = reduce_by_index (replicate n 0) (+) 0 rs (replicate nnz 1)
    let row_off = scan (+) 0 rows
    in {row_off, col_idx, vals, dummy_n=replicate n (),
	dummy_m=replicate m ()}

  -- | Number of non-zero elements. Given a sparse matrix, the
  -- function returns the number of non-zero elements.
  def nnz [n][m] (csr:mat[n][m]) : i64 =
    -- memo: should we first filter out zeros in vals?
    let [nnz] {row_off= _, col_idx= _, vals = _vals : [nnz]t,
	       dummy_n= _, dummy_m= _} = csr
    in nnz

  -- | Convert to coordinate vectors. Given a sparse matrix, convert
  -- it to coordinate vectors.
  def coo [n][m] (csr:mat[n][m]) =
    let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
	       dummy_n= _, dummy_m= _} = csr
    let ns = map3 (\i a b ->
		     let a = if i == n-1 then nnz else a
		     let b = if i == 0 then 0i64 else b
		     in a-b)
                  (iota n)
                  row_off
                  (rotate (-1) row_off)
    let row_idx = replicated_iota ns :> [nnz]i64
    in zip3 row_idx col_idx vals
}
