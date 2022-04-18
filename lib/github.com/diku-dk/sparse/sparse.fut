
import "../segmented/segmented"
import "../linalg/linalg"
import "../sorts/merge_sort"

-- | Module type for sparse matrix operations. The abstract matrix
-- type `mat` is size-lifted to indicate its potential irregular
-- structure. The module type is declared `local` to avoid that
-- outside code makes direct use of the module type.

local module type mat = {
  type t
  type~ mat [n][m]

  val zero         : (n:i64) -> (m:i64) -> mat[n][m]
  val eye          : (n:i64) -> (m:i64) -> mat[n][m]
  val dense [n][m] : mat[n][m] -> [n][m]t
  val scale [n][m] : t -> mat[n][m] -> mat[n][m]
  val sparse [nnz] : (n:i64) -> (m:i64) -> [nnz](i64,i64,t) -> mat[n][m]
  val nnz   [n][m] : mat[n][m] -> i64
  val coo   [n][m] : mat[n][m] -> ?[nnz].[nnz](i64,i64,t)
  val +     [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
  val -     [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
}

-- | Module type including modules for sparse compressed row matrix
-- operations (`csr`) and sparse compressed column matrix operations
-- (`csc`). The abstract matrix types `csr` and `csc` are size-lifted
-- to indicate their potential irregular structure. The module type is
-- declared `local` to avoid that outside code makes direct use of the
-- module type (allowing it to be extended in minor revisions).

local module type sparse = {
  type t
  type~ csr [n][m]
  type~ csc [n][m]

  module csr : {
    include mat with t = t
                with mat [n][m] = csr [n][m]
    val transpose [n][m] : mat[n][m] -> csc[m][n]
    val smvm      [n][m] : mat[n][m] -> [m]t -> [n]t
  }

  module csc : {
    include mat with t = t
                with mat [n][m] = csc [n][m]
    val transpose [n][m] : mat[n][m] -> csr[m][n]
  }
}

-- | Sparse matrix module based on a compressed sparse row (CSR)
-- representation, parameterised over a field (defined in the linalg
-- package). The resulting module includes two submodules, a `csr`
-- module and a `csc` module. Sparse matrix-vector multiplication is
-- available in the `csr` module.

module sparse (T : field) -- : sparse with t = T.t
= {

  type t = T.t

  -- sorting and merging of coo values
  local type~ coo [nnz] = [nnz](i64,i64,t)

  local def sort_coo [nnz] (coo: coo[nnz]) : coo[nnz] =
    merge_sort (\(r1,c1,_) (r2,c2,_) -> r1 < r2 || (r1 == r2 && c1 <= c2))
               coo

  local def merge_coo [nnz] (coo: coo[nnz]) : coo[] =
    let flags = map2 (\(r1,c1,_) (r2,c2,_) -> r1!=r2 || c1!=c2)
                     coo
                     (rotate (-1) coo)
    in segmented_reduce (\(r1,c1,v1) (_,_,v2) -> (r1,c1,v1 T.+ v2))
                        (0,0,T.i64 0) flags coo

  local def norm_coo [nnz] (coo: coo[nnz]) : coo[] =
    sort_coo coo |> merge_coo

  module csr = {

    type t = t

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
    local def csr [nnz] (n:i64) (m:i64) {row_off:[n]i64, col_idx:[nnz]i64,
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
    def sparse [nnz0] (n:i64) (m:i64) (coo:coo[nnz0]) : mat[n][m] =
      let [nnz] coo : coo[nnz] = norm_coo coo
      let _ = map (\(r,c,_) -> assert (0 <= r && r < n && 0 <= c && c < m) 0) coo
      let vals = map (.2) coo
      let col_idx = map (.1) coo
      let rs = map (.0) coo
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

    def (+) [n][m] (csr1:mat[n][m]) (csr2:mat[n][m]) : mat[n][m] =
      (coo csr1 ++ coo csr2) |> sparse n m

    def (-) [n][m] (csr1:mat[n][m]) (csr2:mat[n][m]) : mat[n][m] =
      (coo csr1 ++ coo (scale (T.i64(-1)) csr2)) |> sparse n m

    def transpose [n][m] (mat:mat[n][m]) : mat[n][m] =
      mat
  }

  module csc = {

    type t = t

    def zero (n:i64) (m:i64) : csr.mat[m][n] =
      csr.zero m n

    def scale [n][m] (v:t) (mat:csr.mat[n][m]) : csr.mat[n][m] =
      csr.scale v mat

    def eye (n:i64) (m:i64) : csr.mat[m][n] =
      csr.eye m n

    def nnz [n][m] (mat:csr.mat[n][m]) : i64 =
      csr.nnz mat

    def coo [n][m] (mat: csr.mat[n][m]) : ?[nnz].[nnz](i64,i64,t) =
      csr.coo mat |> map (\(r,c,v) -> (c,r,v))

    def sparse [nnz] (n:i64) (m:i64) (coo: [nnz](i64,i64,t)) : csr.mat[m][n] =
      map (\(r,c,v) -> (c,r,v)) coo |> csr.sparse m n

    def dense [n][m] (mat: csr.mat[n][m]) : [m][n]t =
      csr.dense mat |> transpose

    def (+) x y = x csr.+ y
    def (-) x y = x csr.- y

    def transpose [n][m] (mat:csr.mat[n][m]) : csr.mat[n][m] =
      mat

    type~ mat[n][m] = csr.mat[m][n]
  }

  type~ csr[n][m] = csr.mat[n][m]
  type~ csc[n][m] = csc.mat[n][m]

}
