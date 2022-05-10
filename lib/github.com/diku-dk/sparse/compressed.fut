-- | Compressed sparse matrices.
--
-- A compressed sparse matrix is a matrix that stores fewer elements
-- than a corresponding dense regular matrix (non-stored elements are
-- assumed to be zero). There are two different kinds of compressed
-- sparse matrices, compressed sparse row matrices, which are indexed
-- by row, and compressed sparse column matrices that are indexed by
-- column. Transposing a compressed sparse row matrix yields (with
-- zero cost) a compressed sparse column matrix (and vice versa).

import "../segmented/segmented"
import "../linalg/linalg"
import "../sorts/merge_sort"

import "matrix"

-- | Module type including modules for compressed sparse row matrix
-- operations (`csr`) and compressed sparse column matrix operations
-- (`csc`). The abstract matrix types `csr` and `csc` are size-lifted
-- to indicate their potential irregular structure. The module type is
-- declared `local` to avoid that outside code makes direct use of the
-- module type (allowing it to be extended in minor revisions).

local module type compressed = {
  type t
  type~ csr [n][m]
  type~ csc [n][m]

  -- | Compressed sparse row
  module csr : {
    include matrix with t = t
                   with mat [n][m] = csr[n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> csc[m][n]
    -- | Sparse matrix vector multiplication. Given a sparse `n` times
    -- `m` matrix and a vector of size `m`, the function returns a
    -- vector of size `n`, the result of multiplying the argument matrix
    -- with the argument vector.
    val smvm      [n][m] : mat[n][m] -> [m]t -> [n]t
  }

  -- | Compressed sparse column
  module csc : {
    include matrix with t = t
                   with mat [n][m] = csc[n][m]
    -- | Matrix transposition.
    val transpose [n][m] : mat[n][m] -> csr[m][n]
  }

  -- | Sparse matrix-matrix multiplication.
  val smm [n][m][k] : csr[n][m] -> csc[m][k] -> csr[n][k]
}

-- | Parameterised compressed sparse matrix module with individual
-- submodules for compressed sparse row (CSR) and compressed sparse
-- column (CSC) representations. The module is parameterised over a
-- field (defined in the linalg package).

module mk_compressed (T : field) : compressed with t = T.t = {

  type t = T.t

  -- sorting and merging of coo values
  type coo [nnz] = [nnz](i64,i64,t)

  def zero_val = T.i64 0
  def one_val = T.i64 1
  def eq a b = !(a T.< b) && !(b T.< a)

  def sort_coo [nnz] (coo: coo[nnz]) : coo[nnz] =
    merge_sort (\(r1,c1,_) (r2,c2,_) -> r1 < r2 || (r1 == r2 && c1 <= c2))
               coo

  def merge_coo [nnz] (coo: coo[nnz]) : coo[] =
    let flags = map2 (\(r1,c1,_) (r2,c2,_) -> r1!=r2 || c1!=c2)
                     coo
                     (rotate (-1) coo)
    let flags = if nnz >= 1 then flags with [0] = true
		else flags

    in segmented_reduce (\(r1,c1,v1) (_,_,v2) -> (r1,c1,v1 T.+ v2))
                        (0,0,zero_val) flags coo

  def norm_coo [nnz] (coo: coo[nnz]) : coo[] =
    sort_coo coo |> merge_coo

  module csr = {

    type t = t

    type~ mat [n][m] = ?[nnz]. {dummy_m  : [m](),
 				row_off  : [n]i64,    -- prefix 0
				col_idx  : [nnz]i64,  -- size nnz
				vals     : [nnz]t}    -- size nnz

    def zero (n:i64) (m:i64) : mat[n][m] =
      {dummy_m = replicate m (),
       row_off=replicate n 0,
       col_idx=[],
       vals=[]
       }

    def eye (n:i64) (m:i64) : mat[n][m] =
      let e = i64.min n m
      let row_off =
	(map (+1) (iota e) ++ replicate (i64.max 0 (n-e)) e) :> [n]i64
      in {dummy_m = replicate m (),
	  row_off=row_off,
	  col_idx=iota e,
	  vals=replicate e one_val
	  }

    def dense [n][m] (csr: mat[n][m]) : [n][m]t =
      let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
		 dummy_m=_} = csr
      let roff0 (i:i64) = if i == 0 then 0 else row_off[i-1]
      let rs = map2 (\i r -> (i,r - roff0 i))
		    (iota n)
		    row_off
      let rss = (expand (\(_,n) -> n) (\(r,_) _ -> r) rs) :> [nnz]i64
      let iss = map2 (\r c -> (r,c)) rss col_idx
      let arr : *[n][m]t = tabulate_2d n m (\ _ _ -> T.i64 0)
      in scatter_2d arr iss vals

    def smvm [n][m] (csr:mat[n][m]) (v:[m]t) : [n]t =
      -- expand each row into an irregular number of multiplications, then
      -- do a segmented reduction with + and 0.
      let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
		 dummy_m=_} = csr
      let roff0 (i:i64) = if i == 0 then 0 else row_off[i-1]
      let rows = map2 (\i r -> (i,
				roff0 i,
				r-roff0 i))
		     (iota n) row_off
      let sz r = r.2
      let get r i = (T.*) (vals[r.1+i]) (v[col_idx[r.1+i]])
      in (expand_outer_reduce sz get (T.+) (T.i64 0) rows) :> [n]t

    def scale [n][m] (v:t) (csr:mat[n][m]) : mat[n][m] =
      let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
		 dummy_m} = csr
      in {row_off, col_idx, dummy_m,
	  vals = map ((T.*) v) vals}

    def sparse [nnz0] (n:i64) (m:i64) (coo:coo[nnz0]) : mat[n][m] =
      let [nnz] coo : coo[nnz] = norm_coo coo
      let _ = map (\(r,c,_) -> assert (0 <= r && r < n && 0 <= c && c < m) 0) coo
      let (rs,col_idx,vals) = unzip3 coo
      let rows = reduce_by_index (replicate n 0) (+) 0 rs (replicate nnz 1)
      let row_off = scan (+) 0 rows
      in {row_off, col_idx, vals, dummy_m=replicate m ()}

    def nnz [n][m] (csr:mat[n][m]) : i64 =
      map (\v -> if eq v zero_val then 0 else 1) csr.vals
      |> reduce (+) 0

    def coo [n][m] (csr:mat[n][m]) =
      let [nnz] {row_off: [n]i64, col_idx: [nnz]i64, vals: [nnz]t,
		 dummy_m= _} = csr
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

  -- SMM (sparse matrix multiply) algorithm for C[n][k] := A[n][m] * B[m][k]
  -- Assumption: A is in CSR format; B is in CSC format
  --  1. expand each row in A into contributions to elements in C
  --  2. expand each column in B into contributions to elements in C
  --  3. shuffle and merge contribution from A and B (eliminate or multiply)
  --  4. use the resulting COO representation to create a CSR or CSC representation

  type contrib = {r:i64, c:i64, v:t}
  def swap_rc ({r,c,v} : contrib) : contrib =
    {r=c,c=r,v}

  def szs [n][m] (csr:csr[n][m]) : [n]i64 =
    map2 (\i r -> if i == 0 then r else r - csr.row_off[i-1])
	 (iota n) csr.row_off

  def get_contrib [n][m] (csr:csr[n][m]) (r:i64) (i:i64) : contrib =
    let roff = if r == 0 then 0 else csr.row_off[r-1]
    in { r = r,
	 v = csr.vals[roff+i],
	 c = csr.col_idx[roff+i] }

  type option 't = #None | #Some t
  type contr = {tr:i64, tc:i64, s:i64, v:t}
  def cmp_contr (c:contr) (c':contr) : bool =
    c.tr == c'.tr && c.tc == c'.tc && c.s == c'.s

  def lte_contr (c:contr) (c':contr) : bool =
    c.tr < c'.tr ||
    (c.tr == c'.tr &&
     (c.tc < c'.tc ||
      (c.tc == c'.tc &&
       (c.s < c'.s ||
	(c.s == c'.s &&
	 (c.v T.< c'.v || eq c.v c'.v))))))

  def smm [n][m][k] (A:csr[n][m]) (B:csc[m][k]) : csr[n][k] =
    let szsA = szs A
    let szsB = szs B
    let szA r = szsA[r]
    let getA r i = get_contrib A r i
    let szB c = szsB[c]
    let getB c i = get_contrib B c i |> swap_rc

    let contribsRows =
      expand szA getA (iota n)
      |> map (\{r,c,v} ->
		map (\tc -> {tc,tr=r,s=c,v})  -- tc: target column, tr: target row
	            (iota k))
      |> flatten

    let contribsCols =
      expand szB getB (iota k)
      |> map (\{r,c,v} ->
		map (\tr -> {tc=c,tr,s=r,v})
  		    (iota n))
      |> flatten

    let [cn] contribs : [cn]contr =
      contribsRows ++ contribsCols |>
      merge_sort lte_contr

    -- eliminate components that have no friends
    let dummy : contr = {tr=0,tc=0,s=0,v=zero_val}

    let [cn2] contribs2 : [cn2]contr =
      map2 (\i c ->
	      if i == 0 && i == cn-1 then (#None : option contr)
	      else if i == 0 then
		   let c' = contribs[i+1]
		   in if cmp_contr c c' then #Some c
		      else #None
	      else if i == cn-1 then
		   let c' = contribs[i-1]
		   in if cmp_contr c c' then #Some c
		      else #None
	      else let c' = contribs[i-1]
		   let c'' = contribs[i+1]
		   in if cmp_contr c c' || cmp_contr c c'' then #Some c
		      else #None
	   ) (iota cn) contribs
      |> filter (\c -> match c
		       case #None -> false
		       case _ -> true)
      |> map (\c -> match c
		    case #Some x -> x
		    case #None -> dummy)
    let () = assert (cn2 % 2 == 0) ()

    let contribs3 =
      unflatten (cn2 / 2) 2 contribs2 |>
      map (\cs ->
	     let c0 = cs[0]
	     let c1 = cs[1]
	     in assert (cmp_contr c0 c1)
		       (c0 with v = c0.v T.* c1.v)
	  )

    let coos = map (\c -> (c.tr,c.tc,c.v)) contribs3
    in csr.sparse n k coos
}
