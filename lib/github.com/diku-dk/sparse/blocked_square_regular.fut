-- | Blocked square regular matrices. Blocked square regular matrices are
-- represented as a sequence of identically-sized square dense blocks. The
-- structure is sparse meaning that blocks that are known to consist only of
-- zeros need not be represented. Matrices are required to be a multiple of the
-- block size in each dimension.

import "../sorts/radix_sort"
import "../segmented/segmented"
import "../linalg/linalg"
import "../linalg/lup"
import "../linalg/perm"
import "../containers/setops"

local
-- | The `blocked_square_regular` module type. The module type
-- `blocked_square_regular` is declared `local`, which means that it may not be
-- referenced directly by name from client code.  This limitation makes it
-- possible for the interface to be enriched by new members in future minor
-- versions.
module type blocked_square_regular = {

  -- | Type of elements.
  type t

  -- | Type of square matrices of size `n` x `n`.
  type~ mat [n]

  -- | The entry `bsz` is the block size of blocks in each of the two
  -- dimensions.
  val bsz : i64

  -- | The size of each dimension. The expression `dim a` returns `n` when `a :
  -- mat[n]`.
  val dim [n] : mat [n] -> i64

  -- | The zero-matrix. The expression `zero n` returns the zero-matrix of
  -- dimension `n` x `n`. Here `n` must be a multiple of `bsz`.
  val zero : (n: i64) -> mat [n]

  -- | Matrix construction. The expression `mk n bs` returns a blocked matrix of
  -- dimension `n` x `n` with blocks specified by `bs`. Here `n` must be a
  -- multiple of `bsz`.
  val mk [nz] : (n: i64) -> [nz](i64, i64, [bsz][bsz]t) -> mat [n]

  -- | The identity matrix. The expression `eye n` returns the identity matrix
  -- of dimension `n` x `n`. Here `n` must be a multiple of `bsz`.
  val eye : (n: i64) -> mat [n]

  -- | Transposition. The expression `transp a` returns `a` transposed.
  val transp [n] : mat [n] -> mat [n]

  -- | Conversion to a dense matrix. The expression `dense a` returns a dense
  -- version of the blocked matrix `a`.
  val dense [n] : mat [n] -> [n][n]t

  -- | Element-wise addition. The expression `add a b` returns the result of
  -- adding `a` and `b`, element-wise.
  val add [n] : mat [n] -> mat [n] -> mat [n]

  -- | Element-wise subtraction. The expression `sub a b` returns the result of
  -- subtracting `b` from `a`, element-wise.
  val sub [n] : mat [n] -> mat [n] -> mat [n]

  -- | Element-wise multiplication. The expression `mul a b` returns the result
  -- of multiplying `a` and `b`, element-wise.
  val mul [n] : mat [n] -> mat [n] -> mat [n]

  -- | Scaling. The expression `scale s a` returns matrix `a` with all elements
  -- scaled by `s`.
  val scale [n] : t -> mat [n] -> mat [n]

  -- | Construction of a diagonal matrix. The expression `diag v` returns a
  -- diagonal matrix of dimension `n` x `n` with diagonal elements taken from
  -- `v`.
  val diag [n] : [n]t -> mat [n]

  -- | Matrix-vector multiplication. The expression `smvm a v` returns the
  -- vector resulting from multiplying the sparse matrix `a` with the dense
  -- vector `v`.
  val smvm [n] : mat [n] -> [n]t -> [n]t

  -- | Matrix-matrix multiplication. The expression `smsmm a b` returns the
  -- sparse blocked matrix resulting from multiplying the sparse matrix `a` with
  -- the sparse matrix `b`.
  val smsmm [n] : mat [n] -> mat [n] -> mat [n]

  -- | LU-decomposition with block-partial (row) pivoting but without
  -- fill-ins. The expression `lup_nofill a` returns a pair `(LU,p)` of a sparse
  -- blocked matrix `LU` representing an LU-decomposition of `permute p a`,
  -- assuming no fill-ins will occur. The returned matrix `LU` embeds a lower
  -- triangular matrix L such that `lower LU = L` and an upper triangular matrix
  -- `U` such that `upper LU = U`. The intention is that `permute p (dense a) =
  -- dense(smsmm (lower b) (upper b))`. See below for more information about
  -- `lower` and `upper`. The partial pivoting is limited to be performed within
  -- a block.
  val lup_nofill [n] : mat [n] -> (mat [n], perm.t [n])

  -- | Fill-in computation. The expression `lu_find_fills m` returns the block
  -- coordinates for fill-elements required for LU decomposition.
  val lu_find_fills [n] : mat [n] -> ?[k].[k](i64,i64)

  -- | LU-decomposition with block-partial (row) pivoting. The expression `lup
  -- a` returns a pair `(LU, p)` of a sparse blocked matrix `LU` representing an
  -- LU-decomposition of `permute p a` and a permutation `p` representing the
  -- row-permutations performed by the block-limited partial (row) pivoting. The
  -- returned matrix `LU` embeds a lower triangular matrix `L` and an upper
  -- triangular matrix `U` such that `lower LU = L` and `upper LU = U`. The
  -- intention is that `permute p (dense a) = dense(smsmm (lower b) (upper
  -- b))`. See below for more information about `lower` and `upper`. The partial
  -- pivoting is limited to be performed within a block.
  val lup [n] : mat [n] -> (mat [n], perm.t [n])

  -- | LU-decomposition without pivoting. The expression `lu a` returns a parse
  -- blocked matrix `LU` representing an LU-decomposition of `a`. Similar to
  -- `lup a` but without pivoting.
  val lu [n] : mat [n] -> mat [n]

  -- | Extract lower-triangular matrix. The expression `lower a` returns the
  -- strictly lower-triangular part of `a` (i.e., excluding the diagonal
  -- elements in a). The result includes unit elements in the diagonal.
  val lower [n] : mat [n] -> mat [n]

  -- | Extract upper-triangular matrix. The expression `upper a` returns the
  -- upper-triangular part of `a`, including the diagonal elements.
  val upper [n] : mat [n] -> mat [n]

  -- | Forward solving. The expression `forsolve L b` solves (`Lx = b`, `x`),
  -- where `x` and `b` are vectors and `L` is a lower-triangular matrix. Reads
  -- only lower part of `L`, excluding the diagonal, and assumes implicit unit
  -- diagonal elements.
  val forsolve [n] : mat [n] -> [n]t -> [n]t

  -- | Backward solving. The expression `backsolve U y` solves (`Ux = y`, `x`),
  -- where `x` and `y` are vectors and `U` is an upper-triangular square matrix.
  -- Reads only upper part of `U`, including the diagonal.
  val backsolve [n] : mat [n] -> [n]t -> [n]t

  -- | Solve a sparse linear system using sparse LU-decomposition with partial
  -- block-limited (row) pivoting.
  val ols [n] : mat [n] -> [n]t -> [n]t

  -- | Convert to coordinate vectors. Given a sparse matrix, convert it to
  -- coordinate vectors. Zero-elements within blocks are removed. Non-zero
  -- elements are returned in row-major order.
  val coo [n] : mat [n] -> ?[nnz].[nnz](i64,i64,t)

  -- | Convert from coordinate vectors. The argument `n` must be a multiple of
  -- `bsz`.
  val from_coo [nnz] : (n:i64) -> [nnz](i64,i64,t) -> mat [n]

}

-- | Parameterised module for creating blocked square regular matrices. The
-- module is parameterised by a field (defined in the `linalg` package) and a
-- block size (`bsz`).
module blocked_square_regular (T: ordered_field) (X: {val bsz : i64})
  : blocked_square_regular with t = T.t = {
  type t = T.t
  def bsz = X.bsz
  def zero_t = T.i64 0
  def one_t = T.i64 1

  def idx_unflatten (n: i64) (i: i64) : (i64, i64) =
    (i / n, i % n)

  def idx_flatten (n: i64) (r: i64, c: i64) : i64 =
    r * n + c

  module linalg = mk_linalg (T)
  def matmul = linalg.matmul

  module lup_mod = mk_lup (T)

  -- Assertion error messages
  def ERROR_block_size_must_divide_n x = x
  def ERROR_diagonal_block_must_be_nonempty x = x
  def ERROR_backsolve_diagonal_element_is_zero x = x

  type~ mat [n] =
    ?[nz].{ n: [n]()              -- nz is the number of non-zero blocks
          , idxs: [nz]i64         -- n is dimension (a multiple of bsz)
          , blks: [nz][bsz][bsz]t -- Non-zero block indexes (row-major)
          }

  -- As an invariance, we assume bsz divides n

  def dim [n] (_bm: mat [n]) : i64 = n

  def zero (n: i64) : mat [n] =
    { n = replicate n ()
    , idxs = []
    , blks = []
    }

  def mk [nz] (n: i64) (blks: [nz](i64, i64, [bsz][bsz]t)) : mat [n] =
    assert (ERROR_block_size_must_divide_n (n % bsz == 0))
	   (let nb = n / bsz
	    in { n = replicate n ()
	       , idxs = map (\(r, c, _) -> r * nb + c) blks
	       , blks = map (.2) blks
	       })

  def blk_zero = tabulate_2d bsz bsz (\_ _ -> zero_t)

  def blk_eye (n: i64) : [n][n]t =
    tabulate_2d n n (\r c -> if c == r then one_t else zero_t)

  def eye (n: i64) : mat [n] =
    let blk = blk_eye bsz
    in mk n (tabulate (n / bsz) (\i -> (i, i, blk)))

  def dense [n] (bm: mat [n]) : [n][n]t =
    let nb = n / bsz
    let nzvs =
      map2 (\idx blk ->
              let (r, c) = idx_unflatten nb idx
              in tabulate_2d bsz bsz (\i j -> (bsz * r + i, bsz * c + j, blk[i][j])))
           bm.idxs
           bm.blks
      |> flatten
      |> flatten
    in scatter_2d (replicate n (replicate n zero_t))
                  (map (\(i, j, _) -> (i, j)) nzvs)
                  (map (\(_, _, v) -> v) nzvs)

  def transp [n] (bm: mat [n]) : mat [n] =
    let sw (x, y) = (y, x)
    let nb = n / bsz
    in { idxs = map (\i -> idx_flatten nb (sw (idx_unflatten nb i))) bm.idxs
       , n = bm.n
       , blks = map transpose bm.blks
       }

  def setop [m] [n] (a: [m]i64) (b: [n]i64) : [m + n](i64, i64, i64) =
    let c: [m + n](i64, i64, i64) =
      map2 (\x i -> (x, i, -1)) a (iota m)
      ++ map2 (\x i -> (x, -1, i)) b (iota n)
    in c
       |> radix_sort_by_key (.0) i64.num_bits i64.get_bit
       |> scan (\(x, a1, b1) (y, a2, b2) ->
                  if x == y
                  then (x, i64.max a1 a2, i64.max b1 b2)
                  else (y, a2, b2))
               (0, -1, -1)

  def union [m] [n] (a: [m]i64) (b: [n]i64) : ?[k].[k](i64, i64, i64) =
    let vs = setop a b
    in filter (\((v, _, _), i) ->
                 if i < m + n - 1 && vs[i + 1].0 == v
                 then false
                 else true)
              (zip vs (iota (m + n)))
       |> map (.0)

  def intersect [m] [n] (a: [m]i64) (b: [n]i64) : ?[k].[k](i64, i64, i64) =
    setop a b |> filter (\(_, x, y) -> x >= 0 && y >= 0)

  def binop_union [n] (f: t -> t -> t) (m1: mat [n]) (m2: mat [n]) : mat [n] =
    let xs = union m1.idxs m2.idxs
    let blks =
      map (\(_, i, j) ->
             if i >= 0
             then if j >= 0
                  then map2 (map2 f) m1.blks[i] m2.blks[j]
                  else m1.blks[i]
             else map (map (f zero_t)) (m2.blks[j]))
          xs
    let idxs = map (.0) xs
    in { n = m1.n
       , blks = blks
       , idxs = idxs
       }

  def add [n] (m1: mat [n]) (m2: mat [n]) : mat [n] =
    binop_union (T.+) m1 m2

  def sub [n] (m1: mat [n]) (m2: mat [n]) : mat [n] =
    binop_union (T.-) m1 m2

  def mul [n] (m1: mat [n]) (m2: mat [n]) : mat [n] =
    let xs = intersect m1.idxs m2.idxs
    let blks =
      map (\(_, i, j) ->
             map2 (map2 (T.*)) m1.blks[i] m2.blks[j])
          xs
    let idxs = map (.0) xs
    in { n = m1.n
       , blks = blks
       , idxs = idxs
       }

  def scale [n] (e: t) (m: mat [n]) : mat [n] =
    { n = m.n
    , blks = map (map (map (e T.*))) m.blks
    , idxs = m.idxs
    }

  def blk_diag [n] (d: [n]t) : [n][n]t =
    tabulate_2d n n (\c r -> if c == r then d[c] else zero_t)

  def diag [n] (d: [n]t) : mat [n] =
    assert (ERROR_block_size_must_divide_n (n % bsz == 0))
	   (mk n (tabulate (n / bsz) (\i ->
					let db = d[i * bsz:(i + 1) * bsz] :> [bsz]t
					in (i, i, blk_diag db))))

  def blk_mvm [n] (a: [n][n]t) (v: [n]t) =
    map (\c -> map2 (T.*) v c |> reduce (T.+) zero_t) a

  def smvm [n] (m: mat [n]) (v: [n]t) : [n]t =
    let nb = n / bsz
    let rws =
      map2 (\idx blk ->
              let (r, c) = idx_unflatten nb idx
              let vs = v[c * bsz:(c + 1) * bsz] :> [bsz]t
              let w = blk_mvm blk vs
              in (r, w))
           m.idxs
           m.blks
    let (is, as) =
      map (\(r, w) ->
             map (\j -> (r * bsz + j, w[j]))
                 (iota bsz))
          rws
      |> flatten
      |> unzip
    in reduce_by_index (replicate n zero_t)
                       (T.+)
                       zero_t
                       is
                       as

  def blk_add [n] (a: [n][n]t) (b: [n][n]t) : [n][n]t =
    map2 (map2 (T.+)) a b

  def smsmm [n] (a: mat [n]) (b: mat [n]) : mat [n] =
    let nb = n / bsz
    let (aridxs, acidxs) = map (idx_unflatten nb) a.idxs |> unzip
    let (bridxs, bcidxs) = map (idx_unflatten nb) b.idxs |> unzip
    let is = intersect acidxs bridxs
    let blks =
      map (\(_, ai, bi) ->
             let blk = matmul a.blks[ai] b.blks[bi]
             let r = aridxs[ai]
             let c = bcidxs[bi]
             let j = idx_flatten nb (r, c)
             in (j, blk))
          is
    let blks = radix_sort_by_key (.0) i64.num_bits i64.get_bit blks
    let (js, bs) = unzip blks
    let fs =
      map (\i ->
             if i == 0 || js[i] != js[i - 1]
             then 1
             else 0)
          (indices js)
    let k = reduce (+) 0 fs
    let idxs = filter (\(f, _) -> f > 0) (zip fs js) |> map (.1) :> [k]i64
    let blks = segmented_reduce blk_add blk_zero (map bool.i64 fs) bs :> [k][bsz][bsz]t
    in { n = replicate n ()
       , idxs = idxs
       , blks = blks
       }

  -- Find fill-elements for general sparse LU decomposition
  def lu_find_fills [n] (m:mat[n]) : ?[k].[k](i64,i64) =
    let nb = n / bsz
    let rcs = map (idx_unflatten nb) m.idxs
    let (_, acc) =
      loop (rcs,acc) = (rcs,[]) for i < nb do
      let rs = filter (\(r,c) -> r == i && c > i) rcs
      let cs = filter (\(r,c) -> c == i && r > i) rcs
      let fills = map (\(_,c) -> map (\(r,_) -> [(r,c)]) cs) rs
		  |> flatten |> flatten
      let newfills = setops.diff_by_key (idx_flatten nb) fills rcs
      let rest = filter (\(r,c) -> r > i && c > i) rcs
      let rest = rest++newfills
      let acc = acc++newfills
      in (rest,acc)
    in acc

  -- Some tools
  def matsub a b = linalg.matop (T.-) a b
  def dotprod a b = linalg.dotprod a b

  -- Solve (xU = y, x), where x and y are vectors and U is an upper-triangular
  -- square matrix. Notice x is on the left of U. Reads only upper part of U,
  -- including diagonal elements.
  def backsolve' [n] (U: [n][n]t) (y: [n]t) : [n]t =
    let x: *[n]t = replicate n (T.i64 0)
    in loop x for i in 0..<n do
         let sum = dotprod x[:i] U[:i, i]
         let x[i] = copy (y[i] T.- sum) T./ U[i, i]
         in x

  -- lu_nofill. The algoritm divides `a` into a block matrix
  --
  -- |  b  A12 |
  -- | A21  D  |
  --
  -- where `b` is one block (`bsz` x `bsz` elements) and A12 is a block row
  -- (multiple blocks in a row) and A21 is a block column (multiple blocks in a
  -- column). Notice that A12, A21, and D may be sparse. After performing a
  -- dense LU-decomposition of `b`, and thereby obtaining lu(b) = b', the
  -- algorithm uses forwards- and backwards-solving for computing X12 and X21,
  -- respectively. It then calculates the Schur-complement, which is subtracted
  -- from D, reaching D'', before repeating the process of LU-decomposition of
  -- D'', reaching X. Finally, the results are collected to reach
  --
  -- | b'  X12 |
  -- | X21  X  |
  --

  def lup_nofill [n] (a: mat [n]) : (mat [n], perm.t [n]) =
    let nb = n / bsz  -- number of blocks in each dimension
    let hrcbs =
      map3 (\h i b ->
              let (r, c) = idx_unflatten nb i
              in (h, r, c, b))
           (indices a.idxs)
           a.idxs
           a.blks
    let (hrcbs, p) =
      loop (hrcbs,p0) = (hrcbs,perm.id 0) for i < nb
      do
         let p0 = p0 :> perm.t[i*bsz]
         let blks = filter (\(_, r, c, _) -> r >= i && c >= i) hrcbs
	 let (h, r, c, b) = blks[0] -- h is the idx into hrcbs identifying the current diagonal block
  	 let (b,p:perm.t[bsz]) = assert (ERROR_diagonal_block_must_be_nonempty(r == i && c == i))
					(lup_mod.lup (copy b))
         let A21 = filter (\(_, r, c, _) -> r > i && c == i) blks
         let A12 = filter (\(_, r, c, _) -> c > i && r == i) blks
         let X21 = map (\(h, r, _, a) ->
			  (h, r, map (backsolve' b) a)
		       ) A21
         let X12 = map (\(h, _, c, a) ->
			  let a = perm.permute p a
			  in (h, c, transpose (map (lup_mod.forsolve b) (transpose a)))
		       ) A12
         let hrcbs[h] = (h, r, c, b)
         let hrcbs = scatter hrcbs (map (.0) X21) (map (\(h, r, b) -> (h, r, c, b)) X21)
         let hrcbs = scatter hrcbs (map (.0) X12) (map (\(h, c, b) -> (h, r, c, b)) X12)
         let D' =
           map (\(_, r, A) ->
                  map (\(_, c, B) ->
                         (idx_flatten nb (r, c), r, c, matmul A B))
                      X12)
               X21
           |> flatten
         let D =
           filter (\(_, r, c, _) -> r > i && c > i) blks
           |> map (\(h, r, c, b) -> (h, idx_flatten nb (r, c), r, c, b))
         let D'' =
           setops.join_by_key (.1) (.0) D D' -- ignore fillins
	   |> map (\((h, _, r, c, b), (_, _, _, b')) -> (h, r, c, matsub b b'))
         let hrcbs = scatter hrcbs (map (.0) D'') D''
	 -- permute blocks to the left of the diagonal in block row i
	 let bs = filter (\(_,r,c,_) -> r == i && c < i) hrcbs
	 let hrcbs = scatter hrcbs (map (.0) bs) (map (\(h,r,c,b) -> (h,r,c,perm.permute p (copy b))) bs)
         in (hrcbs, perm.add p0 p)
    in ({ n = a.n
	, idxs = a.idxs
	, blks = map (.3) hrcbs
        }, p :> perm.t[n])

  def lup [n] (a:mat [n]) : (mat[n], perm.t[n]) =
    let fills = lu_find_fills a
    let x = mk n (map (\(r,c) -> (r,c,tabulate_2d bsz bsz (\_ _ -> T.i64 0))) fills)
    in lup_nofill (add a x)

  def dense_strict_lower [n] (a:[n][n]t) =
    tabulate_2d n n (\i j -> if i > j then a[i][j] else T.i64 0)

  def dense_upper [n] (a:[n][n]t) =
    tabulate_2d n n (\i j -> if i <= j then a[i][j] else T.i64 0)

  def lower [n] (a:mat[n]) : mat[n] =
    let nb = n / bsz
    let (idxs,blks) =
      map2 (\i b -> (i,b)) a.idxs a.blks
      |> filter (\(i,_) -> let (r,c) = idx_unflatten nb i
			   in r >= c)
      |> map (\(i,b) -> let (r,c) = idx_unflatten nb i
			in if r == c then (i,dense_strict_lower b)
			   else (i,b))
      |> unzip
    let b = {n=a.n, idxs, blks}
    in add (eye n) b

  def upper [n] (a:mat[n]) : mat[n] =
    let nb = n / bsz
    let (idxs,blks) =
      map2 (\i b -> (i,b)) a.idxs a.blks
      |> filter (\(i,_) -> let (r,c) = idx_unflatten nb i
			   in r <= c)
      |> map (\(i,b) -> let (r,c) = idx_unflatten nb i
			in if r == c then (i,dense_upper b)
			   else (i,b))
      |> unzip
    in {n=a.n, idxs, blks}

  def lu_nofill [n] (a: mat [n]) : mat [n] =
    let nb = n / bsz  -- number of blocks in each dimension
    let hrcbs =
      map3 (\h i b ->
              let (r, c) = idx_unflatten nb i
              in (h, r, c, b))
           (indices a.idxs)
           a.idxs
           a.blks
    let hrcbs =
      loop hrcbs = hrcbs for i < nb
      do let blks = filter (\(_, r, c, _) -> r >= i && c >= i) hrcbs
	 let (h, r, c, b) = blks[0] -- h is the idx into hrcbs identifying the current diagonal block
  	 let b = assert (ERROR_diagonal_block_must_be_nonempty(r == i && c == i))
			(lup_mod.lu (copy b))
         let A21 = filter (\(_, r, c, _) -> r > i && c == i) blks
         let A12 = filter (\(_, r, c, _) -> c > i && r == i) blks
         let X21 = map (\(h, r, _, a) ->
			  (h, r, map (backsolve' b) a)
		       ) A21
         let X12 = map (\(h, _, c, a) ->
			  (h, c, transpose (map (lup_mod.forsolve b) (transpose a)))
		       ) A12
         let hrcbs[h] = (h, r, c, b)
         let hrcbs = scatter hrcbs (map (.0) X21) (map (\(h, r, b) -> (h, r, c, b)) X21)
         let hrcbs = scatter hrcbs (map (.0) X12) (map (\(h, c, b) -> (h, r, c, b)) X12)
         let D' =
           map (\(_, r, A) ->
                  map (\(_, c, B) ->
                         (idx_flatten nb (r, c), r, c, matmul A B))
                      X12)
               X21
           |> flatten
         let D =
           filter (\(_, r, c, _) -> r > i && c > i) blks
           |> map (\(h, r, c, b) -> (h, idx_flatten nb (r, c), r, c, b))
         let D'' =
           setops.join_by_key (.1) (.0) D D' -- ignore fillins
	   |> map (\((h, _, r, c, b), (_, _, _, b')) -> (h, r, c, matsub b b'))
         let hrcbs = scatter hrcbs (map (.0) D'') D''
         in hrcbs
    in { n = a.n
       , idxs = a.idxs
       , blks = map (.3) hrcbs
       }

  def lu [n] (a:mat [n]) : mat[n] =
    let fills = lu_find_fills a
    let x = mk n (map (\(r,c) -> (r,c,tabulate_2d bsz bsz (\_ _ -> T.i64 0))) fills)
    in lu_nofill (add a x)

  -- Solve (Lx = b, x), where x and b are vectors and L is a lower-triangular
  -- matrix. Reads only lower part of L, excluding the diagonal, and assumes
  -- implicit unit diagonal elements.
  def forsolve [n] (L:mat[n]) (b:[n]t) : [n]t =
    let y : *[n]t = replicate n (T.i64 0)
    let nb = n / bsz
    let blks = map2 (\h b ->
		       let (r,c) = idx_unflatten nb h
		       in (h, r, c, b)) L.idxs L.blks
    in loop y for k < nb
       do let bs = filter (\(_,r,c,_) -> r == k && c <= r) blks
	  in loop y for j < bsz do
	     let i = k * bsz + j
	     let sums = map (\(_,r,c,b) ->
			       let bound = if c == r then j else bsz
			       let bslice = b[j,:bound]
			       let yslice = y[c*bsz:c*bsz+bound] :> [bound]t
			       in dotprod bslice yslice) bs   --  dotprod L[i,:i] y[:i]
	     let sum = reduce (T.+) (T.i64 0) sums
	     let y[i] = copy(b[i] T.- sum)
	     in y

  -- Solve (Ux = y, x), where x and y are vectors and U is an upper-triangular
  -- square matrix.  Reads only upper part of U, including the diagonal.
  def backsolve [n] (U:mat[n]) (y:[n]t) : [n]t =
    let x : *[n]t = replicate n (T.i64 0)
    let nb = n / bsz
    let blks = map2 (\h b ->
		       let (r,c) = idx_unflatten nb h
		       in (h, r, c, b)) U.idxs U.blks
    in loop x for k in (iota nb)[::-1]
       do let bs = filter (\(_,r,c,_) -> r == k && c >= r) blks
	  let diagblk =
	    let diag = filter (\(_,r,c,_) -> r == c) bs
	    in assert (length diag == 1) (diag[0].3)
	  in loop x for j in (iota bsz)[::-1] do
	     let i = k * bsz + j
	     let sums = map (\(_,r,c,b) ->
			       let bound = if c == r then j+1 else 0
			       let bslice = b[j,bound:]
			       let xslice = x[c*bsz+bound:(c+1)*bsz] :> [bsz-bound]t
			       in dotprod bslice xslice) bs   -- dotprod U[i,i+1:] x[i+1:]
	     let sum = reduce (T.+) zero_t sums
	     let ejj = diagblk[j][j]
	     let x[i] = assert (ERROR_backsolve_diagonal_element_is_zero(ejj T.!= zero_t))
			       (copy(y[i] T.- sum) T./ ejj)
	     in x

  def ols [n] (a: mat [n]) (b:[n]t) : [n]t =
    let (LU,p) = lup a
    in backsolve LU (forsolve LU (perm.permute p (copy b)))

  def coo [n] (a: mat [n]) : ?[nnz].[nnz](i64,i64,t) =
    let nb = n / bsz
    in map2 (\h b ->
	       let (r,c) = idx_unflatten nb h
	       in map2 (\i row ->
			  map2 (\j v -> (r*bsz+i,c*bsz+j,v)) (iota bsz) row
		       ) (iota bsz) b
	    ) a.idxs a.blks
       |> flatten |> flatten
       |> filter (\(_,_,v) -> v T.!= zero_t)

  def from_coo [nnz] (n:i64) (xs:[nnz](i64,i64,t)) : mat [n] =
    let nb = assert (ERROR_block_size_must_divide_n(n % bsz == 0))
		    (n / bsz)
    let bs = map (\(i,j,b) ->
		    let h = assert (0 <= i && i < n && 0 <= j && j < n)
				   (idx_flatten nb (i/bsz,j/bsz))
		    in (h, i%bsz, j%bsz, b)
		 ) xs
    let bs = radix_sort_by_key (.0) i64.num_bits i64.get_bit bs
    let fs = map (\i -> if i == 0 || bs[i].0 != bs[i-1].0 then 1
			else 0) (indices bs)
    let bis = scan (+) 0 fs |> map (\bi -> bi-1)  -- block indices
    let idxs = map2 (\(h,_,_,_) f -> (h,f)) bs fs
	       |> filter (\(_,f) -> f > 0)
	       |> map (.0)
    let blks = map (\ _ -> tabulate_2d bsz bsz (\ _ _ -> zero_t)) idxs
    let is = map2 (\bi (_,s,t,_) -> (bi,s,t)) bis bs
    let vs = map (.3) bs
    in { n = replicate n ()
       , idxs = idxs
       , blks = scatter_3d blks is vs
       }

}
