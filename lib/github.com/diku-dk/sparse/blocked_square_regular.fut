-- | Blocked square regular matrices

import "../sorts/radix_sort"
import "../segmented/segmented"
import "../linalg/linalg"
import "../linalg/lu"
import "../containers/setops"

local
-- | The `blocked_square_regular` module type. Blocked square regular matrices
-- are represented as a sequence of identically-sized square dense blocks. The
-- structure is sparse meaning that blocks that are known to consist only of
-- zeros need not be represented.  The module type `blocked_square_regular` is
-- declared `local`, which means that it may not be referenced directly by name
-- from client code.  This limitation makes it possible for the interface to be
-- enriched by new members in future minor versions.
module type blocked_square_regular = {
  type t
  type~ mat [n]

  -- | `bsz` is the blocksize of blocks in each of the two dimensions.
  val bsz : i64

  -- | `dim a` returns `n` when `a : mat[n]`.
  val dim [n] : mat [n] -> i64

  -- | `zero n` returns the zero-matrix of dimension `n` x `n`. Here `n` must be
  -- a multiple of `bsz`.
  val zero : (n: i64) -> mat [n]

  -- | `mk n bs` returns a blocked matrix of dimension `n` x `n` with blocks
  -- specified by `bs`. Here `n` must be a multiple of `bsz`.
  val mk [nz] : (n: i64) -> [nz](i64, i64, [bsz][bsz]t) -> mat [n]

  -- | `eye n` returns the identity matrix of dimension `n` x `n`. Here `n` must
  -- be a multiple of `bsz`.
  val eye : (n: i64) -> mat [n]

  -- | `transp a` returns `a` transposed.
  val transp [n] : mat [n] -> mat [n]

  -- | `dense a` returns a dense version of the blocked matrix `a`.
  val dense [n] : mat [n] -> [n][n]t

  -- | `add a b` returns the result of adding `a` and `b`, element-wise.
  val add [n] : mat [n] -> mat [n] -> mat [n]

  -- | `sub a b` returns the result of subtracting `b` from `a`, element-wise.
  val sub [n] : mat [n] -> mat [n] -> mat [n]

  -- | `mul a b` returns the result of multiplying `a` and `b`, element-wise.
  val mul [n] : mat [n] -> mat [n] -> mat [n]

  -- | `scale s a` returns matrix `a` with all elements scaled by `s`.
  val scale [n] : t -> mat [n] -> mat [n]

  -- | `diag v` returns a diagonal matrix with dimension `n` x `n` with diagonal
  -- elements from `v`.
  val diag [n] : [n]t -> mat [n]

  -- | `smvm a v` returns the vector resulting from multiplying the sparse
  -- matrix `a` with the dense vector `v`.
  val smvm [n] : mat [n] -> [n]t -> [n]t

  -- | `smsmm a b` returns the sparse blocked matrix resulting from multiplying
  -- the sparse matrix `a` with the sparse matrix `b`.
  val smsmm [n] : mat [n] -> mat [n] -> mat [n]

  -- | `blu_nofill a` returns a sparse blocked matrix representing an
  -- LU-decomposition of `a`, assuming no fill-ins will occur.
  val blu_nofill [n] : mat [n] -> mat [n]
}

-- | Module for creating blocked square regular matrices. The module is
-- parameterised by a field (defined in the `linalg` package) and a block size
-- (`bsc`).
module blocked_square_regular (T: field) (X: {val bsz : i64})
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

  def ERROR_block_size_must_divide_n x = x  -- assertion message

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
    let () = assert (ERROR_block_size_must_divide_n (n % bsz == 0)) ()
    let nb = n / bsz
    in { n = replicate n ()
       , idxs = map (\(r, c, _) -> r * nb + c) blks
       , blks = map (.2) blks
       }

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
    let () = assert (ERROR_block_size_must_divide_n (n % bsz == 0)) ()
    in mk n (tabulate (n / bsz) (\i ->
                                   let db = d[i * bsz:(i + 1) * bsz] :> [bsz]t
                                   in (i, i, blk_diag db)))

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

  -- Some tools
  def dotprod [n] (a: [n]T.t) (b: [n]T.t) : T.t =
    map2 (T.*) a b |> reduce (T.+) (T.i64 0)

  -- Solve (xU = y, x), where x and y are vectors and U is an upper-triangular square matrix
  def backsolve' [n] (y: [n]t) (U: [n][n]t) : [n]t =
    let x: *[n]t = replicate n (T.i64 0)
    in loop x for i in 0..<n do
         let sum = dotprod x[:i] U[:i, i]
         let x[i] = copy (y[i] T.- sum) T./ U[i, i]
         in x

  -- Solve (Lx = b, x), where x and b are vectors and L is a lower-triangular matrix
  def forsolve [n] (L: [n][n]t) (b: [n]t) : [n]t =  -- reads only lower triangular entries in L
    let y: *[n]t = replicate n (T.i64 0)
    in loop y for i in 0..<n do
         let sum = dotprod L[i, :i] y[:i]
         let y[i] = copy (b[i] T.- sum)
         in y

  module lu = mk_lu (T)

  def matsub a b = linalg.matop (T.-) a b

  def blu_nofill [n] (a: mat [n]) : mat [n] =
    let [nz] a: {idxs: [nz]i64, blks: [nz][bsz][bsz]t, n: [n]()} = a
    let idxs = a.idxs
    let blks = a.blks
    let nb = n / bsz
    -- number of blocks in each dimension
    let hrcbs =
      map3 (\h i b ->
              let (r, c) = idx_unflatten nb i
              in (h, r, c, b))
           (iota nz)
           idxs
           blks
    let hrcbs =
      loop hrcbs for i < nb
      do let blks = filter (\(_, r, c, _) -> r >= i && c >= i) (copy hrcbs)
	 let (h, r, c, b) = blks[0] -- h is the idx into hrcbs identifying the current diagonal block
         let b = lu.lu 1 b
  	 let () = assert (r == i && c == i) ()
         let A21 = filter (\(_, r, c, _) -> r > i && c == i) blks
         let A12 = filter (\(_, r, c, _) -> c > i && r == i) blks
         let X21 = map (\(h, r, _, a) ->
			  (h, r, map (\aa -> backsolve' aa b) a)
		       ) A21
         let X12 = map (\(h, _, c, a) ->
			  (h, c, transpose (map (\aa -> forsolve b aa) (transpose a)))
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
    let blks = map (.3) hrcbs
    in { n = a.n
       , idxs = idxs
       , blks = blks
       }

  def lu_dense [n] (a: [n][n]t) : [n][n]t =
    lu.lu 2 a
}
