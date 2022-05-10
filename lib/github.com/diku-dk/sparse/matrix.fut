-- | Module type for sparse matrix operations. The abstract matrix
-- type `mat` is size-lifted to indicate its potential irregular-sized
-- structure. The module type is declared `local` to avoid that
-- outside code makes direct use of the module type.

module type matrix = {
  -- | The scalar type.
  type t
  -- | The type of sparse matrices of dimension `n` times `m`.
  type~ mat [n][m]
  -- | The zero matrix. Given `n` and `m`, the function returns an `n`
  -- times `m` empty sparse matrix (zeros everywhere).
  val zero         : (n:i64) -> (m:i64) -> mat[n][m]
  -- | The eye. Given `n` and `m`, the function returns an `n` times
  -- `m` sparse matrix with ones in the diagonal and zeros elsewhere.
  val eye          : (n:i64) -> (m:i64) -> mat[n][m]
  -- | Convert to dense format. Given a sparse matrix, the function
  -- returns a dense representation of the matrix.
  val dense [n][m] : mat[n][m] -> [n][m]t
  -- | Scale elements. Given a sparse matrix and a scale value `v`,
  -- the function returns a new sparse matrix with the elements scaled
  -- by `v`.
  val scale [n][m] : t -> mat[n][m] -> mat[n][m]
  -- | Create a sparse matrix from a coordinate array.
  val sparse [nnz] : (n:i64) -> (m:i64) -> [nnz](i64,i64,t) -> mat[n][m]
  -- | Number of non-zero elements. Given a sparse matrix, the
  -- function returns the number of non-zero elements.
  val nnz   [n][m] : mat[n][m] -> i64
  -- | Convert to coordinate vectors. Given a sparse matrix, convert
  -- it to coordinate vectors.
  val coo   [n][m] : mat[n][m] -> ?[nnz].[nnz](i64,i64,t)
  -- | Element-wise addition.
  val +     [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
  -- | Element-wise subtraction.
  val -     [n][m] : mat[n][m] -> mat[n][m] -> mat[n][m]
}
