//! Sparse symmetric-positive-definite (SPD) solve via a sparse Cholesky
//! factorisation.
//!
//! The matrix is supplied as COO triplets `(row, col, value)` over `n`
//! unknowns. The factorisation runs once; the resulting [`SparseCholesky`] then
//! solves any number of right-hand-side columns cheaply — the consuming
//! Laplacian-smoothing operator solves `(I + λ·L̃)·x = x_old` independently for
//! the x/y/z position components, i.e. three RHS columns against one factor.
//!
//! Everything is `f64` for numerical stability; callers holding `f32` mesh
//! positions cast on the way in and out.
//!
//! The factorisation is delegated to the `faer` crate (MIT-licensed, pure
//! Rust), keeping loopmesh's permissive licensing posture intact.

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Llt;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};

/// A factored SPD system, reusable across many right-hand sides.
///
/// Built by [`SparseCholesky::factor`] from COO triplets describing the **full
/// symmetric** matrix (both off-diagonal entries of each pair); only the lower
/// triangle is read for the factorisation. Duplicate `(row, col)` entries are
/// summed, which is convenient for assembling a Laplacian one stencil at a time.
pub struct SparseCholesky {
    factor: Llt<usize, f64>,
    dim: usize,
}

/// Failure modes of assembling or factoring a sparse SPD system.
///
/// These are *expected* outcomes for malformed or numerically unsuitable input,
/// so they surface as a `Result` rather than a panic.
#[derive(Debug)]
pub enum SparseSolveError {
    /// A triplet referenced a row/column index `>= n`, or the matrix could not
    /// be assembled (allocation / index overflow).
    Assembly(String),
    /// The matrix is not positive definite, so no Cholesky factor exists (a
    /// zero/negative pivot was encountered), or the factorisation failed for a
    /// non-numeric reason.
    NotPositiveDefinite(String),
    /// A right-hand-side column had a length other than `n`.
    DimensionMismatch { expected: usize, found: usize },
}

impl std::fmt::Display for SparseSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseSolveError::Assembly(msg) => write!(f, "sparse assembly failed: {msg}"),
            SparseSolveError::NotPositiveDefinite(msg) => {
                write!(f, "matrix is not positive definite: {msg}")
            }
            SparseSolveError::DimensionMismatch { expected, found } => write!(
                f,
                "right-hand-side length {found} does not match system size {expected}"
            ),
        }
    }
}

impl std::error::Error for SparseSolveError {}

impl SparseCholesky {
    /// Assemble an `n`×`n` SPD matrix from COO `triplets` and factor it.
    ///
    /// Pass the full symmetric matrix (for each off-diagonal pair supply both
    /// `(i, j)` and `(j, i)`); the lower triangle is read. Duplicate
    /// `(row, col)` entries are summed.
    ///
    /// Returns [`SparseSolveError::Assembly`] for out-of-range indices and
    /// [`SparseSolveError::NotPositiveDefinite`] when the system has no Cholesky
    /// factor.
    pub fn factor(dim: usize, triplets: &[(usize, usize, f64)]) -> Result<Self, SparseSolveError> {
        let entries: Vec<Triplet<usize, usize, f64>> = triplets
            .iter()
            .map(|&(row, col, value)| Triplet::new(row, col, value))
            .collect();

        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(dim, dim, &entries)
            .map_err(|err| SparseSolveError::Assembly(format!("{err:?}")))?;

        let factor = matrix
            .sp_cholesky(Side::Lower)
            .map_err(|err| SparseSolveError::NotPositiveDefinite(format!("{err:?}")))?;

        Ok(SparseCholesky { factor, dim })
    }

    /// The number of unknowns `n` in the factored system.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Solve `A·x = b` for a single right-hand-side column.
    ///
    /// `rhs` must have length [`dim`](Self::dim). Returns the solution vector.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, SparseSolveError> {
        let mut solutions = self.solve_columns(std::slice::from_ref(&rhs))?;
        Ok(solutions.pop().expect("one column in, one column out"))
    }

    /// Solve `A·X = B` for several right-hand-side columns at once, reusing the
    /// single factorisation.
    ///
    /// Each element of `columns` is one RHS vector of length [`dim`](Self::dim);
    /// the returned `Vec` holds one solution vector per input column, in order.
    /// This is the multi-RHS path the Laplacian operator uses to solve the x/y/z
    /// position components against one factor.
    pub fn solve_columns(&self, columns: &[&[f64]]) -> Result<Vec<Vec<f64>>, SparseSolveError> {
        for column in columns {
            if column.len() != self.dim {
                return Err(SparseSolveError::DimensionMismatch {
                    expected: self.dim,
                    found: column.len(),
                });
            }
        }
        if columns.is_empty() {
            return Ok(Vec::new());
        }

        let ncols = columns.len();
        let mut rhs = Mat::<f64>::from_fn(self.dim, ncols, |row, col| columns[col][row]);

        self.factor.solve_in_place(rhs.as_mut());

        Ok((0..ncols)
            .map(|col| (0..self.dim).map(|row| rhs[(row, col)]).collect())
            .collect())
    }
}
