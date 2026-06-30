//! Solver-isolation unit tests (prerequisite testing mode 2): pin the `faer`
//! sparse Cholesky against an independent dense Gaussian-elimination reference
//! on small SPD systems, and verify `A·x ≈ b` directly.

use super::sparse_solve::SparseCholesky;

const TOL: f64 = 1e-10;

/// Dense reference solve of `A·x = b` by Gaussian elimination with partial
/// pivoting. Deliberately independent of `faer` so it can cross-check it.
fn dense_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut a: Vec<Vec<f64>> = matrix.iter().cloned().collect();
    let mut b: Vec<f64> = rhs.to_vec();

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        assert!(a[col][col].abs() > 1e-14, "singular reference matrix");

        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for k in (row + 1)..n {
            acc -= a[row][k] * x[k];
        }
        x[row] = acc / a[row][row];
    }
    x
}

/// COO triplets for the full symmetric matrix (both off-diagonal entries).
fn dense_to_triplets(matrix: &[Vec<f64>]) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::new();
    for (row, cells) in matrix.iter().enumerate() {
        for (col, &value) in cells.iter().enumerate() {
            if value != 0.0 {
                triplets.push((row, col, value));
            }
        }
    }
    triplets
}

fn residual_max(matrix: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
    let n = b.len();
    let mut worst = 0.0_f64;
    for row in 0..n {
        let mut acc = 0.0;
        for col in 0..n {
            acc += matrix[row][col] * x[col];
        }
        worst = worst.max((acc - b[row]).abs());
    }
    worst
}

fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < TOL, "expected {y}, got {x}");
    }
}

#[test]
fn solve_2x2() {
    let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
    let rhs = vec![1.0, 2.0];

    let factor = SparseCholesky::factor(2, &dense_to_triplets(&matrix)).unwrap();
    let x = factor.solve(&rhs).unwrap();

    assert_close(&x, &dense_solve(&matrix, &rhs));
    assert!(residual_max(&matrix, &x, &rhs) < TOL);
}

#[test]
fn solve_3x3_with_off_diagonals() {
    // SPD: diagonally dominant, symmetric, all positive pivots.
    let matrix = vec![
        vec![6.0, 2.0, 1.0],
        vec![2.0, 5.0, 2.0],
        vec![1.0, 2.0, 4.0],
    ];
    let rhs = vec![7.0, -3.0, 5.0];

    let factor = SparseCholesky::factor(3, &dense_to_triplets(&matrix)).unwrap();
    let x = factor.solve(&rhs).unwrap();

    assert_close(&x, &dense_solve(&matrix, &rhs));
    assert!(residual_max(&matrix, &x, &rhs) < TOL);
}

#[test]
fn solve_diagonal_system() {
    let matrix = vec![
        vec![2.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0],
        vec![0.0, 0.0, 0.25],
    ];
    let rhs = vec![4.0, 10.0, 1.0];

    let factor = SparseCholesky::factor(3, &dense_to_triplets(&matrix)).unwrap();
    let x = factor.solve(&rhs).unwrap();

    // Trivial: x_i = b_i / a_ii.
    assert_close(&x, &[2.0, 2.0, 4.0]);
    assert!(residual_max(&matrix, &x, &rhs) < TOL);
}

#[test]
fn solve_multi_rhs_reuses_factor() {
    // The 3-RHS shape the Laplacian operator uses (x/y/z components).
    let matrix = vec![
        vec![10.0, 1.0, 2.0],
        vec![1.0, 8.0, 3.0],
        vec![2.0, 3.0, 7.0],
    ];
    let cols = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![3.0, -1.0, 2.0],
    ];

    let factor = SparseCholesky::factor(3, &dense_to_triplets(&matrix)).unwrap();
    let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    let solutions = factor.solve_columns(&col_refs).unwrap();

    assert_eq!(solutions.len(), 3);
    for (solution, rhs) in solutions.iter().zip(&cols) {
        assert_close(solution, &dense_solve(&matrix, rhs));
        assert!(residual_max(&matrix, solution, rhs) < TOL);
    }
}

#[test]
fn duplicate_triplets_sum() {
    // Two triplets at (0,0) summing to 4.0; off-diagonals supplied once each side.
    let triplets = vec![
        (0usize, 0usize, 1.5),
        (0, 0, 2.5),
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 3.0),
    ];
    let factor = SparseCholesky::factor(2, &triplets).unwrap();
    let x = factor.solve(&[1.0, 2.0]).unwrap();

    let matrix = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
    assert_close(&x, &dense_solve(&matrix, &[1.0, 2.0]));
}

#[test]
fn non_spd_matrix_is_rejected() {
    // Indefinite (negative eigenvalue): no Cholesky factor exists.
    let matrix = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
    let result = SparseCholesky::factor(2, &dense_to_triplets(&matrix));
    assert!(matches!(
        result,
        Err(super::SparseSolveError::NotPositiveDefinite(_))
    ));
}

#[test]
fn rhs_length_mismatch_is_rejected() {
    let factor = SparseCholesky::factor(2, &[(0, 0, 1.0), (1, 1, 1.0)]).unwrap();
    let result = factor.solve(&[1.0, 2.0, 3.0]);
    assert!(matches!(
        result,
        Err(super::SparseSolveError::DimensionMismatch {
            expected: 2,
            found: 3
        })
    ));
}
