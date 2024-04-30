// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use ndarray::array;
use num_complex::Complex64;
use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

// Compute `kron(identity, mat)` for 2x2 matrix inputs
fn kron_id2_oneq(oneq_mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    let _zero = Complex64::new(0., 0.);
    array![
        [oneq_mat[[0, 0]], oneq_mat[[0, 1]], _zero, _zero],
        [oneq_mat[[1, 0]], oneq_mat[[1, 1]], _zero, _zero],
        [_zero, _zero, oneq_mat[[0, 0]], oneq_mat[[0, 1]]],
        [_zero, _zero, oneq_mat[[1, 0]], oneq_mat[[1, 1]]],
    ]
}

// Compute `kron(mat, identity)` for 2x2 matrix inputs
fn kron_oneq_id2(oneq_mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    let _zero = Complex64::new(0., 0.);
    array![
        [oneq_mat[[0, 0]], _zero, oneq_mat[[0, 1]], _zero],
        [_zero, oneq_mat[[0, 0]], _zero, oneq_mat[[0, 1]]],
        [oneq_mat[[1, 0]], _zero, oneq_mat[[1, 1]], _zero],
        [_zero, oneq_mat[[1, 0]], _zero, oneq_mat[[1, 1]]],
    ]
}

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let input_matrix = op_list[0].0.as_array();
    let mut matrix: Array2<Complex64> = match op_list[0].1.as_slice() {
        [0] => kron_id2_oneq(input_matrix),
        [1] => kron_oneq_id2(input_matrix),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis(input_matrix),
        [] => Array2::eye(4),
        _ => unreachable!(),
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array();

        let result = match q_list.as_slice() {
            [0] => Some(kron_id2_oneq(op_matrix)),
            [1] => Some(kron_oneq_id2(op_matrix)),
            [1, 0] => Some(change_basis(op_matrix)),
            [] => Some(Array2::eye(4)),
            _ => None,
        };
        matrix = match result {
            Some(result) => result.dot(&matrix),
            None => op_matrix.dot(&matrix),
        };
    }
    Ok(matrix.into_pyarray(py).to_owned())
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
