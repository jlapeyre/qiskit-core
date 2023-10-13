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

use std::env;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;

mod convert_2q_block_matrix;
mod dense_layout;
mod edge_collections;
mod error_map;
mod euler_one_qubit_decomposer;
mod nlayout;
mod optimize_1q_gates;
mod pauli_exp_val;
mod results;
mod sabre_layout;
mod sabre_swap;
mod sampled_exp_val;
mod sparse_pauli_op;
mod stochastic_swap;
mod vf2_layout;
mod two_qubit_decompose;

#[inline]
pub fn getenv_use_multiple_threads() -> bool {
    let parallel_context = env::var("QISKIT_IN_PARALLEL")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    let force_threads = env::var("QISKIT_FORCE_THREADS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    !parallel_context || force_threads
}

#[pymodule]
fn _accelerate(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(nlayout::nlayout))?;
    m.add_wrapped(wrap_pymodule!(stochastic_swap::stochastic_swap))?;
    m.add_wrapped(wrap_pymodule!(sabre_swap::sabre_swap))?;
    m.add_wrapped(wrap_pymodule!(pauli_exp_val::pauli_expval))?;
    m.add_wrapped(wrap_pymodule!(dense_layout::dense_layout))?;
    m.add_wrapped(wrap_pymodule!(error_map::error_map))?;
    m.add_wrapped(wrap_pymodule!(sparse_pauli_op::sparse_pauli_op))?;
    m.add_wrapped(wrap_pymodule!(results::results))?;
    m.add_wrapped(wrap_pymodule!(optimize_1q_gates::optimize_1q_gates))?;
    m.add_wrapped(wrap_pymodule!(sampled_exp_val::sampled_exp_val))?;
    m.add_wrapped(wrap_pymodule!(sabre_layout::sabre_layout))?;
    m.add_wrapped(wrap_pymodule!(vf2_layout::vf2_layout))?;
    m.add_wrapped(wrap_pymodule!(
        euler_one_qubit_decomposer::euler_one_qubit_decomposer
    ))?;
    m.add_wrapped(wrap_pymodule!(
        convert_2q_block_matrix::convert_2q_block_matrix
    ))?;
    Ok(())
}
