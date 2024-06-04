// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This module contains objects imported from Python that are reused. These are
// typically data model classes that are used to identify an object, or for
// python side casting

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

use crate::operations::{StandardGate, STANDARD_GATE_SIZE};

/// builtin list
pub static BUILTIN_LIST: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.operation.Operation
pub static OPERATION: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.instruction.Instruction
pub static INSTRUCTION: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.gate.Gate
pub static GATE: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.quantumregister.Qubit
pub static QUBIT: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.classicalregister.Clbit
pub static CLBIT: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.parameterexpression.ParameterExpression
pub static PARAMETER_EXPRESSION: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.quantumcircuit.QuantumCircuit
pub static QUANTUM_CIRCUIT: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.singleton.SingletonGate
pub static SINGLETON_GATE: GILOnceCell<PyObject> = GILOnceCell::new();
/// qiskit.circuit.singleton.SingletonControlledGate
pub static SINGLETON_CONTROLLED_GATE: GILOnceCell<PyObject> = GILOnceCell::new();

/// A mapping from the enum variant in crate::operations::StandardGate to the python
/// module path and class name to import it. This is used to populate the conversion table
/// when a gate is added directly via the StandardGate path and there isn't a Python object
/// to poll the _standard_gate attribute for.
///
/// NOTE: the order here is significant it must match the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
static STDGATE_IMPORT_PATHS: [[&str; 2]; STANDARD_GATE_SIZE] = [
    // ZGate = 0
    ["qiskit.circuit.library.standard_gates.z", "ZGate"],
    // YGate = 1
    ["qiskit.circuit.library.standard_gates.y", "YGate"],
    // XGate = 2
    ["qiskit.circuit.library.standard_gates.x", "XGate"],
    // CZGate = 3
    ["qiskit.circuit.library.standard_gates.z", "CZGate"],
    // CYGate = 4
    ["qiskit.circuit.library.standard_gates.y", "CYGate"],
    // CXGate = 5
    ["qiskit.circuit.library.standard_gates.x", "CXGate"],
    // CCXGate = 6
    ["qiskit.circuit.library.standard_gates.x", "CCXGate"],
    // RXGate = 7
    ["qiskit.circuit.library.standard_gates.rx", "RXGate"],
    // RYGate = 8
    ["qiskit.circuit.library.standard_gates.ry", "RYGate"],
    // RZGate = 9
    ["qiskit.circuit.library.standard_gates.rz", "RZGate"],
    // ECRGate = 10
    ["qiskit.circuit.library.standard_gates.ecr", "ECRGate"],
    // SwapGate = 11
    ["qiskit.circuit.library.standard_gates.swap", "SwapGate"],
    // SXGate = 12
    ["qiskit.circuit.library.standard_gates.sx", "SXGate"],
    // GlobalPhaseGate = 13
    [
        "qiskit.circuit.library.standard_gates.global_phase",
        "GlobalPhaseGate",
    ],
    // IGate = 14
    ["qiskit.circuit.library.standard_gates.i", "IGate"],
    // HGate = 15
    ["qiskit.circuit.library.standard_gates.h", "HGate"],
    // PhaseGate = 16
    ["qiskit.circuit.library.standard_gates.p", "PhaseGate"],
    // UGate = 17
    ["qiskit.circuit.library.standard_gates.u", "UGate"],
];

/// A mapping from the enum variant in crate::operations::StandardGate to the python object for the
/// class that matches it. This is typically used when we need to convert from the internal rust
/// representation to a Python object for a python user to interact with.
///
/// NOTE: the order here is significant it must match the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
static mut STDGATE_PYTHON_GATES: GILOnceCell<[Option<PyObject>; STANDARD_GATE_SIZE]> =
    GILOnceCell::new();

#[inline]
pub fn populate_std_gate_map(py: Python, rs_gate: StandardGate, py_gate: PyObject) {
    let gate_map = unsafe {
        match STDGATE_PYTHON_GATES.get_mut() {
            Some(gate_map) => gate_map,
            None => {
                // A fixed size array is initialized like this because using the `[T; 5]` syntax
                // requires T to be `Copy`. But `PyObject` isn't Copy so therefore Option<PyObject>
                // as T isn't Copy. To avoid that we just list out None STANDARD_GATE_SIZE times
                let array: [Option<PyObject>; STANDARD_GATE_SIZE] = [
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None,
                ];
                STDGATE_PYTHON_GATES.set(py, array).unwrap();
                STDGATE_PYTHON_GATES.get_mut().unwrap()
            }
        }
    };
    let gate_cls = &gate_map[rs_gate as usize];
    if gate_cls.is_none() {
        gate_map[rs_gate as usize] = Some(py_gate.clone_ref(py));
    }
}

#[inline]
pub fn get_std_gate_class(py: Python, rs_gate: StandardGate) -> PyResult<PyObject> {
    let gate_map = unsafe {
        STDGATE_PYTHON_GATES.get_or_init(py, || {
            // A fixed size array is initialized like this because using the `[T; 5]` syntax
            // requires T to be `Copy`. But `PyObject` isn't Copy so therefore Option<PyObject>
            // as T isn't Copy. To avoid that we just list out None STANDARD_GATE_SIZE times
            let array: [Option<PyObject>; STANDARD_GATE_SIZE] = [
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None,
            ];
            array
        })
    };
    let gate = &gate_map[rs_gate as usize];
    let populate = gate.is_none();
    let out_gate = match gate {
        Some(gate) => gate.clone_ref(py),
        None => {
            let [py_mod, py_class] = STDGATE_IMPORT_PATHS[rs_gate as usize];
            py.import_bound(py_mod)?.getattr(py_class)?.unbind()
        }
    };
    if populate {
        populate_std_gate_map(py, rs_gate, out_gate.clone_ref(py));
    }
    Ok(out_gate)
}