// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// In numpy matrices real and imaginary components are adjacent:
//   np.array([1,2,3], dtype='complex').view('float64')
//   array([1., 0., 2., 0., 3., 0.])
// The matrix faer::Mat<c64> has this layout.
// faer::Mat<num_complex::Complex<f64>> instead stores a matrix
// of real components and one of imaginary components.
// In order to avoid copying we want to use `MatRef<c64>` or `MatMut<c64>`.

use num_complex::{Complex64, Complex};
use std::f64::consts::FRAC_1_SQRT_2;

// This is almost the same as the function that became available in
// num-complex 0.4.6. The difference is that two generic parameters are
// used here rather than one. This allows call like `c64(half_theta.cos(), 0);`
// that mix f64 and integer arguments.
/// Create a new [`Complex<f64>`] with arguments that can convert [`Into<f64>`].
///
/// ```
/// use num_complex::{c64, Complex64};
/// assert_eq!(c64(1, 2), Complex64::new(1.0, 2.0));
/// ```
#[inline]
fn c64<T: Into<f64>, V: Into<f64>>(re: T, im: V) -> Complex64 {
    Complex::new(re.into(), im.into())
}

// Many computations are not avaialable when these `const`s are compiled.

// ZERO and ONE are defined in num_complex 0.4.6
const ZERO: Complex64 = Complex64::new(0., 0.);
const ONE: Complex64 = Complex64::new(1., 0.);
const M_ONE: Complex64 = Complex64::new(-1., 0.);
const IM: Complex64 = Complex64::new(0., 1.);
const M_IM: Complex64 = Complex64::new(0., -1.);

pub static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [[ONE, ZERO], [ZERO, ONE]];

pub fn rx_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0);
    let isin = c64(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

pub fn ry_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0);
    let sin = c64(half_theta.sin(), 0);
    [[cos, -sin], [sin, cos]]
}

pub fn rz_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let ilam2 = c64(0, 0.5 * theta);
    [[(-ilam2).exp(), ZERO], [ZERO, ilam2.exp()]]
}

pub static HGATE: [[Complex64; 2]; 2] = [
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(-FRAC_1_SQRT_2, 0.),
    ],
];

pub static CXGATE: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ONE, ZERO, ZERO],
];

pub static SXGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5)],
    [Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5)],
];

pub static XGATE: [[Complex64; 2]; 2] = [[ZERO, ONE], [ONE, ZERO]];

pub static ZGATE: [[Complex64; 2]; 2] = [[ONE, ZERO], [ZERO, M_ONE]];

pub static YGATE: [[Complex64; 2]; 2] = [[M_IM, ZERO], [IM, ZERO]];

pub static CZGATE: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, M_ONE],
];

pub static CYGATE: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, M_IM],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, IM, ZERO, ZERO],
];

pub static CCXGATE: [[Complex64; 8]; 8] = [
    [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO],
];

pub static ECRGATE: [[Complex64; 4]; 4] = [
    [
        ZERO,
        Complex64::new(FRAC_1_SQRT_2, 0.),
        ZERO,
        Complex64::new(0., FRAC_1_SQRT_2),
    ],
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        ZERO,
        Complex64::new(0., -FRAC_1_SQRT_2),
        ZERO,
    ],
    [
        ZERO,
        Complex64::new(0., FRAC_1_SQRT_2),
        ZERO,
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(0., -FRAC_1_SQRT_2),
        ZERO,
        Complex64::new(FRAC_1_SQRT_2, 0.),
        ZERO,
    ],
];

pub static SWAPGATE: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
];

pub fn global_phase_gate(theta: f64) -> [[Complex64; 1]; 1] {
    [[c64(0., theta).exp()]]
}

pub fn phase_gate(lam: f64) -> [[Complex64; 2]; 2] {
    [[ONE, ZERO], [ZERO, c64(0., lam).exp()]]
}
