use crate::xx_decompose::utilities::{safe_acos, Square};

fn decompose_xxyy_into_xxyy_xx(
    a_target: f64,
    b_target: f64,
    a_source: f64,
    b_source: f64,
    interaction: f64,
) {
    let cplus = (a_source + b_source).cos();
    let cminus = (a_source - b_source).cos();
    let splus = (a_source + b_source).sin();
    let sminus = (a_source - b_source).sin();
    let ca = interaction.cos();
    let sa = interaction.sin();

    let uplusv =
        1. / 2. *
        safe_acos(cminus.sq() * ca.sq() + sminus.sq() * sa.sq() - (a_target - b_target).cos().sq(),
                  2. * cminus * ca * sminus * sa);

    let uminusv =
        1. / 2.
        * safe_acos(
            cplus.sq() * ca.sq() + splus.sq() * sa.sq() - (a_target + b_target).cos().sq(),
            2. * cplus * ca * splus * sa,
        );

    let (u, v) = ((uplusv + uminusv) / 2., (uplusv - uminusv) / 2.);
}

//     cplus, cminus = np.cos(a_source + b_source), np.cos(a_source - b_source)
//     splus, sminus = np.sin(a_source + b_source), np.sin(a_source - b_source)
//     ca, sa = np.cos(interaction), np.sin(interaction)

//     uplusv = (
//         1
//         / 2
//         * safe_arccos(
//             cminus**2 * ca**2 + sminus**2 * sa**2 - np.cos(a_target - b_target) ** 2,
//             2 * cminus * ca * sminus * sa,
//         )
//     )
//     uminusv = (
//         1
//         / 2
//         * safe_arccos(
//             cplus**2 * ca**2 + splus**2 * sa**2 - np.cos(a_target + b_target) ** 2,
//             2 * cplus * ca * splus * sa,
//         )
//     )

//     u, v = (uplusv + uminusv) / 2, (uplusv - uminusv) / 2

//     # NOTE: the target matrix is phase-free
//     middle_matrix = reduce(
//         np.dot,
//         [
//             RXXGate(2 * a_source).to_matrix() @ RYYGate(2 * b_source).to_matrix(),
//             np.kron(RZGate(2 * u).to_matrix(), RZGate(2 * v).to_matrix()),
//             RXXGate(2 * interaction).to_matrix(),
//         ],
//     )

//     phase_solver = np.array(
//         [
//             [
//                 1 / 4,
//                 1 / 4,
//                 1 / 4,
//                 1 / 4,
//             ],
//             [
//                 1 / 4,
//                 -1 / 4,
//                 -1 / 4,
//                 1 / 4,
//             ],
//             [
//                 1 / 4,
//                 1 / 4,
//                 -1 / 4,
//                 -1 / 4,
//             ],
//             [
//                 1 / 4,
//                 -1 / 4,
//                 1 / 4,
//                 -1 / 4,
//             ],
//         ]
//     )
//     inner_phases = [
//         np.angle(middle_matrix[0, 0]),
//         np.angle(middle_matrix[1, 1]),
//         np.angle(middle_matrix[1, 2]) + np.pi / 2,
//         np.angle(middle_matrix[0, 3]) + np.pi / 2,
//     ]
//     r, s, x, y = np.dot(phase_solver, inner_phases)

//     # If there's a phase discrepancy, need to conjugate by an extra Z/2 (x) Z/2.
//     generated_matrix = reduce(
//         np.dot,
//         [
//             np.kron(RZGate(2 * r).to_matrix(), RZGate(2 * s).to_matrix()),
//             middle_matrix,
//             np.kron(RZGate(2 * x).to_matrix(), RZGate(2 * y).to_matrix()),
//         ],
//     )
//     if (abs(np.angle(generated_matrix[3, 0]) - np.pi / 2) < 0.01 and a_target > b_target) or (
//         abs(np.angle(generated_matrix[3, 0]) + np.pi / 2) < 0.01 and a_target < b_target
//     ):
//         x += np.pi / 4
//         y += np.pi / 4
//         r -= np.pi / 4
//         s -= np.pi / 4

//     return r, s, u, v, x, y
