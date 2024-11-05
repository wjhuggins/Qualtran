#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pytest
import sympy
import numpy as np

from qualtran.bloqs import basic_gates, mcmt, rotations
from qualtran.bloqs.basic_gates import Hadamard, TGate, Toffoli
from qualtran.bloqs.basic_gates._shims import Measure
from qualtran.bloqs.for_testing.costing import make_example_costing_bloqs
from qualtran.bloqs.mcmt import MultiAnd, MultiTargetCNOT
from qualtran.cirq_interop.t_complexity_protocol import TComplexity
from qualtran.resource_counting import DetailedGateCounts, SurfaceCodeGatesCost, get_cost_value


def test_detailed_gate_counts():
    gc = DetailedGateCounts(t=100, toffoli=13)
    assert str(gc) == 't: 100, toffoli: 13'
    assert gc.asdict() == {'t': 100, 'toffoli': 13}

    assert DetailedGateCounts(t=10) * 2 == DetailedGateCounts(t=20)
    assert 2 * DetailedGateCounts(t=10) == DetailedGateCounts(t=20)

    gc2 = DetailedGateCounts(t=sympy.Symbol('n'), toffoli=sympy.sympify('0'), cswap=2)
    assert str(gc2) == 't: n, cswap: 2'


def test_surface_code_gates_cost():
    algo = make_example_costing_bloqs()
    gc = get_cost_value(algo, SurfaceCodeGatesCost())

    assert gc == DetailedGateCounts(toffoli=100, t=2 * 2 * 10, hadamard=2 * 10)

@pytest.mark.parametrize(
    ['bloq'],
    [
        [basic_gates.XGate(),],
        [basic_gates.ZGate(),],
        [basic_gates.Rx(angle=-1.0 * np.pi),],
        [basic_gates.Ry(angle=0.0),],
        [basic_gates.ZPowGate(exponent=0.0),],
        [basic_gates.XPowGate(exponent=-2.0),],
    ],
)
def test_surface_code_gates_cost_paulis(bloq):
    gc = get_cost_value(bloq, SurfaceCodeGatesCost())

    assert gc == DetailedGateCounts()


def test_surface_code_gates_cost_cbloq():
    bloq = MultiAnd(cvs=(1,) * 5)
    cbloq = bloq.decompose_bloq()
    assert get_cost_value(bloq, SurfaceCodeGatesCost()) == get_cost_value(cbloq, SurfaceCodeGatesCost())


@pytest.mark.parametrize(
    ['bloq', 'counts'],
    [
        # T Gate
        [basic_gates.TGate(is_adjoint=False), DetailedGateCounts(t=1)],
        # Toffoli
        [basic_gates.Toffoli(), DetailedGateCounts(toffoli=1)],
        # Measure
        [Measure(), DetailedGateCounts(measurement_total_qubits=1)],
        # CSwap
        [basic_gates.TwoBitCSwap(), DetailedGateCounts(cswap=1)],
        # And
        [mcmt.And(), DetailedGateCounts(and_bloq=1)],
        # Rotations
        [basic_gates.ZPowGate(exponent=0.1, global_shift=0.0, eps=1e-11), DetailedGateCounts(rotation=1)],
        [
            rotations.phase_gradient.PhaseGradientUnitary(
                bitsize=10, exponent=1, is_controlled=False, eps=1e-10
            ),
            # TODO: Verify that this is the correct way to count the costs of this example.
            DetailedGateCounts(t=1, rotation=7, arbitrary_1q_clifford=1),
        ],
        # Recursive
        # TODO: Verify that this is the correct way to count the costs of this example.
        [mcmt.MultiControlX(cvs=(1, 1, 1)), DetailedGateCounts(and_bloq=2, measurement_total_qubits=2, cnot=1)],
    ],
)
def test_get_cost_value_qec_gates_cost(bloq, counts):
    print(get_cost_value(bloq, SurfaceCodeGatesCost()))
    print(counts)
    assert get_cost_value(bloq, SurfaceCodeGatesCost()) == counts


def test_count_multi_target_cnot():
    b = MultiTargetCNOT(bitsize=12)

    assert get_cost_value(b, SurfaceCodeGatesCost()) == DetailedGateCounts(
        multi_target_pauli=1, 
        multi_target_pauli_total_targets=12)