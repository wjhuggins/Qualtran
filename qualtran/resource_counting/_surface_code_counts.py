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
import logging
import warnings
from collections import defaultdict
from typing import Callable, cast, Dict, Sequence, Tuple, TYPE_CHECKING

import attrs
import networkx as nx
import sympy
from attrs import field, frozen

from qualtran.symbolics import is_zero, SymbolicInt

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey
from .classify_bloqs import (
    bloq_is_single_qubit_pauli,
    bloq_is_single_qubit_clifford,
    bloq_is_two_qubit_clifford,
    bloq_is_rotation,
    bloq_is_state_or_effect,
    bloq_is_t_like,
)

if TYPE_CHECKING:
    from qualtran import Bloq
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity

logger = logging.getLogger(__name__)


@frozen(kw_only=True)
class DetailedGateCounts:
    """A data class of counts of the typical target gates in a compilation.

    Specifically, this class holds counts for the number of `TGate` (and adjoint), `Toffoli`,
    `TwoBitCSwap`, `And`, clifford bloqs, single qubit rotations, and measurements.
    """

    t: SymbolicInt = 0
    toffoli: SymbolicInt = 0
    cswap: SymbolicInt = 0
    and_bloq: SymbolicInt = 0
    rotation: SymbolicInt = 0
    hadamard: SymbolicInt = 0
    s_gate: SymbolicInt = 0
    cnot: SymbolicInt = 0
    arbitrary_1q_clifford: SymbolicInt = 0
    arbitrary_2q_clifford: SymbolicInt = 0
    multi_target_pauli: SymbolicInt = 0
    multi_target_pauli_total_targets: SymbolicInt = 0
    measurement_total_qubits: SymbolicInt = 0
    other_bloqs: Dict['Bloq', SymbolicInt] = {}

    def __add__(self, other):
        if not isinstance(other, DetailedGateCounts):
            raise TypeError(f"Can only add other `DetailedGateCounts` objects, not {self}")

        other_bloqs_sum = self.other_bloqs.copy()
        for bloq, count in other.other_bloqs.items():
            other_bloqs_sum[bloq] = other_bloqs_sum.get(bloq, 0) + count

        return DetailedGateCounts(
            t=self.t + other.t,
            toffoli=self.toffoli + other.toffoli,
            cswap=self.cswap + other.cswap,
            and_bloq=self.and_bloq + other.and_bloq,
            rotation=self.rotation + other.rotation,
            hadamard=self.hadamard + other.hadamard,
            s_gate=self.s_gate + other.s_gate,
            cnot=self.cnot + other.cnot,
            arbitrary_1q_clifford=self.arbitrary_1q_clifford + other.arbitrary_1q_clifford,
            arbitrary_2q_clifford=self.arbitrary_2q_clifford + other.arbitrary_2q_clifford,
            multi_target_pauli=self.multi_target_pauli + other.multi_target_pauli,
            multi_target_pauli_total_targets=self.multi_target_pauli_total_targets
            + other.multi_target_pauli_total_targets,
            measurement_total_qubits=self.measurement_total_qubits + other.measurement_total_qubits,
            other_bloqs=other_bloqs_sum,
        )

    def __mul__(self, other):
        new_other_bloqs = {}
        for bloq, count in self.other_bloqs.items():
            new_other_bloqs[bloq] = count * other

        return DetailedGateCounts(
            t=other * self.t,
            toffoli=other * self.toffoli,
            cswap=other * self.cswap,
            and_bloq=other * self.and_bloq,
            rotation=other * self.rotation,
            hadamard=other * self.hadamard,
            s_gate=other * self.s_gate,
            cnot=other * self.cnot,
            arbitrary_1q_clifford=other * self.arbitrary_1q_clifford,
            arbitrary_2q_clifford=other * self.arbitrary_2q_clifford,
            multi_target_pauli=other * self.multi_target_pauli,
            multi_target_pauli_total_targets=other * self.multi_target_pauli_total_targets,
            measurement_total_qubits=other * self.measurement_total_qubits,
            other_bloqs=new_other_bloqs,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        strs = [f'{k}: {v}' for k, v in self.asdict().items()]
        if strs:
            return ', '.join(strs)
        return '-'

    def asdict(self) -> Dict[str, int]:
        d = attrs.asdict(self, recurse=False)

        def _is_nonzero(v):
            maybe_nonzero = sympy.sympify(v)
            if maybe_nonzero is None:
                return True
            return maybe_nonzero

        return_dict = {}
        for k, v in d.items():
            if k != "other_bloqs" and _is_nonzero(v):
                return_dict[k] = v
            elif k == "other_bloqs":
                if len(v) > 0:
                    return_dict[k] = v

        return return_dict
        # return {k: v for k, v in d.items() if _is_nonzero(v)}

    def lattice_surgery_spacetime_volume(
        self,
        volume_per_t: SymbolicInt = sympy.Symbol('volume_per_t'),
        volume_per_toffoli: SymbolicInt = sympy.Symbol('volume_per_toffoli'),
        volume_per_cswap: SymbolicInt = sympy.Symbol('volume_per_cswap'),
        volume_per_and_bloq: SymbolicInt = sympy.Symbol('volume_per_and_bloq'),
        volume_per_rotation: SymbolicInt = sympy.Symbol('volume_per_rotation'),
    ) -> SymbolicInt:
        # TODO: Implement this.
        pass


@frozen(kw_only=True)
class SurfaceCodeGatesCost(CostKey[DetailedGateCounts]):
    """Counts the gates in a surface code error correction scheme in detail.

    The cost value type for this CostKey is `DetailedGateCounts`.
    """

    def compute(
        self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], DetailedGateCounts]
    ) -> DetailedGateCounts:
        from qualtran.bloqs.basic_gates import Hadamard, SGate, CNOT
        from qualtran.bloqs.basic_gates import GlobalPhase, Identity, Toffoli, TwoBitCSwap
        from qualtran.bloqs.basic_gates._shims import Measure
        from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
        from qualtran.bloqs.mcmt import And, MultiTargetCNOT

        # Pauli gates:
        if bloq_is_single_qubit_pauli(bloq):
            return DetailedGateCounts()

        # T gates
        if bloq_is_t_like(bloq):
            return DetailedGateCounts(t=1)

        # Hadamard
        if isinstance(bloq, Hadamard):
            return DetailedGateCounts(hadamard=1)

        # s gate
        if isinstance(bloq, SGate):
            return DetailedGateCounts(s_gate=1)

        # Toffolis
        if isinstance(bloq, Toffoli):
            return DetailedGateCounts(toffoli=1)

        # Measurement
        if isinstance(bloq, Measure):
            # TODO: Make sure this counts multi-qubit measurements correctly.
            return DetailedGateCounts(measurement_total_qubits=1)

        # CNOT
        if isinstance(bloq, CNOT):
            return DetailedGateCounts(cnot=1)

        # 'And' bloqs
        if isinstance(bloq, And):
            if bloq.uncompute:
                # TODO: Double check that this is correct.
                return DetailedGateCounts(measurement_total_qubits=1)

            return DetailedGateCounts(and_bloq=1)

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            return DetailedGateCounts(cswap=1)

        if isinstance(bloq, MultiTargetCNOT):
            return DetailedGateCounts(
                multi_target_pauli=1, multi_target_pauli_total_targets=bloq.bitsize
            )

        # Cliffords
        if bloq_is_single_qubit_clifford(bloq):
            return DetailedGateCounts(arbitrary_1q_clifford=1)
        if bloq_is_two_qubit_clifford(bloq):
            return DetailedGateCounts(arbitrary_2q_clifford=2)

        # TODO: Deal with the other Clifford cases.

        # States and effects
        if bloq_is_state_or_effect(bloq):
            # TODO: Verify that this behaviour is correct in all cases...
            # Maybe we should count states since initialization can have a cost? Especially Y basis...
            return DetailedGateCounts()

        # Bookkeeping, empty bloqs
        if isinstance(bloq, _BookkeepingBloq) or isinstance(bloq, (GlobalPhase, Identity)):
            return DetailedGateCounts()

        if bloq_is_rotation(bloq):
            # TODO: Figure out how to correctly count rotations. Y basis might be more expensive than X/Z for example...
            # Maybe we count them separately for now?
            return DetailedGateCounts(rotation=1)

        callees = get_bloq_callee_counts(bloq, ignore_decomp_failure=True)
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))

        # Case where bloq is atomic but unrecognized.
        if len(callees) == 0:
            return DetailedGateCounts(other_bloqs={bloq: 1})

        # Recursive case
        totals = DetailedGateCounts()
        for callee, n_times_called in callees:
            callee_cost = get_callee_cost(callee)
            totals += n_times_called * callee_cost
        return totals

    def zero(self) -> DetailedGateCounts:
        return DetailedGateCounts()

    def validate_val(self, val: DetailedGateCounts):
        if not isinstance(val, DetailedGateCounts):
            raise TypeError(f"{self} values should be `DetailedGateCounts`, got {val}")

    def __str__(self):
        return 'detailed gate counts'
