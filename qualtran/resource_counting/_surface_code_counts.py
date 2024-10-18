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
from qualtran.bloqs.bookkeeping.arbitrary_clifford import ArbitraryClifford
from qualtran.bloqs.mcmt import multi_control_pauli
import sympy
from attrs import field, frozen

from qualtran.symbolics import is_zero, SymbolicInt

from ._call_graph import get_bloq_callee_counts
from ._costing import CostKey
from .classify_bloqs import (
    bloq_is_clifford,
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
    multi_control_pauli_count: SymbolicInt = 0
    multi_control_pauli_total_targets: SymbolicInt = 0
    measurement_total_qubits: SymbolicInt = 0

    def __add__(self, other):
        if not isinstance(other, DetailedGateCounts):
            raise TypeError(f"Can only add other `DetailedGateCounts` objects, not {self}")

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
            multi_control_pauli_count=self.multi_control_pauli_count + other.multi_control_pauli_count,
            multi_control_pauli_total_targets=self.multi_control_pauli_total_targets + other.multi_control_pauli_total_targets,
            measurement_total_qubits=self.measurement_total_qubits + other.measurement_total_qubits,
        )

    def __mul__(self, other):
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
            multi_control_pauli_count=other * self.multi_control_pauli_count,
            multi_control_pauli_total_targets=other * self.multi_control_pauli_total_targets,
            measurement_total_qubits=other * self.measurement_total_qubits,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        strs = [f'{k}: {v}' for k, v in self.asdict().items()]
        if strs:
            return ', '.join(strs)
        return '-'

    def asdict(self) -> Dict[str, int]:
        d = attrs.asdict(self)

        def _is_nonzero(v):
            maybe_nonzero = sympy.sympify(v)
            if maybe_nonzero is None:
                return True
            return maybe_nonzero

        return {k: v for k, v in d.items() if _is_nonzero(v)}
    
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

    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], DetailedGateCounts]) -> DetailedGateCounts:
        from qualtran.bloqs.basic_gates import Hadamard, SGate, CNOT
        from qualtran.bloqs.basic_gates import GlobalPhase, Identity, Toffoli, TwoBitCSwap
        from qualtran.bloqs.basic_gates._shims import Measure
        from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
        from qualtran.bloqs.mcmt import And, MultiTargetCNOT

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
            # To match the legacy `t_complexity` protocol, we can hack in the explicit
            # counts for the clifford operations used to invert the control bit.
            # Note: we *only* add in the clifford operations that correspond to correctly
            # setting the control line. The other clifford operations inherent in compiling
            # an And gate to the gateset considered by the legacy `t_complexity` protocol can be
            # simply added in as part of `GateCounts.to_legacy_t_complexity()`

            # TODO: Add special AND uncomputation counts, as in commented out code below.
            # n_inverted_controls = (bloq.cv1 == 0) + int(bloq.cv2 == 0)
            # if bloq.uncompute:
            #     if self.legacy_shims:
            #         return GateCounts(clifford=3 + 2 * n_inverted_controls, measurement=1)
            #     else:
            #         return GateCounts(measurement=1, clifford=1)

            # if self.legacy_shims:
            #     return GateCounts(and_bloq=1, clifford=2 * n_inverted_controls)
            # else:
            #     return GateCounts(and_bloq=1)
            return DetailedGateCounts(and_bloq=1)

        # CSwaps aka Fredkin
        if isinstance(bloq, TwoBitCSwap):
            return DetailedGateCounts(cswap=1)

        if isinstance(bloq, MultiTargetCNOT):
            return DetailedGateCounts(multi_control_pauli_count=1, multi_control_pauli_total_targets=bloq.bitsize)

        # Cliffords
        if isinstance(bloq, ArbitraryClifford):
            if bloq.n == 1:
                return DetailedGateCounts(arbitrary_1q_clifford=1)
            elif bloq.n == 2:
                return DetailedGateCounts(arbitrary_2q_clifford=1)

        # TODO: Deal with the other Clifford cases.
        # if bloq_is_clifford(bloq):
        #     return GateCounts(clifford=1)

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

        # Recursive case
        totals = DetailedGateCounts()
        callees = get_bloq_callee_counts(bloq, ignore_decomp_failure=False)
        logger.info("Computing %s for %s from %d callee(s)", self, bloq, len(callees))
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
