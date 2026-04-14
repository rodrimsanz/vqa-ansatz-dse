"""
ansatzes.py  –  Módulo unificado de ansatzes VQE (n=1..19)
==============================================================
Modos de salida:
  mode="expval"  →  <H>  (para trainability / VQE)
  mode="state"   →  statevector  (para expressibility)

Uso básico
----------
import ansatzes as A

A.configure(n_qubits=4, L_layers=2, mode="expval")
A.H = mi_hamiltoniano          # opcional; si no, usa sum(Z_i)
circuits = A.get_circuits()    # lista de 19 QNodes
energy   = circuits[0](params) # devuelve <H>

A.configure(4, 2, mode="state")
circuits = A.get_circuits()
psi      = circuits[0](params) # devuelve statevector
"""

import pennylane as qml
from pennylane import numpy as np

# ─────────────────────────────────────────────
#  Estado global del módulo
# ─────────────────────────────────────────────
n_qubits: int = 4
L: int = 1
mode: str = "expval"          # "expval" | "state"
dev = None
H = None
_circuits: list | None = None


# ─────────────────────────────────────────────
#  Utilidades internas
# ─────────────────────────────────────────────

class ParamStream:
    """Consume parámetros de un array de forma secuencial."""
    def __init__(self, params):
        self.params = params
        self.i = 0

    def one(self):
        v = self.params[self.i]
        self.i += 1
        return v

    def take(self, n):
        v = self.params[self.i : self.i + n]
        self.i += n
        return v


def layer_rot(ps: ParamStream, wires, pattern: str = "RXRZ"):
    """Capa de rotaciones locales."""
    pattern = pattern.upper()
    for w in wires:
        if pattern == "RXRZ":
            qml.RX(ps.one(), wires=w)
            qml.RZ(ps.one(), wires=w)
        elif pattern == "RY":
            qml.RY(ps.one(), wires=w)
        elif pattern == "RYRZ":
            qml.RY(ps.one(), wires=w)
            qml.RZ(ps.one(), wires=w)
        else:
            raise ValueError(f"Patrón de rotación no soportado: {pattern!r}")


def apply_pairs(pairs, gate: str = "CNOT", ps: ParamStream | None = None):
    """Aplica puertas de dos qubits sobre una lista de pares."""
    gate = gate.upper()
    for a, b in pairs:
        if gate == "CNOT":
            qml.CNOT(wires=[a, b])
        elif gate == "CZ":
            qml.CZ(wires=[a, b])
        elif gate == "CRZ":
            qml.CRZ(ps.one() if ps else 0.0, wires=[a, b])
        elif gate == "CRX":
            qml.CRX(ps.one() if ps else 0.0, wires=[a, b])
        else:
            raise ValueError(f"Puerta de dos qubits no soportada: {gate!r}")


# ─────────────────────────────────────────────
#  Topologías de conectividad
# ─────────────────────────────────────────────

def pairs_chain(n):
    return [(n - i, n - i - 1) for i in range(1, n)]

def pairs_brick(n, offset=1):
    assert offset in (1, 2)
    return [(i, i - 1) for i in range(offset, n, 2)]

def pairs_all2all(n):
    return [(c, t) for c in range(n - 1, -1, -1)
            for t in range(n - 1, -1, -1) if c != t]

def pairs_ring(n):
    return [(i, (i + 1) % n) for i in range(n - 1, -1, -1)]

def pairs_ring_inverse(n):
    order = [n - 1] + list(range(n - 1))
    return [(i % n, (i - 1) % n) for i in order]


# ─────────────────────────────────────────────
#  Presupuesto de parámetros
# ─────────────────────────────────────────────

def per_layer_budget(n: int) -> int:
    """Número máximo de parámetros por layer (caso más costoso)."""
    return 4 * n + n * (n - 1)


# ─────────────────────────────────────────────
#  Medida final: expval vs state
# ─────────────────────────────────────────────

def _measurement():
    """Retorna la medida apropiada según el modo activo del módulo."""
    if mode == "state":
        return qml.state()
    return qml.expval(H)


# ─────────────────────────────────────────────
#  Configuración y construcción de QNodes
# ─────────────────────────────────────────────

def configure(
    n: int,
    L_layers: int,
    output_mode: str = "expval",
    device_name: str = "default.qubit",
):
    """
    Reconfigura el módulo y reconstruye los 19 QNodes.

    Parámetros
    ----------
    n           : número de qubits
    L_layers    : número de capas del ansatz
    output_mode : "expval" (valor esperado de H) o "state" (statevector)
    device_name : nombre del dispositivo PennyLane
    """
    global n_qubits, L, mode, dev, H, _circuits
    n_qubits = int(n)
    L = int(L_layers)
    mode = output_mode.lower()

    if mode not in ("expval", "state"):
        raise ValueError(f"output_mode debe ser 'expval' o 'state', no {mode!r}")

    dev = qml.device(device_name, wires=n_qubits)

    # Hamiltoniano por defecto: sum(Z_i)
    H = qml.Hamiltonian(
        [1.0] * n_qubits,
        [qml.PauliZ(i) for i in range(n_qubits)],
    )

    # ── Definición de los 19 ansatzes ──────────────────────────────────────

    @qml.qnode(dev, interface="autograd")
    def circuit1(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit2(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_chain(n_qubits), "CNOT")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit3(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_chain(n_qubits), "CRZ", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit4(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_chain(n_qubits), "CRX", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit5(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_all2all(n_qubits), "CRZ", ps)
            layer_rot(ps, range(n_qubits), "RXRZ")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit6(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_all2all(n_qubits), "CRX", ps)
            layer_rot(ps, range(n_qubits), "RXRZ")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit7(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CRZ", ps)
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 2), "CRZ", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit8(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CRX", ps)
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 2), "CRX", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit9(params):
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        ps = ParamStream(params)
        for _ in range(L):
            apply_pairs(pairs_chain(n_qubits), "CZ")
            for w in range(n_qubits):
                qml.RX(ps.one(), wires=w)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit10(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_chain(n_qubits), "CZ")
            qml.CZ(wires=[n_qubits - 1, 0])
            layer_rot(ps, range(n_qubits), "RY")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit11(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RYRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CNOT")
            if n_qubits > 2:
                for w in range(1, n_qubits - 1):
                    qml.RY(ps.one(), wires=w)
                    qml.RZ(ps.one(), wires=w)
            apply_pairs(pairs_brick(n_qubits, 2), "CNOT")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit12(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RYRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CZ")
            if n_qubits > 2:
                for w in range(1, n_qubits - 1):
                    qml.RY(ps.one(), wires=w)
                    qml.RZ(ps.one(), wires=w)
            apply_pairs(pairs_brick(n_qubits, 2), "CZ")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit13(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring(n_qubits), "CRZ", ps)
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring_inverse(n_qubits), "CRZ", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit14(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring(n_qubits), "CRX", ps)
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring_inverse(n_qubits), "CRX", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit15(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring(n_qubits), "CNOT")
            layer_rot(ps, range(n_qubits), "RY")
            apply_pairs(pairs_ring_inverse(n_qubits), "CNOT")
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit16(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CRZ", ps)
            apply_pairs(pairs_brick(n_qubits, 2), "CRZ", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit17(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_brick(n_qubits, 1), "CRX", ps)
            apply_pairs(pairs_brick(n_qubits, 2), "CRX", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit18(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_ring(n_qubits), "CRZ", ps)
        return _measurement()

    @qml.qnode(dev, interface="autograd")
    def circuit19(params):
        ps = ParamStream(params)
        for _ in range(L):
            layer_rot(ps, range(n_qubits), "RXRZ")
            apply_pairs(pairs_ring(n_qubits), "CRX", ps)
        return _measurement()

    _circuits = [
        circuit1,  circuit2,  circuit3,  circuit4,  circuit5,
        circuit6,  circuit7,  circuit8,  circuit9,  circuit10,
        circuit11, circuit12, circuit13, circuit14, circuit15,
        circuit16, circuit17, circuit18, circuit19,
    ]


def get_circuits() -> list:
    """Devuelve la lista de los 19 QNodes en orden (1-indexados por posición)."""
    if _circuits is None:
        raise RuntimeError("Llama primero a configure().")
    return list(_circuits)


# ── Inicialización por defecto ─────────────────────────────────────────────
configure(n_qubits, L, mode)
