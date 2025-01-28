import numpy as np
import scipy.stats
import cudaq

# --- GHZ ---
class GHZ:
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count

    @property
    def kernel(self):
        n = self.qubit_count
        k = cudaq.make_kernel()
        qvector = k.qalloc(n)
        k.h(qvector[0])

        for i in range(n - 1):
            k.cx(qvector[i], qvector[i+1])

        k.mz(qvector)
        return k

    @property
    def kernel_params(self):
        return ()
# --- ---

class CounterfeitCoin:
    """
    Counterfeit coin finding problem.
    from https://github.com/pnnl/QASMBench
    """
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count

    @property
    def kernel(self):
        def _kernel(qubit_count: int):
            n = qubit_count
            qubits = cudaq.qvector(n)
        
            # Hadamard gate on all qubits
            for i in range(n-1):
                h(qubits[i])
                cx(qubits[i], qubits[n-1])

            r = mz(qubits[n-1])
            if r:
                # apply Hadamard to all but last qubit
                for i in range(n-1): h(qubits[i])
            else:
                x(qubits[n-1])
                h(qubits[n-1])

                cx(qubits[n // 2], qubits[n-1])

                # apply Hadamard to all but last qubit
                for i in range(n-1): h(qubits[i])

            for i in range(1, n):
                mz(qubits[i])

        return cudaq.kernel(_kernel)
    
    @property
    def kernel_params(self):
        return (self.qubit_count, )

class QuantumVolume:
    """
    A quantum volume model circuit.
    
    Simplified version of https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/quantum_volume.py
    """
    TMP_OP = 'tmp_op'

    def __init__(self, qubit_count: int, depth: int = None):
        self.qubit_count = qubit_count
        self.depth = depth or qubit_count # how many layers of SU(4)

    @property
    def kernel(self):
        """
        Construct the CUDA-Q kernel
        """

        qubit_count = self.qubit_count
        depth       = self.depth
        width = qubit_count // 2

        k = cudaq.make_kernel()
        qubits = k.qalloc(qubit_count)

        rng = np.random.default_rng()

        # For each layer, generate a permutation of qubits
        # Then generate and apply a Haar-random SU(4) to each pair
        unitaries = scipy.stats.unitary_group.rvs(4, depth*width, rng).reshape(depth, width, 16)

        c = 0
        self.gate_qubits = []
        for row in unitaries:
            perm = rng.permutation(qubit_count)
            
            for w, unitary in enumerate(row):
                opName = f'{self.TMP_OP}_{c}'

                # we register a custom operation
                cudaq.register_operation(opName, unitary)

                qubit = 2*w
                q0 = qubits[int(perm[qubit])]
                q1 = qubits[int(perm[qubit+1])]

                # Hack to apply our temporary operation on q0, q1
                getattr(k, opName)(q0, q1)
                c += 1

        k.mz(qubits)

        return k

    @property
    def kernel_params(self):
        return ()

class QFT:
    """
    Quantum Fourier Transform
    from https://nvidia.github.io/cuda-quantum/latest/examples/python/tutorials/quantum_fourier_transform.html
    """

    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.input_state = np.random.randint(0, 2, qubit_count).tolist()

    @property
    def kernel(self):
        def _kernel(input_state: list[int]):
            """
            Args:
            input_state (list[int]): specifies the input state to be Fourier transformed.
            """
            qubit_count = len(input_state)

            # Initialize qubits.
            qubits = cudaq.qvector(qubit_count)

            # Initialize the quantum circuit to the initial state.
            for i in range(qubit_count):
                if input_state[i] == 1:
                    x(qubits[i])

            # Apply Hadamard gates and controlled rotation gates.
            for i in range(qubit_count):
                h(qubits[i])
                for j in range(i + 1, qubit_count):
                    angle = (2 * np.pi) / (2**(j - i + 1))
                    cr1(angle, [qubits[j]], qubits[i])            

        return cudaq.kernel(_kernel)
    
    @property
    def kernel_params(self):
        return (self.input_state, )

class QAOA:
    """
    Dummy circuit for Quantum Approximate Optimization Algorithm (QAOA)

    from https://github.com/Infleqtion/client-superstaq/blob/main/supermarq-benchmarks/supermarq/benchmarks/qaoa_vanilla_proxy.py
    """
    def __init__(self, qubit_count: int, gamma: float = np.pi/3, beta: float = np.pi/6):
        self.qubit_count = qubit_count
        self.gamma = gamma
        self.beta  = beta
        
        # generate random array for in-kernel use in QAOA
        rs = np.random.choice([-1, 1], qubit_count*(qubit_count-1)//2)

        # a bug in "cudaq/kernel/ast_bridge.py" prevents using `ndarray[np.int]`
        # root cause is that `isinstance(val, np.int_(0)) == False`
        self.rs = rs.tolist()

    @property
    def kernel(self):
        def _kernel(qubit_count: int, rs: list[int], gamma: float, beta: float):
            """
            QAOA kernel
            - rs: list of {-1, 1} weight of terms in the Hamiltonian.
            - gamma, beta: parameters for the QAOA circuit
            """

            qvector = cudaq.qvector(qubit_count)

            for qubit in qvector:
                h(qubit)

            c = 0
            for i in range(qubit_count):
                for j in range(i+1, qubit_count):
                    phi = gamma * rs[c]
                    
                    # perform a ZZ interaction
                    x.ctrl(qvector[i], qvector[j]) #CNOT
                    rz(2*phi, qvector[j])
                    x.ctrl(qvector[i], qvector[j]) #CNOT

                    c += 1

            for q in qvector:
                rx(2*beta, q)

            mz(qvector)

        return cudaq.kernel(_kernel)

    @property
    def kernel_params(self):
        return self.qubit_count, self.rs, self.gamma, self.beta
