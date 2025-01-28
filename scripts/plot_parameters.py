import numpy as np
import matplotlib.pyplot as plt

n = np.arange(2, 35)

# --- Entanglement ---
plt.figure(figsize=(4.5, 3.5), layout='constrained')

plt.plot(n, n/(4*n-1), '+-', label='Counterfeit Coin')
plt.plot(n, 1-1/n, 'x-', label='GHZ')
plt.plot(n, (n+1)/(n+5), 'o-', mfc='none', label='QFT')
plt.plot(n, n/n, 'v-', mfc='none', label='Quantum Volume')
plt.plot(n, 2*(n-1)/(3*n+1), 'D-', mfc='none', label='QAOA')

plt.xlabel('qubits')
plt.ylabel('entanglement ratio ($N_{2q}/N_{tot}$)')
plt.ylim(0, 1.01)
plt.xlim(2, 34)
plt.legend(ncols=2, fontsize='small')
plt.xticks(n[::2])
plt.savefig('entanglement_ratio.pdf')


# --- Number of gate ---
plt.figure(figsize=(4.5, 3.5), layout='constrained')

plt.plot(n, 4*n-1, '+-', label='Counterfeit Coin')
plt.plot(n, n, 'x-', label='GHZ')
plt.plot(n, .5*n*(n+5), 'o-', mfc='none', label='QFT')
plt.plot(n, .5*n*n, 'v-', mfc='none', label='Quantum Volume')
plt.plot(n, .5*n*(2*n+1), 'D-', mfc='none', label='QAOA')

plt.xlabel('qubits')
plt.ylabel('number of gates ($N_{tot}$)')
plt.ylim(0, .5*max(n)*(2*max(n)+1)+0.1)
plt.xlim(2, 34)
plt.legend()
plt.xticks(n[::2])
plt.savefig('gate_number.pdf')


