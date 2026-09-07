"""
Plot a saved single-phaseshifter sweep.

Figure 1  optical power against dissipated electrical power.
          Phase is linear in dissipated power, so the fringe appears as a
          plain sinusoid here rather than the chirped one you get against
          current.

Figure 2  dissipated electrical power against measured current.
          The heater warms as it is driven, so its resistance rises and the
          power grows faster than I^2. Since the dissipation depends on the
          current only through I^2, the natural model has no odd terms:
              x = c2 I^2 + c4 I^4
          which is the quartic behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt

OUT = r'H:\QML_PIC\prakash_calib\repo\ch107_sweep'

results = np.load(OUT + '.npy')

I   = results[:, :, 1].mean(axis=1)
I_e = results[:, :, 1].std(axis=1)
X   = results[:, :, 3].mean(axis=1)
X_e = results[:, :, 3].std(axis=1)
P   = np.nanmean(results[:, :, 5], axis=1)
P_e = np.nanstd(results[:, :, 5], axis=1)

# drop points clipped at compliance, where the dissipated power stops rising
keep = np.concatenate([[True], np.diff(X) > 1e-3])
print(f'{keep.sum()} of {len(X)} points kept')
I, I_e, X, X_e, P, P_e = (a[keep] for a in (I, I_e, X, X_e, P, P_e))

# --------------------------- 1: interference fringe -------------------------
fig1, ax1 = plt.subplots(figsize=(7, 4.5), layout='constrained')
ax1.errorbar(X, P, yerr=P_e, xerr=X_e, fmt='o-', ms=3.5, lw=1, capsize=2)
ax1.set_xlabel('dissipated electrical power / mW')
ax1.set_ylabel('optical power / mW')
ax1.set_title('interference fringe, phaseshifter (0, 9)')
fig1.savefig(OUT + '_fringe.png', dpi=150)

# ------------------------ 2: power against current --------------------------
# fit x = c2 I^2 + c4 I^4, no odd terms and no constant
A = np.column_stack([I ** 2, I ** 4])
coef, *_ = np.linalg.lstsq(A, X, rcond=None)
c2, c4 = coef
I_fit = np.linspace(0, I.max(), 300)
X_fit = c2 * I_fit ** 2 + c4 * I_fit ** 4
print(f'x = {c2:.6g} I^2 + {c4:.6g} I^4   (I in mA, x in mW)')
print(f'cold resistance from c2: {c2 * 1e3:.1f} ohm')

fig2, ax2 = plt.subplots(figsize=(7, 4.5), layout='constrained')
ax2.errorbar(I, X, yerr=X_e, xerr=I_e, fmt='o', ms=3.5, capsize=2, label='measured')
ax2.plot(I_fit, X_fit, '-', lw=1, label=r'$c_2I^2 + c_4I^4$')
ax2.set_xlabel('current / mA')
ax2.set_ylabel('dissipated electrical power / mW')
ax2.set_title('phaseshifter power against drive current')
ax2.legend()
fig2.savefig(OUT + '_power.png', dpi=150)

plt.show()
