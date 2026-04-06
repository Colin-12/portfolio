"""
03_Charts_Portfolio.py
======================
Génère des visuels professionnels pour le portfolio CRM.
Remplace le graphique matplotlib basique par des charts plus soignés.

Auteur : Armand K.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': '#f8fafc',
    'axes.facecolor': '#f8fafc',
    'axes.edgecolor': '#cbd5e1',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': '#cbd5e1',
})

INK = '#0f172a'
EMAIL = '#2563eb'
SMS = '#d97706'
CONTROL = '#6366f1'
POSITIVE = '#059669'
NEGATIVE = '#dc2626'
MUTED = '#94a3b8'

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
segments = ['VIP', 'Régulier', 'En Risque']
conv_email = [33.6, 12.6, 7.1]
conv_sms = [37.6, 12.1, 7.9]
conv_control = [42.4, 9.9, 9.7]

uplift_email = [-8.7, 2.6, -2.6]
uplift_sms = [-4.8, 2.2, -1.7]

pval_email = [0.212, 0.387, 0.501]
pval_sms = [0.493, 0.465, 0.658]

# ──────────────────────────────────────────────
# FIGURE 1: Taux de conversion par canal et segment
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# ── Panel A : Taux de conversion ──
ax = axes[0]
x = np.arange(len(segments))
w = 0.22

bars_email = ax.bar(x - w, conv_email, w, label='Email', color=EMAIL, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)
bars_sms = ax.bar(x, conv_sms, w, label='SMS', color=SMS, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)
bars_ctrl = ax.bar(x + w, conv_control, w, label='Témoin (Control)', color=CONTROL, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)

# Value labels
for bars in [bars_email, bars_sms, bars_ctrl]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.6,
                f'{h:.1f}%', ha='center', va='bottom',
                fontsize=8.5, fontweight='600', color=INK)

ax.set_xticks(x)
ax.set_xticklabels(segments, fontweight='600')
ax.set_ylabel('Taux de conversion (%)', fontweight='600', color=INK)
ax.set_title('A. Taux de conversion par canal et segment', fontweight='700', color=INK, pad=15, loc='left')
ax.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9, loc='upper right')
ax.set_ylim(0, 50)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Panel B : Uplift + P-values ──
ax2 = axes[1]

bars_u_email = ax2.bar(x - w/2 - 0.02, uplift_email, w, label='Email', color=EMAIL, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)
bars_u_sms = ax2.bar(x + w/2 + 0.02, uplift_sms, w, label='SMS', color=SMS, alpha=0.85, edgecolor='white', linewidth=0.8, zorder=3)

# Zero line
ax2.axhline(y=0, color=INK, linewidth=1.2, zorder=2)

# Hatch for non-significant (all of them here)
for bar in list(bars_u_email) + list(bars_u_sms):
    bar.set_hatch('///')
    bar.set_edgecolor('white')

# Value + P-value labels
for i, (ue, us, pe, ps) in enumerate(zip(uplift_email, uplift_sms, pval_email, pval_sms)):
    # Email
    yoff_e = -1.0 if ue < 0 else 0.3
    ax2.text(x[i] - w/2 - 0.02, ue + yoff_e,
             f'{ue:+.1f} pts\np={pe:.2f}', ha='center', va='top' if ue < 0 else 'bottom',
             fontsize=7.5, fontweight='600', color=NEGATIVE if ue < 0 else POSITIVE, linespacing=1.3)
    # SMS
    yoff_s = -1.0 if us < 0 else 0.3
    ax2.text(x[i] + w/2 + 0.02, us + yoff_s,
             f'{us:+.1f} pts\np={ps:.2f}', ha='center', va='top' if us < 0 else 'bottom',
             fontsize=7.5, fontweight='600', color=NEGATIVE if us < 0 else POSITIVE, linespacing=1.3)

ax2.set_xticks(x)
ax2.set_xticklabels(segments, fontweight='600')
ax2.set_ylabel('Uplift (points de %)', fontweight='600', color=INK)
ax2.set_title('B. Uplift vs Témoins (toutes p-values > 0.05)', fontweight='700', color=INK, pad=15, loc='left')

# Custom legend with hatch explanation
legend_email = mpatches.Patch(facecolor=EMAIL, alpha=0.85, label='Email', hatch='///', edgecolor='white')
legend_sms = mpatches.Patch(facecolor=SMS, alpha=0.85, label='SMS', hatch='///', edgecolor='white')
ax2.legend(handles=[legend_email, legend_sms], frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9, loc='lower right',
          title='Hachuré = non significatif', title_fontsize=8)

ax2.set_ylim(-13, 6)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Significance threshold annotation
ax2.annotate('⚠ Aucun résultat significatif (α = 0.05)',
            xy=(0.5, 0.97), xycoords='axes fraction',
            fontsize=9, color=NEGATIVE, fontweight='600',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee2e2', edgecolor='#fca5a5', alpha=0.9))

plt.tight_layout(pad=2.5)
plt.savefig('chart-crm.png', dpi=180, bbox_inches='tight', facecolor='#f8fafc')
plt.show()
print("✅ Graphique sauvegardé : chart-crm.png")
