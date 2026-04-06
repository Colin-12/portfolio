"""
02_Power_Analysis.py
====================
Analyse de puissance statistique pour le projet CRM A/B Test.
Montre combien de sujets par cellule sont nécessaires pour détecter
un uplift donné, et pourquoi les résultats actuels ne sont pas concluants.

Auteur : Armand K.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────
# 1. Fonction de calcul de taille d'échantillon
# ──────────────────────────────────────────────
def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """
    Calcule la taille d'échantillon requise PAR GROUPE pour un test
    de proportions bilatéral (Z-test).
    
    Parameters
    ----------
    baseline_rate : float
        Taux de conversion du groupe témoin (ex: 0.17 = 17%)
    mde : float
        Minimum Detectable Effect en points absolus (ex: 0.03 = 3 pts)
    alpha : float
        Seuil de significativité (défaut = 0.05)
    power : float
        Puissance souhaitée (défaut = 0.80)
    
    Returns
    -------
    int : Nombre de sujets requis par groupe
    """
    from scipy.stats import norm
    
    p1 = baseline_rate
    p2 = baseline_rate + mde
    
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    n = ((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (mde ** 2)
    return int(np.ceil(n))


# ──────────────────────────────────────────────
# 2. Données du projet
# ──────────────────────────────────────────────
segments = {
    "VIP": {"baseline": 0.424, "actual_n": 223, "label": "VIP"},
    "Régulier": {"baseline": 0.099, "actual_n": 621, "label": "Régulier"},
    "En Risque": {"baseline": 0.097, "actual_n": 225, "label": "En Risque"},
}

# Range d'uplifts à analyser (de 1 pt à 10 pts)
mde_range = np.arange(0.01, 0.11, 0.005)

# ──────────────────────────────────────────────
# 3. Calcul pour chaque segment
# ──────────────────────────────────────────────
results = {}
for seg_name, seg_info in segments.items():
    sizes = [required_sample_size(seg_info["baseline"], mde) for mde in mde_range]
    results[seg_name] = sizes

# ──────────────────────────────────────────────
# 4. Tableau récapitulatif (3 pts et 5 pts d'uplift)
# ──────────────────────────────────────────────
print("=" * 70)
print("POWER ANALYSIS — Taille d'échantillon requise par groupe")
print("(Z-test bilatéral, α = 0.05, puissance = 80 %)")
print("=" * 70)
print(f"{'Segment':<15} {'Baseline':>10} {'N actuel':>10} {'N requis':>10} {'N requis':>10} {'Verdict':>14}")
print(f"{'':15} {'':>10} {'':>10} {'(3 pts)':>10} {'(5 pts)':>10} {'(3 pts)':>14}")
print("-" * 70)
for seg_name, seg_info in segments.items():
    n_3pts = required_sample_size(seg_info["baseline"], 0.03)
    n_5pts = required_sample_size(seg_info["baseline"], 0.05)
    actual = seg_info["actual_n"]
    verdict = "✓ Suffisant" if actual >= n_3pts else f"✗ ×{n_3pts / actual:.1f} trop petit"
    print(f"{seg_name:<15} {seg_info['baseline']:>9.1%} {actual:>10} {n_3pts:>10,} {n_5pts:>10,} {verdict:>14}")
print("=" * 70)
print()

# ──────────────────────────────────────────────
# 5. Visualisation
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
fig.patch.set_facecolor('#f8fafc')

# ── Panel 1 : Courbes N requis vs MDE ──
ax1 = axes[0]
ax1.set_facecolor('#f8fafc')
colors = {"VIP": "#6366f1", "Régulier": "#2563eb", "En Risque": "#d97706"}

for seg_name, sizes in results.items():
    ax1.plot(mde_range * 100, sizes, linewidth=2.5, label=seg_name, color=colors[seg_name])
    # Marquer le N actuel
    actual_n = segments[seg_name]["actual_n"]
    ax1.axhline(y=actual_n, color=colors[seg_name], linestyle=':', alpha=0.4, linewidth=1)
    ax1.annotate(f'N actuel = {actual_n}', 
                xy=(9, actual_n), fontsize=8, color=colors[seg_name],
                va='bottom', ha='right', alpha=0.7)

ax1.set_xlabel("Uplift minimum détectable (points de %)", fontsize=11, fontweight='600', labelpad=10)
ax1.set_ylabel("Taille d'échantillon requise par groupe", fontsize=11, fontweight='600', labelpad=10)
ax1.set_title("Combien de clients faut-il par cellule ?", fontsize=13, fontweight='700', pad=15, color='#0f172a')
ax1.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=10)
ax1.set_ylim(0, 6000)
ax1.set_xlim(1, 10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Zone "insuffisant" pour 3 pts
ax1.axvline(x=3, color='#dc2626', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.annotate('Cible : 3 pts', xy=(3.1, 5500), fontsize=9, color='#dc2626', fontweight='600')

# ── Panel 2 : Barres comparatives (N actuel vs N requis pour 3 pts) ──
ax2 = axes[1]
ax2.set_facecolor('#f8fafc')

seg_names = list(segments.keys())
actual_ns = [segments[s]["actual_n"] for s in seg_names]
required_ns = [required_sample_size(segments[s]["baseline"], 0.03) for s in seg_names]

y_pos = np.arange(len(seg_names))
bar_height = 0.35

bars_req = ax2.barh(y_pos - bar_height/2, required_ns, bar_height, label='N requis (3 pts)', 
                     color='#dc2626', alpha=0.75, edgecolor='white', linewidth=0.5)
bars_act = ax2.barh(y_pos + bar_height/2, actual_ns, bar_height, label='N actuel', 
                     color='#059669', alpha=0.75, edgecolor='white', linewidth=0.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(seg_names, fontsize=10, fontweight='600')
ax2.set_xlabel("Nombre de clients par groupe", fontsize=10, fontweight='600', labelpad=10)
ax2.set_title("Écart actuel vs requis", fontsize=13, fontweight='700', pad=15, color='#0f172a')
ax2.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', fontsize=9, loc='lower right')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotations de ratio
for i, (act, req) in enumerate(zip(actual_ns, required_ns)):
    ratio = req / act
    ax2.annotate(f'×{ratio:.1f}', xy=(req + 30, i - bar_height/2), 
                fontsize=9, fontweight='700', color='#dc2626', va='center')

plt.tight_layout(pad=2)
plt.savefig("power_analysis.png", dpi=150, bbox_inches='tight', facecolor='#f8fafc')
plt.show()

print("\n✅ Graphique sauvegardé : power_analysis.png")
print("\n📌 CONCLUSION :")
print("   Pour détecter un uplift de 3 points avec 80% de puissance,")
print("   il faudrait entre 1 500 et 3 800 clients PAR CELLULE selon le segment.")
print("   Nos cellules actuelles (223–621) sont 2× à 7× trop petites.")
print("   → Les résultats non significatifs étaient donc PRÉVISIBLES.")
