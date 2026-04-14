
import numpy as np
import matplotlib.pyplot as plt
from plots.style import (
    DARK_BG, PANEL_BG, OUT_CLR, BACK_CLR, TURN_CLR,
    SPEED_CLR, POWER_CLR, VE_CLR, style_ax, LEG_PALETTE
)

def make_diagnostic_plot(analyzer):
    """
    Diagnostic plot using the ACTUAL legs from build_legs(),
    including adaptive trim and distance balancing.
    """
    df = analyzer.segment_data
    if df is None:
        df = analyzer.raw_data

    time   = df['time'].values
    speed  = df['speed'].values * 3.6
    power  = df['power'].values
    turns  = analyzer.turnaround_indices
    legs   = analyzer.legs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                   sharex=True, facecolor=PANEL_BG)
    style_ax(ax1)
    style_ax(ax2)

    # ── Grey background — full segment ────────────────────────────────────
    ax1.plot(time, speed, color='#444444', lw=0.8, zorder=1)
    ax2.plot(time, power, color='#444444', lw=0.8, zorder=1)

    # ── Compute actual leg time boundaries (flat lists, plain floats) ─────
    leg_starts = []
    leg_ends = []
    for leg in legs:
        leg_starts.append(float(leg['time'].iloc[0]))
        leg_ends.append(float(leg['time'].iloc[-1]))

    t_first = float(time[0])
    t_last = float(time[-1])

    # ── Shade crop zones (all gaps between actual legs) ───────────────────
    if len(leg_starts) > 0:
        # Gap before first leg
        if leg_starts[0] > t_first:
            for ax in (ax1, ax2):
                ax.axvspan(t_first, leg_starts[0],
                           color=TURN_CLR, alpha=0.12, zorder=3)

        # Gaps between consecutive legs (turnaround zones)
        for i in range(len(leg_starts) - 1):
            gap_s = leg_ends[i]
            gap_e = leg_starts[i + 1]
            if gap_e > gap_s:
                for ax in (ax1, ax2):
                    ax.axvspan(gap_s, gap_e, color=TURN_CLR, alpha=0.18, zorder=3)

                mask = (time >= gap_s) & (time <= gap_e)
                if mask.any():
                    spd_crop = df['speed'].values[mask]
                    dt_crop = np.diff(time[mask], prepend=time[mask][0])
                    d_crop = float(np.sum(spd_crop * dt_crop))
                    t_mid = (gap_s + gap_e) / 2
                    ax2.annotate(
                        f'crop\n{d_crop:.0f} m',
                        xy=(t_mid, float(np.nanmax(power)) * 0.88),
                        ha='center', va='top',
                        color=TURN_CLR, fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc=DARK_BG,
                                  ec=TURN_CLR, alpha=0.85, lw=0.8),
                        zorder=6,
                    )

        # Gap after last leg
        if leg_ends[-1] < t_last:
            for ax in (ax1, ax2):
                ax.axvspan(leg_ends[-1], t_last,
                           color=TURN_CLR, alpha=0.12, zorder=3)

    # ── Coloured legs — from actual analyzer.legs ─────────────────────────
    for k, leg in enumerate(legs):
        col   = LEG_PALETTE[k % len(LEG_PALETTE)]
        tag   = 'out' if k % 2 == 0 else 'back'
        t_leg = leg['time'].values
        spd_l = leg['speed'].values * 3.6
        pwr_l = leg['power'].values

        ax1.plot(t_leg, spd_l, color=col, lw=1.8, zorder=2,
                 label=f'Leg {k+1} ({tag})')
        ax2.plot(t_leg, pwr_l, color=col, lw=1.8, zorder=2)

        # Distance from v × dt
        dt_leg = np.diff(t_leg, prepend=t_leg[0])
        d_m    = float(np.sum(leg['speed'].values * dt_leg))

        # Pair partner distance (for comparison annotation)
        pair_idx = k + 1 if k % 2 == 0 else k - 1
        pair_txt = ""
        if 0 <= pair_idx < len(legs):
            t_p  = legs[pair_idx]['time'].values
            dt_p = np.diff(t_p, prepend=t_p[0])
            d_p  = float(np.sum(legs[pair_idx]['speed'].values * dt_p))
            diff = abs(d_m - d_p)
            pair_txt = f'\nΔ pair: {diff:.0f}m'

        t_mid = float(t_leg[len(t_leg) // 2])
        y_top = float(np.nanmax(speed)) * 0.92
        ax1.annotate(
            f'Leg {k+1}\n{d_m/1000:.3f} km{pair_txt}',
            xy=(t_mid, y_top),
            ha='center', va='top',
            color=col, fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.2', fc=DARK_BG, ec=col,
                      alpha=0.85, lw=0.8),
            zorder=5,
        )

    # ── Turnaround vertical lines ─────────────────────────────────────────
    for ti in turns:
        t_val = float(time[ti])
        ax1.axvline(t_val, color=TURN_CLR, lw=1.5, ls='--', zorder=4,
                    label='Turnaround' if ti == turns[0] else '')
        ax2.axvline(t_val, color=TURN_CLR, lw=1.5, ls='--', zorder=4)

    ax1.set_ylabel('Speed (km/h)', color='#aaaaaa')
    ax1.legend(fontsize=7.5, facecolor=PANEL_BG, labelcolor='white',
               loc='lower right', ncol=4)
    ax1.grid(True, alpha=0.12)

    ax2.set_ylabel('Power (W)', color='#aaaaaa')
    ax2.set_xlabel('Time (s)', color='#aaaaaa')
    ax2.grid(True, alpha=0.12)

    fig.suptitle(
        f'Diagnostic — {len(turns)} turnaround(s)  |  {len(legs)} leg(s)  '
        f'|  red zones = cropped out  |  distances from actual balanced legs',
        color='white', fontsize=10, fontweight='bold',
    )
    fig.tight_layout(pad=1.8)
    return fig