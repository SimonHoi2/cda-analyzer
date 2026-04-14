from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import G
from plots.style import (
    DARK_BG, PANEL_BG, OUT_CLR, BACK_CLR, TURN_CLR,
    SPEED_CLR, POWER_CLR, VE_CLR, style_ax, LEG_PALETTE, PAIR_COLOURS
)

def _clean_coord_series(s):
    """
    Fix coordinates stored as strings with extra dots, e.g. '52.182.095' → 52.182095.
    Also handles plain floats and ints — those pass through unchanged.
    """
    if s.dtype == object:
        def _fix(v):
            if pd.isna(v):
                return np.nan
            v = str(v)
            parts = v.split('.')
            if len(parts) > 2:
                # More than one dot: keep first as integer part, join rest as decimals
                v = parts[0] + '.' + ''.join(parts[1:])
            try:
                return float(v)
            except ValueError:
                return np.nan
        return s.apply(_fix)
    return pd.to_numeric(s, errors='coerce')

def _compute_ve_leg(leg, cda, crr, mass, rho, wind_ms):
    v     = leg['speed'].values
    P     = leg['power'].values
    t     = leg['time'].values
    acc   = np.gradient(v, t)
    dt    = np.diff(t, prepend=t[0])
    v_s   = np.maximum(v, 0.5)
    v_air = np.maximum(v + wind_ms, 0.1)
    P_net = P - 0.5*rho*cda*v_air**2*v - crr*mass*G*v - mass*v*acc
    dh    = P_net / (mass * G * v_s) * dt
    ve    = np.cumsum(dh)
    return ve - ve[0]


def _draw_gps_track(ax, analyzer, directions, ref_hdg):
    seg = analyzer.segment_data
    lat = seg['latitude'].values
    lon = seg['longitude'].values
    clr_map = {'out': OUT_CLR, 'back': BACK_CLR, 'turn': TURN_CLR}

    for k in range(len(lat) - 1):
        if np.isnan(lat[k]) or np.isnan(lat[k+1]):
            continue
        ax.plot([lon[k], lon[k+1]], [lat[k], lat[k+1]],
                color=clr_map.get(directions[k], 'gray'),
                lw=1.8, solid_capstyle='round')

    for ti in analyzer.turnaround_indices:
        if not np.isnan(lat[ti]):
            ax.plot(lon[ti], lat[ti], 'rx', ms=11, mew=2.5, zorder=5)
    ax.plot(lon[0], lat[0], 'wo', ms=9, zorder=6)

    for k, leg in enumerate(analyzer.legs):
        if 'latitude' not in leg.columns:
            continue
        mid = len(leg) // 2
        if mid + 2 >= len(leg):
            continue
        y0 = leg['latitude'].iloc[mid-1:mid+1].mean()
        x0 = leg['longitude'].iloc[mid-1:mid+1].mean()
        y1 = leg['latitude'].iloc[mid+1:mid+3].mean()
        x1 = leg['longitude'].iloc[mid+1:mid+3].mean()
        if np.isnan(y0) or np.isnan(y1):
            continue
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=OUT_CLR if k % 2 == 0 else BACK_CLR,
                                   lw=2.2, mutation_scale=16), zorder=7)

    handles = [Patch(color=OUT_CLR,  label="Outbound"),
               Patch(color=BACK_CLR, label="Return"),
               Patch(color=TURN_CLR, label="Turnaround")]
    ax.legend(handles=handles, fontsize=8,
              facecolor=PANEL_BG, labelcolor="white", loc="best")
    ax.set_aspect('equal', adjustable='datalim')

def make_raw_overview(analyzer):
    df  = analyzer.raw_data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), facecolor=PANEL_BG)

    style_ax(ax1)
    ax1.plot(df['time'], df['speed']*3.6, color=SPEED_CLR, lw=1)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Full Ride Overview", color="white", fontweight="bold")
    ax1.grid(True, alpha=0.15)

    style_ax(ax2)
    ax2.fill_between(df['time'], df['power'], alpha=0.25, color=POWER_CLR)
    ax2.plot(df['time'], df['power'], color=POWER_CLR, lw=0.8)
    ax2.set_ylabel("Power (W)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.15)

    fig.tight_layout(pad=2)
    return fig


def make_segment_overview(analyzer):
    df         = analyzer.raw_data
    seg        = analyzer.segment_data
    directions, ref_hdg = analyzer.compute_directions()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=PANEL_BG)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Speed
    style_ax(ax1)
    ax1.plot(df['time'], df['speed']*3.6, color='#444', lw=0.8)
    ax1.plot(seg['time'], seg['speed']*3.6, color=SPEED_CLR, lw=1.5,
             label="Selected segment")
    for ti in analyzer.turnaround_indices:
        ax1.axvline(analyzer.segment_data['time'].iloc[ti],
                    color=TURN_CLR, lw=1.8, alpha=0.85)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Speed + Turnarounds  (red = reversal)",
                  color="white", fontweight="bold", fontsize=9)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax1.grid(True, alpha=0.15)

    # GPS track
    style_ax(ax2)
    if ('latitude' in seg.columns and seg['latitude'].notna().any()
            and directions is not None):
        _draw_gps_track(ax2, analyzer, directions, ref_hdg)
        ax2.set_title(f"GPS Track  |  ref = {ref_hdg:.0f}°\n"
                      "green=out  orange=back  red=turn",
                      color="white", fontweight="bold", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No GPS data", ha="center", va="center",
                 color="gray", transform=ax2.transAxes)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.1)

    # Power
    style_ax(ax3)
    ax3.plot(df['time'], df['power'], color='#444', lw=0.8)
    ax3.fill_between(seg['time'], seg['power'], alpha=0.3, color=POWER_CLR)
    ax3.plot(seg['time'], seg['power'], color=POWER_CLR, lw=1.5)
    for ti in analyzer.turnaround_indices:
        ax3.axvline(analyzer.segment_data['time'].iloc[ti],
                    color=TURN_CLR, lw=1.8, alpha=0.85)
    ax3.set_ylabel("Power (W)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Power + Turnarounds",
                  color="white", fontweight="bold", fontsize=9)
    ax3.grid(True, alpha=0.15)

    # Heading vs time
    style_ax(ax4)
    if directions is not None:
        seg_df = analyzer.segment_data
        t_seg  = seg_df['time'].values
        h_seg  = seg_df['heading'].values
        cmap   = {'out': OUT_CLR, 'back': BACK_CLR, 'turn': TURN_CLR}
        for k in range(len(t_seg)-1):
            ax4.plot(t_seg[k:k+2], h_seg[k:k+2],
                     color=cmap.get(directions[k], 'gray'), lw=1.5)
        if ref_hdg is not None:
            ax4.axhline(ref_hdg, color=OUT_CLR, ls='--', lw=1.2,
                        label=f"Out ref {ref_hdg:.0f}°")
            ax4.axhline((ref_hdg+180) % 360, color=BACK_CLR, ls='--', lw=1.2,
                        label=f"Back ref {(ref_hdg+180)%360:.0f}°")
        ax4.set_ylim(0, 360)
        ax4.set_ylabel("Heading (°)")
        ax4.set_xlabel("Time (s)")
        ax4.set_title("Heading vs Time  (colour = direction)",
                      color="white", fontweight="bold", fontsize=9)
        ax4.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
        ax4.grid(True, alpha=0.15)
    else:
        ax4.text(0.5, 0.5, "No heading data", ha="center", va="center",
                 color="gray", transform=ax4.transAxes)

    fig.suptitle(
        f"Segment Overview  |  {len(analyzer.turnaround_indices)} turnaround(s)",
        color="white", fontsize=12, fontweight="bold")
    fig.tight_layout(pad=2)
    return fig

def make_results_plot(analyzer, crr, rho, mass, wind_ms):
    cda  = analyzer.cda_result
    legs = analyzer.legs
    directions, ref_hdg = analyzer.compute_directions()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=PANEL_BG)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ── Speed + power per leg ──────────────────────────────────────────────
    style_ax(ax1)
    ax1b = ax1.twinx()
    for sp in ax1b.spines.values():
        sp.set_color('#444')
    ax1b.tick_params(colors='#aaa', labelsize=8)

    d_off = 0.0
    for k, leg in enumerate(legs):
        c    = LEG_PALETTE[k % len(LEG_PALETTE)]
        tag  = "out" if k % 2 == 0 else "back"
        dist = leg['distance'].values - leg['distance'].values[0] + d_off
        ax1.plot(dist, leg['speed']*3.6, color=c, lw=1.6,
                 label=f"Leg {k+1} ({tag})")
        ax1b.plot(dist, leg['power'], color=c, lw=0.9, alpha=0.5)
        d_off += (leg['distance'].values[-1] - leg['distance'].values[0]) + 80

    ax1.set_ylabel("Speed (km/h)", color=SPEED_CLR)
    ax1b.set_ylabel("Power (W)", color=POWER_CLR)
    ax1.set_xlabel("Cumulative distance (m)")
    ax1.set_title(f"Speed & Power per Leg\nCdA = {cda:.4f} m²",
                  color="white", fontweight="bold", fontsize=9)
    ax1.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white")
    ax1.grid(True, alpha=0.15)

    # ── GPS track ──────────────────────────────────────────────────────────
    style_ax(ax2)
    seg = analyzer.segment_data
    if ('latitude' in seg.columns and seg['latitude'].notna().any()
            and directions is not None):
        _draw_gps_track(ax2, analyzer, directions, ref_hdg)
        ax2.set_title(f"GPS Track  |  ref = {ref_hdg:.0f}°\n"
                      "green=out  orange=back  arrows=direction",
                      color="white", fontweight="bold", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No GPS data", ha="center", va="center",
                 color="gray", transform=ax2.transAxes)
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.1)

    # ── VE per pair ────────────────────────────────────────────────────────
    style_ax(ax3)
    for k, (out_leg, back_leg) in enumerate(analyzer.pairs):
        c_out, c_back = PAIR_COLOURS[k % len(PAIR_COLOURS)]
        ve_out  = _compute_ve_leg(out_leg,  cda, crr, mass, rho, +wind_ms)
        ve_back = _compute_ve_leg(back_leg, cda, crr, mass, rho, -wind_ms)
        pct_out  = np.linspace(0, 100, len(ve_out))
        pct_back = np.linspace(0, 100, len(ve_back))
        closure = ve_out[-1] + ve_back[-1]

        ax3.plot(pct_out,  ve_out,  color=c_out,  lw=2.0,
                 label=f"Pair {k+1} out   Δ={ve_out[-1]:+.3f}m")
        ax3.plot(pct_back, ve_back, color=c_back, lw=2.0, ls='--',
                 label=f"Pair {k+1} back  Δ={ve_back[-1]:+.3f}m  "
                       f"closure={closure:+.3f}m")
        if abs(ve_out[-1] - ve_back[-1]) > 0.001:
            ax3.annotate("",
                xy=(100, ve_back[-1]), xytext=(100, ve_out[-1]),
                arrowprops=dict(arrowstyle='<->', color='white',
                               lw=1.2, shrinkA=0, shrinkB=0))

    ax3.axhline(0, color='white', ls=':', lw=0.8, alpha=0.5)
    ax3.set_xlabel("Leg progress (%)")
    ax3.set_ylabel("Virtual elevation (m)")
    ax3.set_title("VE per Pair – out (solid) vs back (dashed)\n"
                  "Correct CdA → both legs end at same level",
                  color="white", fontweight="bold", fontsize=9)
    ax3.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white", ncol=2)
    ax3.grid(True, alpha=0.15)

    # ── CdA sensitivity ────────────────────────────────────────────────────
    style_ax(ax4)
    for delta, c, lbl in [(-0.03, '#ef5350', f"CdA={cda-0.03:.3f} (lower)"),
                           (0.00,  VE_CLR,    f"CdA={cda:.3f} (optimal)"),
                           (+0.03, '#42a5f5', f"CdA={cda+0.03:.3f} (higher)")]:
        test, d_off, first = cda + delta, 0.0, True
        for k, leg in enumerate(legs):
            w    = +wind_ms if k % 2 == 0 else -wind_ms
            ve   = _compute_ve_leg(leg, test, crr, mass, rho, w)
            ll   = leg['distance'].values[-1] - leg['distance'].values[0]
            dist = leg['distance'].values - leg['distance'].values[0] + d_off
            ax4.plot(dist, ve, color=c, lw=1.5,
                     label=lbl if first else "", alpha=0.85)
            first = False
            d_off += ll + 80

    ax4.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)
    ax4.set_ylabel("Virtual elevation (m)")
    ax4.set_xlabel("Cumulative distance (m)")
    ax4.set_title("CdA Sensitivity  (±0.03 m²)\n"
                  "Optimal → all legs return closest to 0",
                  color="white", fontweight="bold", fontsize=9)
    ax4.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white")
    ax4.grid(True, alpha=0.15)

    wind_str = f"  |  wind = {wind_ms:+.1f} m/s" if wind_ms != 0 else ""
    fig.suptitle(
        f"CdA Analysis Results – CdA = {cda:.4f} m²"
        f"  |  {len(analyzer.pairs)} pair(s){wind_str}",
        color="white", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2)
    return fig