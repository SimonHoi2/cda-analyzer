#!/usr/bin/env python3
"""
Cycling CdA Analyzer - Web Version (Streamlit)
Chung Virtual Elevation Method
=====================================================
Testversie: gratis, geen betaling vereist.
Later uitbreidbaar met token-systeem / Stripe.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactieve backend voor web (geen Tk!)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import tempfile
import io
import os
from pathlib import Path
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

try:
    from fitparse import FitFile
    FITPARSE_AVAILABLE = True
except ImportError:
    FITPARSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colour scheme (identiek aan desktop versie)
# ---------------------------------------------------------------------------
DARK_BG   = '#1e1e1e'
PANEL_BG  = '#2b2b2b'
OUT_CLR   = '#00e676'
BACK_CLR  = '#ff9100'
TURN_CLR  = '#f44336'
SPEED_CLR = '#00bcd4'
POWER_CLR = '#ffa726'
VE_CLR    = '#69f0ae'

PAIR_COLOURS = [
    ('#00e676', '#69f0ae'),
    ('#ff9100', '#ffcc02'),
    ('#b39ddb', '#e040fb'),
    ('#80cbc4', '#00bcd4'),
    ('#ef9a9a', '#f44336'),
    ('#a5d6a7', '#388e3c'),
]

LEG_PALETTE = [OUT_CLR, BACK_CLR, '#b39ddb', '#80cbc4', '#ffcc80', '#ef9a9a']


# ===========================================================================
# Helper functions  (ongewijzigd t.o.v. desktop versie)
# ===========================================================================

def circular_mean(angles_deg):
    r = np.radians(np.asarray(angles_deg))
    return float(np.degrees(
        np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)

def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='#aaaaaa', labelsize=8)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_color('#444444')


# ===========================================================================
# Analysis engine  (100% ongewijzigd t.o.v. desktop versie)
# ===========================================================================

class CyclingDataAnalyzer:
    """Load FIT data, detect turnarounds, split legs, fit CdA."""

    G = 9.81

    def __init__(self):
        self.raw_data           = None
        self.segment_data       = None
        self.legs               = []
        self.pairs              = []
        self.turnaround_indices = []
        self.cda_result         = None
        self.virtual_elevation  = None
        self.leg_closures       = []
        self.gps_closures       = []

    def load_fit_file(self, filepath, speed_source: str = 'sensor'):
        if not FITPARSE_AVAILABLE:
            raise ImportError("Install fitparse: pip install fitparse")

        rows = []
        for rec in FitFile(filepath).get_messages('record'):
            rows.append({f.name: f.value for f in rec})
        if not rows:
            raise ValueError("No record data in FIT file.")

        df = pd.DataFrame(rows)
        df = df.rename(columns={
            'position_lat':      'latitude',
            'position_long':     'longitude',
            'enhanced_speed':    'speed',
            'enhanced_altitude': 'altitude',
        })

        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp field.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        if 'latitude' in df.columns and df['latitude'].notna().any():
            if df['latitude'].abs().max() > 180:
                df['latitude']  = df['latitude']  * (180 / 2 ** 31)
                df['longitude'] = df['longitude'] * (180 / 2 ** 31)

        if speed_source == 'gps':
            if 'latitude' in df.columns and df['latitude'].notna().any():
                df['speed'] = self._speed_from_gps(
                    df['latitude'], df['longitude'], df['timestamp'])
            else:
                raise ValueError("GPS speed selected but no GPS data found.")
            df['speed'] = (df['speed']
                           .rolling(5, center=True, min_periods=1)
                           .mean()
                           .clip(lower=0))
        else:
            if 'speed' not in df.columns or df['speed'].isna().all():
                if 'latitude' in df.columns:
                    df['speed'] = self._speed_from_gps(
                        df['latitude'], df['longitude'], df['timestamp'])
                else:
                    raise ValueError("No speed or GPS data.")
            else:
                if df['speed'].median() > 50:
                    df['speed'] /= 1000.0
            df['speed'] = df['speed'].ffill().bfill().clip(lower=0)

        if 'power' not in df.columns:
            raise ValueError("No power data — a power meter is required.")
        df['power'] = (pd.to_numeric(df['power'], errors='coerce')
                       .ffill().bfill().clip(lower=0))

        if 'altitude' not in df.columns:
            df['altitude'] = np.nan

        if 'distance' not in df.columns or df['distance'].isna().all():
            if 'latitude' in df.columns:
                df['distance'] = self._dist_from_gps(
                    df['latitude'], df['longitude'])
            else:
                df['distance'] = self._dist_from_speed(
                    df['speed'], df['timestamp'])

        if 'latitude' in df.columns and df['latitude'].notna().any():
            df['heading'] = self._heading_from_gps(
                df['latitude'].values, df['longitude'].values)
        else:
            df['heading'] = np.nan

        df = df.dropna(subset=['speed', 'power']).reset_index(drop=True)
        df['time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

        self.raw_data = df
        return df

    # ── GPS utilities ──────────────────────────────────────────────────────

    def _speed_from_gps(self, lat, lon, ts):
        spds = [0.0]
        for i in range(1, len(lat)):
            if any(pd.isna([lat.iloc[i], lon.iloc[i],
                            lat.iloc[i-1], lon.iloc[i-1]])):
                spds.append(spds[-1]); continue
            d  = self._haversine(lat.iloc[i-1], lon.iloc[i-1],
                                 lat.iloc[i],   lon.iloc[i])
            dt = (ts.iloc[i] - ts.iloc[i-1]).total_seconds()
            spds.append(min(d/dt, 25.0) if dt > 0 else spds[-1])
        return pd.Series(spds, index=lat.index)

    def _dist_from_gps(self, lat, lon):
        cum, d = 0.0, [0.0]
        for i in range(1, len(lat)):
            if any(pd.isna([lat.iloc[i], lon.iloc[i],
                            lat.iloc[i-1], lon.iloc[i-1]])):
                d.append(cum); continue
            cum += self._haversine(lat.iloc[i-1], lon.iloc[i-1],
                                   lat.iloc[i],   lon.iloc[i])
            d.append(cum)
        return pd.Series(d, index=lat.index)

    def _dist_from_speed(self, speed, ts):
        cum, d = 0.0, [0.0]
        for i in range(1, len(speed)):
            try:
                dt = (ts.iloc[i] - ts.iloc[i-1]).total_seconds()
            except Exception:
                dt = 1.0
            cum += speed.iloc[i] * dt
            d.append(cum)
        return pd.Series(d, index=speed.index)

    @staticmethod
    def _haversine(la1, lo1, la2, lo2):
        R = 6_371_000
        a = (np.sin(np.radians(la2-la1)/2)**2 +
             np.cos(np.radians(la1)) * np.cos(np.radians(la2)) *
             np.sin(np.radians(lo2-lo1)/2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    @staticmethod
    def _heading_from_gps(lat, lon):
        h = np.zeros(len(lat))
        for i in range(1, len(lat)):
            if np.isnan(lat[i]) or np.isnan(lon[i]):
                h[i] = h[i-1]; continue
            dlo = np.radians(lon[i] - lon[i-1])
            la1, la2 = np.radians(lat[i-1]), np.radians(lat[i])
            x = np.sin(dlo) * np.cos(la2)
            y = np.cos(la1)*np.sin(la2) - np.sin(la1)*np.cos(la2)*np.cos(dlo)
            h[i] = np.degrees(np.arctan2(x, y)) % 360
        h[0] = h[1] if len(h) > 1 else 0.0
        return h

    # ── Segment / leg management ───────────────────────────────────────────

    def set_segment(self, start_idx, end_idx):
        self.segment_data = (self.raw_data
                             .iloc[start_idx:end_idx+1]
                             .copy()
                             .reset_index(drop=True))
        self.legs               = []
        self.pairs              = []
        self.turnaround_indices = []
        self.cda_result         = None
        self.virtual_elevation  = None
        self.leg_closures       = []
        self.gps_closures       = []

    def detect_turnarounds(self, speed_threshold=1.5):
        if self.segment_data is None:
            return []

        df = self.segment_data

        if 'heading' in df.columns and df['heading'].notna().any():
            hdg   = pd.Series(df['heading'].values)
            hdg_s = hdg.rolling(15, center=True, min_periods=1).mean()
            turns, min_gap = [], 40

            for i in range(30, len(hdg_s) - 30):
                before = hdg_s.iloc[max(0, i-15):i].values
                after  = hdg_s.iloc[i:i+15].values
                if angle_diff(circular_mean(before), circular_mean(after)) > 120:
                    zone_s    = max(0, i-20)
                    zone_e    = min(len(df), i+20)
                    local_min = zone_s + int(
                        np.argmin(df['speed'].values[zone_s:zone_e]))
                    if not turns or local_min - turns[-1] > min_gap:
                        turns.append(local_min)

            if turns:
                self.turnaround_indices = turns
                return turns

        spd            = self.segment_data['speed'].values
        ss             = pd.Series(spd).rolling(5, center=True,
                                                min_periods=1).mean().values
        riding_speed   = np.median(spd[spd > 3.0]) if (spd > 3.0).any() else 5.0
        adap_thr       = max(speed_threshold, riding_speed * 0.40)
        turns          = []

        for i in range(20, len(ss) - 20):
            if ss[max(0, i-10):i].mean() > 4.0 and ss[i] < adap_thr:
                if ss[i] <= ss[max(0, i-5):i+5].min() + 0.3:
                    turns.append(i)

        clean = []
        for t in turns:
            if not clean or t - clean[-1] > 30:
                clean.append(t)

        self.turnaround_indices = clean
        return clean

    def build_legs(self, trim=10):
        if self.segment_data is None:
            return []

        turns  = self.turnaround_indices or self.detect_turnarounds()
        bounds = [0] + turns + [len(self.segment_data) - 1]
        legs   = []

        for k in range(len(bounds) - 1):
            s = bounds[k] + trim
            e = bounds[k + 1] - trim
            if e - s < 20:
                continue
            leg = self.segment_data.iloc[s:e+1].copy().reset_index(drop=True)
            if len(leg) < 20 or leg['speed'].mean() < 2.0:
                continue
            legs.append(leg)

        self.legs  = legs
        self.pairs = [(legs[k], legs[k+1]) for k in range(0, len(legs)-1, 2)]

        self.gps_closures = []
        for out_leg, back_leg in self.pairs:
            if ('latitude' not in out_leg.columns
                    or out_leg['latitude'].isna().all()):
                self.gps_closures.append(np.nan); continue
            lat1, lon1 = out_leg['latitude'].iloc[0],  out_leg['longitude'].iloc[0]
            lat2, lon2 = back_leg['latitude'].iloc[-1], back_leg['longitude'].iloc[-1]
            if any(pd.isna([lat1, lon1, lat2, lon2])):
                self.gps_closures.append(np.nan); continue
            self.gps_closures.append(self._haversine(lat1, lon1, lat2, lon2))

        return legs

    def compute_directions(self, ref_heading=None):
        df = self.segment_data
        if df is None or 'heading' not in df.columns:
            return None, None

        hdg = df['heading'].values
        if ref_heading is None:
            t0   = df['time'].values[0]
            mask = df['time'].values <= t0 + 10
            if mask.sum() == 0:
                mask[:5] = True
            ref_heading = circular_mean(hdg[mask])

        back_hdg   = (ref_heading + 180) % 360
        directions = np.full(len(df), 'turn', dtype=object)
        for i, h in enumerate(hdg):
            if angle_diff(h, ref_heading) <= 90:
                directions[i] = 'out'
            elif angle_diff(h, back_hdg) <= 90:
                directions[i] = 'back'

        return directions, ref_heading

    # ── CdA fitting ────────────────────────────────────────────────────────

    def _leg_delta(self, leg_df, cda, crr, mass, rho, wind_ms=0.0):
        v     = leg_df['speed'].values
        P     = leg_df['power'].values
        t     = leg_df['time'].values
        acc   = np.gradient(v, t)
        dt    = np.diff(t, prepend=t[0])
        v_s   = np.maximum(v, 0.5)
        v_air = np.maximum(v + wind_ms, 0.1)
        P_net = P - 0.5*rho*cda*v_air**2*v - crr*mass*self.G*v - mass*v*acc
        dh    = P_net / (mass * self.G * v_s) * dt
        ve    = np.cumsum(dh)
        return float(ve[-1] - ve[0])

    def _objective(self, cda, crr, mass, rho, wind_ms=0.0):
        if not self.pairs:
            return 1e6
        errors = [
            self._leg_delta(o, cda, crr, mass, rho, +wind_ms) +
            self._leg_delta(b, cda, crr, mass, rho, -wind_ms)
            for o, b in self.pairs
        ]
        return float(np.sqrt(np.mean(np.array(errors)**2)))

    def calculate_cda(self, mass, crr, rho, wind_ms=0.0):
        if self.segment_data is None:
            raise ValueError("No segment selected.")

        self.detect_turnarounds()
        legs = self.build_legs(trim=10)

        if len(legs) < 2:
            raise ValueError(
                f"Only {len(legs)} usable leg(s) found — "
                "include at least one complete out-and-back.")
        if not self.pairs:
            raise ValueError("Could not form any out-and-back pairs.")

        grid  = np.linspace(0.10, 0.70, 400)
        errs  = [self._objective(c, crr, mass, rho, wind_ms) for c in grid]
        best  = grid[np.argmin(errs)]

        fine    = np.linspace(max(0.05, best-0.05), min(0.80, best+0.05), 400)
        errs_f  = [self._objective(c, crr, mass, rho, wind_ms) for c in fine]
        opt_cda = fine[np.argmin(errs_f)]

        self.leg_closures = []
        for o, b in self.pairs:
            d_o = self._leg_delta(o, opt_cda, crr, mass, rho, +wind_ms)
            d_b = self._leg_delta(b, opt_cda, crr, mass, rho, -wind_ms)
            self.leg_closures.append(d_o + d_b)

        ve_parts, offset = [], 0.0
        for k, leg in enumerate(legs):
            w     = +wind_ms if k % 2 == 0 else -wind_ms
            v     = leg['speed'].values
            P     = leg['power'].values
            t     = leg['time'].values
            acc   = np.gradient(v, t)
            dt    = np.diff(t, prepend=t[0])
            v_s   = np.maximum(v, 0.5)
            v_air = np.maximum(v + w, 0.1)
            P_net = P - 0.5*rho*opt_cda*v_air**2*v - crr*mass*self.G*v - mass*v*acc
            dh    = P_net / (mass * self.G * v_s) * dt
            ve    = np.cumsum(dh) + offset
            ve_parts.append(ve)
            offset = float(ve[-1])

        self.virtual_elevation = np.concatenate(ve_parts)
        self.cda_result        = opt_cda
        return opt_cda


# ===========================================================================
# Plotting helpers  (web-adapted: figuren worden teruggegeven als object,
#                   niet getekend op een canvas)
# ===========================================================================

def _compute_ve_leg(leg, cda, crr, mass, rho, wind_ms):
    v     = leg['speed'].values
    P     = leg['power'].values
    t     = leg['time'].values
    acc   = np.gradient(v, t)
    dt    = np.diff(t, prepend=t[0])
    v_s   = np.maximum(v, 0.5)
    v_air = np.maximum(v + wind_ms, 0.1)
    P_net = P - 0.5*rho*cda*v_air**2*v - crr*mass*9.81*v - mass*v*acc
    dh    = P_net / (mass * 9.81 * v_s) * dt
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


# ===========================================================================
# Streamlit UI
# ===========================================================================

def _background_cda_closures(analyzer, mass, crr, rho, wind_ms=0.0):
    """
    Runs a silent CdA optimisation and returns only the VE closures.
    Does NOT expose the CdA value — that stays behind the payment gate.
    Returns list of closure values in metres, or None on failure.
    """
    import copy
    try:
        probe = copy.copy(analyzer)           # shallow copy — shares raw_data
        probe.segment_data       = analyzer.segment_data
        probe.turnaround_indices = list(analyzer.turnaround_indices)
        probe.legs               = []
        probe.pairs              = []
        probe.leg_closures       = []
        probe.gps_closures       = []
        probe.cda_result         = None
        probe.virtual_elevation  = None

        probe.build_legs(trim=10)
        if not probe.pairs:
            return None

        # Run the same two-pass grid search as calculate_cda()
        grid = np.linspace(0.10, 0.70, 200)   # coarser for speed
        errs = [probe._objective(c, crr, mass, rho, wind_ms) for c in grid]
        best = grid[np.argmin(errs)]
        fine = np.linspace(max(0.05, best - 0.05), min(0.80, best + 0.05), 200)
        errs_f = [probe._objective(c, crr, mass, rho, wind_ms) for c in fine]
        opt_cda = fine[np.argmin(errs_f)]

        closures = []
        for o, b in probe.pairs:
            d_o = probe._leg_delta(o, opt_cda, crr, mass, rho, +wind_ms)
            d_b = probe._leg_delta(b, opt_cda, crr, mass, rho, -wind_ms)
            closures.append(d_o + d_b)
        return closures
    except Exception:
        return None


def estimate_tt_time(cda, mass, crr, rho, power_w, distance_m=40000):
    """
    Solve for constant speed on flat ground given power and CdA.
    Returns time in seconds.
    """
    from numpy.polynomial import polynomial as P
    g = 9.81
    # Rearranged: 0.5*rho*cda*v^3 + crr*mass*g*v - power = 0
    # numpy roots for: a*v^3 + 0*v^2 + b*v + c = 0
    a = 0.5 * rho * cda
    b = crr * mass * g
    c = -power_w
    coeffs = [a, 0, b, c]  # descending order
    roots = np.roots(coeffs)
    # Take the single real positive root
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not real_roots:
        return None
    v = max(real_roots)
    return distance_m / v

def compare_cda(cda_a, cda_b, mass, crr, rho, power_w, distance_m=40000):
    """
    Returns dict with TT times and delta for both CdA values.
    """
    t_a = estimate_tt_time(cda_a, mass, crr, rho, power_w, distance_m)
    t_b = estimate_tt_time(cda_b, mass, crr, rho, power_w, distance_m)
    if t_a is None or t_b is None:
        return None
    delta_s  = t_b - t_a   # positive = B is slower
    v_a      = distance_m / t_a
    v_b      = distance_m / t_b
    return {
        'time_a':    t_a,
        'time_b':    t_b,
        'delta_s':   delta_s,
        'speed_a':   v_a * 3.6,
        'speed_b':   v_b * 3.6,
    }

def check_data_quality(analyzer, mass=75.0, crr=0.0031, rho=1.225, wind_ms=0.0):
    """
    Analyzes the loaded FIT data and returns a quality verdict
    before allowing a paid CdA analysis.

    Returns a dict with per-check results and an overall verdict.
    """
    df = analyzer.raw_data
    checks = {}

    # ── 1. Duration ──────────────────────────────────────────────────────────
    duration_s = float(df['time'].iloc[-1])
    checks['duration'] = {
        'label': f'Ride duration: {duration_s / 60:.1f} min',
        'pass': duration_s >= 300,  # ≥ 5 minutes
        'warning': 180 <= duration_s < 300,  # 3-5 min = borderline
        'detail': 'Minimum 5 minutes recommended for a valid CdA test.',
    }

    # ── 2. Power meter presence ───────────────────────────────────────────────
    power_valid_pct = (df['power'].notna() & (df['power'] > 0)).sum() / len(df) * 100
    checks['power_quality'] = {
        'label': f'Valid power data: {power_valid_pct:.1f}%',
        'pass': power_valid_pct >= 90,
        'warning': 70 <= power_valid_pct < 90,
        'detail': 'Power meter required with >90% valid (non-zero) data.',
    }

    # ── 3. Average speed (only moving segments) ───────────────────────────────
    moving = df[df['speed'] > 2.0]
    avg_spd_kmh = moving['speed'].mean() * 3.6 if len(moving) > 0 else 0
    checks['avg_speed'] = {
        'label': f'Avg moving speed: {avg_spd_kmh:.1f} km/h',
        'pass': avg_spd_kmh >= 25.0,
        'warning': 20.0 <= avg_spd_kmh < 25.0,
        'detail': 'CdA test requires sustained speed ≥ 25 km/h. Below 25 km/h, rolling resistance noise masks aerodynamic drag.',
    }

    # ── 4. Speed consistency (Coefficient of Variation) ───────────────────────
    if len(moving) > 10:
        speed_cv = moving['speed'].std() / moving['speed'].mean()
    else:
        speed_cv = 1.0
    checks['speed_consistency'] = {
        'label': f'Speed consistency (CV): {speed_cv:.3f}',
        'pass': speed_cv <= 0.30,
        'warning': 0.30 < speed_cv <= 0.50,
        'detail': 'CV < 0.30 = steady pacing. High CV = braking/surging = unreliable CdA.',
    }

    # ── 5. GPS data present ───────────────────────────────────────────────────
    has_gps = ('latitude' in df.columns and df['latitude'].notna().any())
    checks['gps'] = {
        'label': 'GPS data: ' + ('✓ Available' if has_gps else '✗ Not found'),
        'pass': has_gps,
        'warning': False,
        'detail': 'GPS required for turnaround detection and route validation.',
    }

    # ── 6. Turnaround detection ───────────────────────────────────────────────
    analyzer.set_segment(0, len(df) - 1)
    turns = analyzer.detect_turnarounds()
    n_turns = len(turns)

    if n_turns >= 1:
        analyzer.build_legs(trim=10)  # ← add this line

    checks['turnarounds'] = {
        'label': f'Turnarounds detected: {n_turns}',
        'pass': n_turns >= 1,
        'warning': False,
        'detail': 'At least 1 turnaround required for out-and-back (Chung) method.',
    }

    # ── 6b. VE loop closures (background CdA run — no CdA value shown) ────────
    if n_turns >= 1:
        bg_closures = _background_cda_closures(
            analyzer, mass=mass, crr=crr, rho=rho, wind_ms=wind_ms)
        if bg_closures:
            max_closure = max(abs(c) for c in bg_closures)
            closure_strs = ', '.join(f'{c:+.3f}m' for c in bg_closures)
            checks['ve_closure'] = {
                'label': f'VE loop closures: {closure_strs}',
                'pass': max_closure < 0.05,
                'warning': 0.05 <= max_closure < 0.10,
                'detail': (
                    f'Max closure = {max_closure:.3f} m. '
                    'Values < 0.05 m = excellent. 0.05–0.10 m = acceptable. '
                    '>0.10 m = poor conditions, re-test recommended.'
                ),
            }
        else:
            checks['ve_closure'] = {
                'label': 'VE closure check: could not run (not enough legs)',
                'pass': False,
                'warning': True,
                'detail': 'Background CdA optimisation failed — verify turnaround detection.',
            }
    else:
        checks['ve_closure'] = {
            'label': 'VE closure check: skipped (no turnarounds)',
            'pass': False,
            'warning': False,  # this is a real FAIL — data has no pairs at all
            'detail': 'Cannot evaluate VE closures without at least 1 out-and-back pair.',
        }


    # ── 7. Power steadiness ───────────────────────────────────────────────────
    avg_power_cv = df['power'].std() / (df['power'].mean() + 1e-6)
    checks['power_steadiness'] = {
        'label': f'Power steadiness (CV): {avg_power_cv:.3f}',
        'pass': avg_power_cv <= 0.60,
        'warning': 0.60 < avg_power_cv <= 0.90,
        'detail': 'Very erratic power (CV > 0.90) reduces CdA accuracy.',
    }

    # ── 8. Heading anti-parallel (out vs back ≈ 180°) ────────────────────────
    # Only meaningful if GPS + at least 1 turnaround was found
    if has_gps and n_turns >= 1 and 'heading' in df.columns:
        legs_temp = analyzer.legs  # already built by set_segment + detect above
        if len(legs_temp) >= 2:
            pair_heading_diffs = []
            for k in range(0, len(legs_temp) - 1, 2):
                out_hdg = circular_mean(legs_temp[k]['heading'].dropna().values)
                back_hdg = circular_mean(legs_temp[k + 1]['heading'].dropna().values)
                pair_heading_diffs.append(angle_diff(out_hdg, back_hdg))
            worst_diff = max(pair_heading_diffs) if pair_heading_diffs else 0
            anti_ok = worst_diff >= 135  # should be near 180°
            checks['heading_antiparallel'] = {
                'label': f'Out/back heading diff: {worst_diff:.0f}°  (need ≥135°)',
                'pass': anti_ok,
                'warning': not anti_ok and worst_diff >= 100,
                'detail': 'Out and back legs must go in opposite directions. Figure-8 or lollipop routes will produce unreliable CdA.',
            }
        else:
            checks['heading_antiparallel'] = {
                'label': 'Heading check: not enough legs to evaluate',
                'pass': False,
                'warning': True,
                'detail': 'Could not verify out/back heading — need at least 2 legs.',
            }
    else:
        checks['heading_antiparallel'] = {
            'label': 'Heading check: skipped (no GPS or no turns)',
            'pass': False,
            'warning': True,
            'detail': 'GPS heading required to verify out-and-back geometry.',
        }

    # ── 9. Pair count ≥ 3 (statistical reliability) ──────────────────────────
    n_pairs = len(analyzer.pairs)
    checks['pair_count'] = {
        'label': f'Out-and-back pairs: {n_pairs}  (recommended ≥ 3)',
        'pass': n_pairs >= 3,
        'warning': n_pairs == 2,
        'detail': 'More pairs = more reliable CdA average. 1 pair is very sensitive to a single gust or power spike.',
    }

    # ── 10. Route flatness (altitude balance per pair) ────────────────────────
    if has_gps and 'altitude' in df.columns and df['altitude'].notna().any():
        alt_drifts = []
        for out_leg, back_leg in analyzer.pairs:
            if 'altitude' not in out_leg.columns:
                continue
            alt_start = out_leg['altitude'].iloc[0]
            alt_end = back_leg['altitude'].iloc[-1]
            if not pd.isna(alt_start) and not pd.isna(alt_end):
                alt_drifts.append(abs(alt_end - alt_start))
        if alt_drifts:
            max_drift = max(alt_drifts)
            checks['route_flatness'] = {
                'label': f'Max altitude drift per pair: {max_drift:.1f} m  (need < 10 m)',
                'pass': max_drift < 10.0,
                'warning': 10.0 <= max_drift < 20.0,
                'detail': 'Start and end of each out-and-back pair must be at the same altitude. Large drift = route is not flat = VE closure error is structural.',
            }
        else:
            checks['route_flatness'] = {
                'label': 'Route flatness: altitude data missing in legs',
                'pass': False,
                'warning': True,
                'detail': 'Could not evaluate altitude balance — check altitude data.',
            }
    else:
        checks['route_flatness'] = {
            'label': 'Route flatness: no altitude data available',
            'pass': False,
            'warning': True,
            'detail': 'GPS altitude required to verify route flatness.',
        }



    # ── Overall score ─────────────────────────────────────────────────────────
    # ── Hard blockers — GPS and turnarounds must always be FAIL, never WARNING ──
    HARD_BLOCKERS = ['gps', 'turnarounds']
    has_hard_fail = any(
        not checks[k]['pass'] for k in HARD_BLOCKERS if k in checks
    )

    # ── Overall score (fixed denominator: always 8 checks) ───────────────────
    FIXED_TOTAL = 11  # keep constant so scores are comparable across files
    n_pass = sum(1 for c in checks.values() if c['pass'])
    n_warn = sum(1 for c in checks.values() if not c['pass'] and c.get('warning'))
    n_fail = sum(1 for c in checks.values() if not c['pass'] and not c.get('warning'))

    score = (n_pass * 2 + n_warn * 1) / (FIXED_TOTAL * 2) * 100

    if has_hard_fail:
        overall = 'FAIL'
    elif n_fail == 0:
        overall = 'PASS'
    elif n_fail == 1:
        overall = 'WARNING'
    else:
        overall = 'FAIL'

    return {
        'checks': checks,
        'n_pass': n_pass,
        'n_warn': n_warn,
        'n_fail': n_fail,
        'score': score,
        'overall': overall,
        'turns': turns,
    }

def _pdf_safe(txt: str) -> str:
    """Maak tekst veilig voor PyFPDF pages[n].encode('latin1')."""
    if txt is None:
        return ""
    s = str(txt)

    # vervang probleemtekens (o.a. jouw U+2013)
    s = (s.replace("\u2013", "-")   # en-dash –
           .replace("\u2014", "-")   # em-dash —
           .replace("\u2212", "-")   # minus sign −
           .replace("\u2192", "->")  # arrow →
           .replace("\u2026", "...") # …
           .replace("\u00A0", " ")   # nbsp
           .replace("\u202F", " ")   # narrow nbsp
           .replace("\u2018", "'").replace("\u2019", "'")  # ‘ ’
           .replace("\u201C", '"').replace("\u201D", '"')) # “ ”

    # forceer latin-1 encodability (rest => ?)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _cell(pdf, w, h, txt="", *args, **kwargs):
    return pdf.cell(w, h, _pdf_safe(txt), *args, **kwargs)

def _multi_cell(pdf, w, h, txt="", *args, **kwargs):
    return pdf.multi_cell(w, h, _pdf_safe(txt), *args, **kwargs)

def _pdf_safe(txt: str) -> str:
    if txt is None:
        return ""
    s = str(txt)
    s = (s.replace("\u2013", "-")   # –
           .replace("\u2014", "-")   # —
           .replace("\u2212", "-")   # −
           .replace("\u2192", "->")  # →
           .replace("\u2026", "...") # …
           .replace("\u00A0", " ")
           .replace("\u202F", " ")
           .replace("\u2018", "'").replace("\u2019", "'")
           .replace("\u201C", '"').replace("\u201D", '"'))
    return s.encode("latin-1", errors="replace").decode("latin-1")

def generate_pdf_report(analyzer, quality_result, cda_params=None, comparison=None):
    """
    Generates a complete PDF report containing:
    - File / ride summary
    - Quality control results
    - CdA analysis results (if performed)
    - Embedded plots
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    import os

    # Kies een Unicode font (TTF) van Windows
    font_regular = r"C:\Windows\Fonts\segoeui.ttf"
    font_bold = r"C:\Windows\Fonts\segoeuib.ttf"

    if os.path.exists(font_regular) and os.path.exists(font_bold):
        pdf.add_font("UI", "", font_regular, uni=True)
        pdf.add_font("UI", "B", font_bold, uni=True)
        FONT_FAMILY = "UI"
    else:
        # Fallback: core font (dan moet je alsnog sanitize gebruiken)
        FONT_FAMILY = "Helvetica"

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'B', 18)
    _cell(pdf,0, 10, 'Cycling CdA Analyzer – Analysis Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 9)
    _cell(pdf,0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(6)

    # ── Ride summary ──────────────────────────────────────────────────────────
    df = analyzer.raw_data
    pdf.set_font(FONT_FAMILY, 'B', 18)
    _cell(pdf,0, 8, '1. Ride Summary', ln=True)
    pdf.set_font(FONT_FAMILY, '', 10)
    _cell(pdf,60, 7, f'Total records: {len(df):,}', ln=True)
    _cell(pdf,60, 7, f'Duration: {df["time"].iloc[-1]/60:.1f} min', ln=True)
    _cell(pdf,60, 7, f'Avg speed: {df["speed"].mean()*3.6:.1f} km/h', ln=True)
    _cell(pdf,60, 7, f'Avg power: {df["power"].mean():.0f} W', ln=True)
    pdf.ln(4)

    # ── Quality control ───────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'B', 13)
    _cell(pdf,0, 8, '2. Data Quality Control', ln=True)
    pdf.ln(2)

    overall = quality_result['overall']
    score   = quality_result['score']
    colour  = (0, 180, 0) if overall == 'PASS' else (255, 165, 0) if overall == 'WARNING' else (220, 0, 0)

    # Overall badge
    pdf.set_fill_color(*colour)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    _cell(pdf,60, 9, f'  Overall: {overall}  (Score: {score:.0f}%)', fill=True, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Per-check table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(230, 230, 230)
    _cell(pdf,65, 7, 'Check', border=1, fill=True)
    _cell(pdf,22, 7, 'Status', border=1, fill=True, align='C')
    _cell(pdf,0,  7, 'Detail', border=1, fill=True, ln=True)

    pdf.set_font('Helvetica', '', 9)
    for key, chk in quality_result['checks'].items():
        if chk['pass']:
            status_txt = 'PASS'
            pdf.set_fill_color(220, 255, 220)
        elif chk.get('warning'):
            status_txt = 'WARN'
            pdf.set_fill_color(255, 245, 200)
        else:
            status_txt = 'FAIL'
            pdf.set_fill_color(255, 220, 220)
        _cell(pdf,65, 6, chk['label'][:40], border=1, fill=True)
        _cell(pdf,22, 6, status_txt, border=1, fill=True, align='C')
        _cell(pdf,0,  6, chk['detail'][:65], border=1, fill=True, ln=True)
    pdf.set_fill_color(255, 255, 255)
    pdf.ln(6)

    # ── CdA results ───────────────────────────────────────────────────────────
    if cda_params and analyzer.cda_result is not None:
        cda = analyzer.cda_result
        p   = cda_params

        pdf.set_font('Helvetica', 'B', 13)
        _cell(pdf,0, 8, '3. CdA Analysis Results', ln=True)
        pdf.ln(2)

        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(30, 100, 200)
        _cell(pdf,0, 12, f'CdA = {cda:.4f} m²', ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        pdf.set_font('Helvetica', '', 10)
        params_table = [
            ('Mass',        f'{p["mass"]:.1f} kg'),
            ('Crr',         f'{p["crr"]:.5f}'),
            ('Air density', f'{p["rho"]:.3f} kg/m³'),
            ('Wind',        f'{p["wind_ms"]:+.1f} m/s'),
            ('Pairs used',  str(len(analyzer.pairs))),
        ]
        for label, value in params_table:
            _cell(pdf,60, 7, label, border='LTB')
            _cell(pdf,0,  7, value, border='RTB', ln=True)
        pdf.ln(4)

        # ── Comparison section ────────────────────────────────────────────────────
        if comparison:
            bl = comparison['baseline']
            res = comparison['result']
            pw = comparison['ref_power']
            delta_s = res['delta_s']

            pdf.ln(6)
            pdf.set_font('Helvetica', 'B', 13)
            _cell(pdf, 0, 8, '4. Comparison vs Baseline', ln=True)
            pdf.ln(2)

            pdf.set_font('Helvetica', '', 10)
            _cell(pdf, 60, 7, 'Baseline CdA', border='LTB')
            _cell(pdf, 0, 7, f'{bl["cda"]:.4f} m2', border='RTB', ln=True)
            _cell(pdf, 60, 7, 'Test CdA', border='LB')
            _cell(pdf, 0, 7, f'{comparison["test_cda"]:.4f} m2', border='RB', ln=True)
            _cell(pdf, 60, 7, 'Delta CdA', border='LB')
            _cell(pdf, 0, 7, f'{(comparison["test_cda"] - bl["cda"]) * 1000:+.1f} cm2', border='RB', ln=True)
            pdf.ln(4)

            pdf.set_font('Helvetica', 'B', 11)
            _cell(pdf, 0, 7, f'40km TT estimate at {pw}W:', ln=True)
            pdf.set_font('Helvetica', '', 10)
            t_a, t_b = res['time_a'], res['time_b']
            _cell(pdf, 60, 7, 'Baseline time', border='LTB')
            _cell(pdf, 0, 7, f'{int(t_a // 60)}:{int(t_a % 60):02d}  ({res["speed_a"]:.1f} km/h)', border='RTB',
                  ln=True)
            _cell(pdf, 60, 7, 'Test time', border='LB')
            _cell(pdf, 0, 7, f'{int(t_b // 60)}:{int(t_b % 60):02d}  ({res["speed_b"]:.1f} km/h)', border='RB', ln=True)

            verdict = 'FASTER' if delta_s > 0 else ('SLOWER' if delta_s < 0 else 'SAME')
            _cell(pdf, 60, 7, 'Verdict', border='LB')
            _cell(pdf, 0, 7, f'{verdict} by {abs(delta_s):.1f}s', border='RB', ln=True)

        # VE closures
        pdf.set_font('Helvetica', 'B', 11)
        _cell(pdf,0, 7, 'VE Loop Closures (lower = better):', ln=True)
        pdf.set_font('Helvetica', '', 10)
        for k, lc in enumerate(analyzer.leg_closures):
            if   abs(lc) < 0.01: verdict = 'EXCELLENT'
            elif abs(lc) < 0.05: verdict = 'ACCEPTABLE'
            else:                verdict = 'POOR – recheck conditions'
            _cell(pdf,0, 6, f'  Pair {k+1}: {lc:+.4f} m  →  {verdict}', ln=True)

        # GPS closures
        if any(not np.isnan(gc) for gc in analyzer.gps_closures):
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 11)
            _cell(pdf,0, 7, 'GPS Route Closures (distance back→start):', ln=True)
            pdf.set_font('Helvetica', '', 10)
            for k, gc in enumerate(analyzer.gps_closures):
                if np.isnan(gc):
                    _cell(pdf,0, 6, f'  Pair {k+1}: no GPS', ln=True)
                else:
                    verdict = 'OK' if gc < 20 else ('WARN' if gc < 50 else 'LARGE')
                    _cell(pdf,0, 6, f'  Pair {k+1}: {gc:.1f} m  →  {verdict}', ln=True)
        pdf.ln(4)

        # Embed the results plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig_res = make_results_plot(analyzer, p['crr'], p['rho'], p['mass'], p['wind_ms'])
            fig_res.savefig(tmp.name, format='png', dpi=130,
                            bbox_inches='tight', facecolor=PANEL_BG)
            plt.close(fig_res)
            tmp_path = tmp.name

        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 13)
        _cell(pdf,0, 8, '4. Analysis Plots', ln=True)
        pdf.ln(2)
        pdf.image(tmp_path, w=185)
        os.unlink(tmp_path)

    else:
        pdf.set_font('Helvetica', 'I', 10)
        _cell(pdf,0, 8, '3. CdA Analysis: not performed yet.', ln=True)

    return pdf.output(dest="S").encode("latin-1")
# ===========================================================================
# TT estimation helpers  (keep these above main)
# ===========================================================================

def estimate_tt_time(cda, mass, crr, rho, power_w, distance_m=40000):
    """Solve for constant speed on flat ground. Returns time in seconds."""
    a = 0.5 * rho * cda
    b = crr * mass * 9.81
    c = -power_w
    roots = np.roots([a, 0, b, c])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not real_roots:
        return None
    return distance_m / max(real_roots)


def compare_cda(cda_a, cda_b, mass, crr, rho, power_w, distance_m=40000):
    t_a = estimate_tt_time(cda_a, mass, crr, rho, power_w, distance_m)
    t_b = estimate_tt_time(cda_b, mass, crr, rho, power_w, distance_m)
    if t_a is None or t_b is None:
        return None
    return {
        'time_a':  t_a,
        'time_b':  t_b,
        'delta_s': t_b - t_a,
        'speed_a': (distance_m / t_a) * 3.6,
        'speed_b': (distance_m / t_b) * 3.6,
    }


# ===========================================================================
# Main UI  — complete rewrite
# ===========================================================================

def _quality_badge(overall):
    if overall == 'PASS':    return '✅ PASS'
    if overall == 'WARNING': return '⚠️ WARN'
    return '❌ FAIL'


def main():
    st.set_page_config(
        page_title="Cycling CdA Analyzer",
        page_icon="🚴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .stApp { background-color: #1a1a1a; color: #e0e0e0; }
        section[data-testid="stSidebar"] { background-color: #2b2b2b; }
        .cda-big  { color: #3b8ed0; font-size: 2.6rem; font-weight: bold; text-align: center; }
        .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚴 Cycling CdA Analyzer")
    st.caption("Chung Virtual Elevation Method — upload one or more FIT files, check quality, calculate CdA, compare configurations.")

    # ── Session state init ────────────────────────────────────────────────────
    if 'registry' not in st.session_state:
        st.session_state['registry'] = {}   # key → file record

    registry: dict = st.session_state['registry']

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")

        spd_label = st.radio("Speed source", ["Speed sensor", "GPS"])
        speed_source = "sensor" if spd_label == "Speed sensor" else "gps"

        st.divider()
        mass    = st.number_input("Mass (kg)",           value=95.0,   step=0.5,
                                  min_value=40.0, max_value=200.0)
        crr     = st.number_input("Crr (-)",             value=0.0031, step=0.0001,
                                  format="%.4f", min_value=0.001, max_value=0.010)
        rho     = st.number_input("Air density (kg/m³)", value=1.225,  step=0.001,
                                  format="%.3f", min_value=1.0,   max_value=1.4)
        wind_ms = st.number_input("Wind (m/s)",          value=0.0,    step=0.1,
                                  min_value=-20.0, max_value=20.0,
                                  help="+ = headwind on outbound leg")

        st.divider()
        st.subheader("📁 Upload FIT files")
        uploaded_files = st.file_uploader(
            "Drop one or more .fit files",
            type=["fit"],
            accept_multiple_files=True,
        )

        # ── Process newly uploaded files ──────────────────────────────────────
        if uploaded_files:
            for uf in uploaded_files:
                fkey = f"{uf.name}_{speed_source}"
                if fkey in registry:
                    continue   # already loaded

                with st.spinner(f"Loading {uf.name}…"):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name

                        az = CyclingDataAnalyzer()
                        az.load_fit_file(tmp_path, speed_source=speed_source)
                        os.unlink(tmp_path)

                        qr = check_data_quality(
                            az, mass=mass, crr=crr, rho=rho, wind_ms=wind_ms)

                        registry[fkey] = {
                            'name':         uf.name,
                            'speed_source': speed_source,
                            'analyzer':     az,
                            'quality':      qr,
                            'cda':          None,
                            'cda_params':   None,
                            'loaded_at':    datetime.now().strftime('%H:%M:%S'),
                        }
                        st.success(f"✓ {uf.name}")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

        st.divider()
        if registry and st.button("🗑️ Clear all files", use_container_width=True):
            st.session_state['registry'] = {}
            st.rerun()

    # ── Welcome screen ────────────────────────────────────────────────────────
    if not registry:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.info("👈 Upload one or more FIT files in the sidebar to get started.")
            st.markdown("""
            **Workflow**
            1. Upload FIT file(s) — quality check runs automatically
            2. Select a file and click **Calculate CdA**
            3. Select two files and click **Compare** for a head-to-head report
            4. Export a PDF report at any time

            > CdA is determined by minimising VE loop-closure error between
            > outbound and return legs (Chung method). GPS altitude is not used.
            """)
        return

    # =========================================================================
    # FILE TABLE
    # =========================================================================
    st.subheader("📋 Uploaded files")

    # Build display dataframe
    rows = []
    for fkey, rec in registry.items():
        qr   = rec['quality']
        rows.append({
            'Select':   False,
            '_key':     fkey,
            'File':     rec['name'],
            'Loaded':   rec['loaded_at'],
            'Duration': f"{rec['analyzer'].raw_data['time'].iloc[-1]/60:.1f} min",
            'Avg speed': f"{rec['analyzer'].raw_data['speed'].mean()*3.6:.1f} km/h",
            'Avg power': f"{rec['analyzer'].raw_data['power'].mean():.0f} W",
            'Quality':  f"{qr['score']:.0f}%  {_quality_badge(qr['overall'])}",
            'CdA':      f"{rec['cda']:.4f} m²" if rec['cda'] else "—",
        })

    import pandas as pd
    display_df = pd.DataFrame(rows)

    edited = st.data_editor(
        display_df.drop(columns=['_key']),
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
        },
        disabled=["File","Loaded","Duration","Avg speed","Avg power","Quality","CdA"],
        hide_index=True,
        use_container_width=True,
    )

    # Map selections back to file keys
    selected_mask = edited['Select'].values
    selected_keys = [display_df['_key'].iloc[i]
                     for i, v in enumerate(selected_mask) if v]
    n_selected = len(selected_keys)

    # =========================================================================
    # ACTION BAR
    # =========================================================================
    st.markdown("---")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    btn_calc    = col_a.button("⚡ Calculate CdA",
                               disabled=(n_selected != 1),
                               use_container_width=True,
                               help="Select exactly 1 file")
    btn_compare = col_b.button("⚖️ Compare (2 files)",
                               disabled=(n_selected != 2),
                               use_container_width=True,
                               help="Select exactly 2 files")
    btn_qreport = col_c.button("🔍 Quality Report",
                               disabled=(n_selected != 1),
                               use_container_width=True,
                               help="Select exactly 1 file")
    btn_report  = col_d.button("📑 Analysis Report",
                               disabled=(n_selected != 1),
                               use_container_width=True,
                               help="Select 1 file with a CdA result")
    btn_delete  = col_e.button("🗑️ Remove selected",
                               disabled=(n_selected == 0),
                               use_container_width=True)

    if btn_delete:
        for k in selected_keys:
            registry.pop(k, None)
        st.rerun()

    st.markdown("")

    # =========================================================================
    # CALCULATE CdA  (1 file selected)
    # =========================================================================
    if btn_calc and n_selected == 1:
        fkey = selected_keys[0]
        rec  = registry[fkey]
        az   = rec['analyzer']
        qr   = rec['quality']

        if qr['overall'] == 'FAIL':
            st.error("❌ Quality check FAILED — fix the data before calculating CdA.")
        else:
            if qr['overall'] == 'WARNING':
                st.warning("⚠️ Quality WARNING — proceed carefully.")

            with st.spinner("Calculating CdA…"):
                try:
                    az.set_segment(0, len(az.raw_data) - 1)
                    cda = az.calculate_cda(mass, crr, rho, wind_ms)
                    rec['cda'] = cda
                    rec['cda_params'] = {
                        'cda': cda, 'mass': mass, 'crr': crr,
                        'rho': rho, 'wind_ms': wind_ms,
                    }
                    st.success(f"✅ CdA = **{cda:.4f} m²**  ({rec['name']})")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.code(traceback.format_exc())

    # =========================================================================
    # QUALITY REPORT  (1 file selected)
    # =========================================================================
    if btn_qreport and n_selected == 1:
        fkey = selected_keys[0]
        rec  = registry[fkey]
        qr   = rec['quality']

        st.divider()
        st.subheader(f"🔍 Quality Report — {rec['name']}")

        overall = qr['overall']
        score   = qr['score']
        if overall == 'PASS':
            st.success(f"✅ PASS — Score: {score:.0f}%  "
                       f"({qr['n_pass']} pass, {qr['n_warn']} warn, {qr['n_fail']} fail)")
        elif overall == 'WARNING':
            st.warning(f"⚠️ WARNING — Score: {score:.0f}%")
        else:
            st.error(f"❌ FAIL — Score: {score:.0f}%")

        for key, chk in qr['checks'].items():
            if chk['pass']:
                st.success(f"✅ {chk['label']}  —  {chk['detail']}")
            elif chk.get('warning'):
                st.warning(f"⚠️ {chk['label']}  —  {chk['detail']}")
            else:
                st.error(f"❌ {chk['label']}  —  {chk['detail']}")

        # PDF download
        with st.spinner("Generating quality report PDF…"):
            pdf_bytes = generate_pdf_report(rec['analyzer'], qr, cda_params=None)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("⬇️ Download Quality PDF", data=pdf_bytes,
                           file_name=f"quality_{rec['name']}_{ts}.pdf",
                           mime="application/pdf")

    # =========================================================================
    # ANALYSIS REPORT  (1 file, needs CdA)
    # =========================================================================
    if btn_report and n_selected == 1:
        fkey = selected_keys[0]
        rec  = registry[fkey]

        if rec['cda'] is None:
            st.warning("⚠️ Calculate CdA for this file first.")
        else:
            az  = rec['analyzer']
            p   = rec['cda_params']
            qr  = rec['quality']
            cda = rec['cda']

            st.divider()
            st.subheader(f"📊 Analysis Report — {rec['name']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("CdA",       f"{cda:.4f} m²")
            c2.metric("Pairs",     len(az.pairs))
            c3.metric("Quality",   f"{qr['score']:.0f}%  {_quality_badge(qr['overall'])}")

            # VE closures
            st.markdown("**VE Loop Closures**")
            for k, lc in enumerate(az.leg_closures):
                icon = '✅' if abs(lc) < 0.01 else ('⚠️' if abs(lc) < 0.05 else '❌')
                st.markdown(f"Pair {k+1}: `{lc:+.4f} m` {icon}")

            # Plots in expander (optional, not forced on user)
            with st.expander("📈 View analysis plots", expanded=False):
                with st.spinner("Rendering…"):
                    fig = make_results_plot(az, p['crr'], p['rho'], p['mass'], p['wind_ms'])
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            # PDF
            with st.spinner("Generating PDF…"):
                pdf_bytes = generate_pdf_report(az, qr, cda_params=p)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button("⬇️ Download Analysis PDF", data=pdf_bytes,
                               file_name=f"cda_{rec['name']}_{ts}.pdf",
                               mime="application/pdf", use_container_width=True)

    # =========================================================================
    # COMPARE  (2 files selected)
    # =========================================================================
    if btn_compare and n_selected == 2:
        rec_a = registry[selected_keys[0]]
        rec_b = registry[selected_keys[1]]

        if rec_a['cda'] is None or rec_b['cda'] is None:
            missing = [r['name'] for r in [rec_a, rec_b] if r['cda'] is None]
            st.warning(f"⚠️ Calculate CdA first for: {', '.join(missing)}")
        else:
            st.divider()
            st.subheader("⚖️ Comparison Report")

            cda_a = rec_a['cda']
            cda_b = rec_b['cda']

            # Reference power input
            avg_pwr = (rec_a['analyzer'].raw_data['power'].mean() +
                       rec_b['analyzer'].raw_data['power'].mean()) / 2
            ref_power = st.number_input(
                "Reference TT power (W) — your expected 40km effort",
                min_value=100, max_value=600,
                value=int(avg_pwr), step=5,
            )

            result = compare_cda(cda_a, cda_b, mass, crr, rho, ref_power)

            # ── Header metrics ────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric(f"A — {rec_a['name']}", f"{cda_a:.4f} m²",
                      help=f"Quality: {rec_a['quality']['overall']}")
            c2.metric(f"B — {rec_b['name']}", f"{cda_b:.4f} m²",
                      delta=f"{(cda_b-cda_a)*1000:+.1f} cm²",
                      delta_color="inverse",
                      help=f"Quality: {rec_b['quality']['overall']}")
            c3.metric("Delta CdA", f"{(cda_b-cda_a)*1000:+.1f} cm²")

            # ── TT estimate ───────────────────────────────────────────────────
            if result:
                st.markdown(f"### 40 km TT estimate at {ref_power} W")
                delta_s = result['delta_s']
                verdict = ("🟢 B is FASTER" if delta_s < 0
                           else ("🔴 B is SLOWER" if delta_s > 0 else "➡️ NO DIFFERENCE"))

                c1, c2, c3 = st.columns(3)
                ta, tb = result['time_a'], result['time_b']
                c1.metric("A — time",
                          f"{int(ta//60)}:{int(ta%60):02d}",
                          help=f"{result['speed_a']:.1f} km/h")
                c2.metric("B — time",
                          f"{int(tb // 60)}:{int(tb % 60):02d}",
                          delta=f"{delta_s:+.0f}s",
                          delta_color="inverse")
                c3.metric("Time difference", f"{abs(delta_s):.1f} s", delta=verdict,
                          delta_color="off")

                st.info(
                    f"At {ref_power} W on a flat 40 km course, configuration B "
                    f"(**{rec_b['name']}**) is **{verdict.split()[-1]}** "
                    f"by **{abs(delta_s):.1f} seconds** "
                    f"({'saving' if delta_s < 0 else 'losing'} time) "
                    f"({abs(result['speed_b']-result['speed_a']):.2f} km/h)."
                )

            # ── Plots in expander ─────────────────────────────────────────────
            with st.expander("📈 View plots for both files", expanded=False):
                for label, rec in [("A", rec_a), ("B", rec_b)]:
                    st.markdown(f"**File {label} — {rec['name']}**")
                    p = rec['cda_params']
                    fig = make_results_plot(
                        rec['analyzer'], p['crr'], p['rho'], p['mass'], p['wind_ms'])
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            # ── PDF export ────────────────────────────────────────────────────
            with st.spinner("Generating comparison PDF…"):
                comparison_data = {
                    'baseline':  {
                        'cda':      cda_a,
                        'name':     rec_a['name'],
                        'mass':     rec_a['cda_params']['mass'],
                        'crr':      rec_a['cda_params']['crr'],
                        'rho':      rec_a['cda_params']['rho'],
                        'closures': list(rec_a['analyzer'].leg_closures),
                    },
                    'test_cda':  cda_b,
                    'ref_power': ref_power,
                    'result':    result,
                }
                pdf_bytes = generate_pdf_report(
                    rec_b['analyzer'],
                    rec_b['quality'],
                    cda_params=rec_b['cda_params'],
                    comparison=comparison_data,
                )
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "⬇️ Download Comparison PDF", data=pdf_bytes,
                file_name=f"comparison_{ts}.pdf",
                mime="application/pdf", use_container_width=True,
            )


if __name__ == "__main__":
    main()
