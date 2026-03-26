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

def main():
    st.set_page_config(
        page_title="Cycling CdA Analyzer",
        page_icon="🚴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Dark theme CSS ─────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .stApp            { background-color: #1a1a1a; color: #e0e0e0; }
        section[data-testid="stSidebar"] { background-color: #2b2b2b; }
        .result-card {
            background-color: #2b2b2b;
            border: 2px solid #3b8ed0;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        .cda-value { color: #3b8ed0; font-size: 3rem; font-weight: bold; }
        .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚴 Cycling CdA Analyzer")
    st.markdown("**Chung Virtual Elevation Method** – Loop Closure  |  🆓 Free test version")

    # ──────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # File upload
        st.subheader("📁 Data")
        uploaded_file = st.file_uploader(
            "Upload FIT file", type=["fit"],
            help="FIT file met power meter data van een heen-en-terug rit")

        # Speed source
        st.subheader("🏃 Speed Source")
        spd_label = st.radio(
            "Source", ["Speed sensor", "GPS"],
            help="Sensor: ruw signaal (aanbevolen)\nGPS: afgeleid, 5-punt gemiddelde")
        speed_source = "sensor" if spd_label == "Speed sensor" else "gps"
        if speed_source == "sensor":
            st.success("✓ Geen smoothing – raw sensor speed")
        else:
            st.warning("~ GPS smoothing actief (rolling mean 5 pts)")

        # Parameters
        st.subheader("📊 Parameters")
        mass    = st.number_input("Mass (kg)",          value=95.0,   step=0.5,
                                  min_value=40.0, max_value=200.0)
        crr     = st.number_input("Crr (-)",            value=0.0031, step=0.0001,
                                  format="%.4f", min_value=0.001, max_value=0.010)
        rho     = st.number_input("Air density (kg/m³)", value=1.225, step=0.001,
                                  format="%.3f", min_value=1.0, max_value=1.4)
        wind_ms = st.number_input("Wind speed (m/s)",  value=0.0,    step=0.1,
                                  min_value=-20.0, max_value=20.0,
                                  help="+ = kopwind op heenweg  |  - = meewind")
        st.caption("+ = kopwind heen  |  − = meewind heen")

        st.divider()
        calc_btn = st.button("🔄 Calculate CdA",
                             type="primary", use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # WELCOME screen (geen bestand geladen)
    # ──────────────────────────────────────────────────────────────────────
    if uploaded_file is None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.info("👈 Upload een FIT bestand in de sidebar om te beginnen.")
            st.markdown("""
            ### Hoe werkt het?
            1. Upload je `.fit` bestand (heen-en-terug rit, met vermogensmeter)
            2. Stel massa, Crr, luchtdichtheid en wind in
            3. Klik **Calculate CdA**
            4. Bekijk de resultaten en download de rapportage

            ---
            > **Chung methode:** CdA wordt bepaald door de loop-closure fout
            > tussen heen- en terugweg te minimaliseren. GPS-hoogte wordt
            > **niet** gebruikt.
            """)
        return

    # ──────────────────────────────────────────────────────────────────────
    # LOAD  (gecached in session_state zodat de analyzer niet bij elke
    #        slider-interactie opnieuw ingeladen wordt)
    # ──────────────────────────────────────────────────────────────────────
    file_key = f"{uploaded_file.name}_{speed_source}"

    if (st.session_state.get("file_key") != file_key):
        with st.spinner(f"⏳ Loading {uploaded_file.name} …"):
            try:
                # fitparse heeft een echte bestandspad nodig
                with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                analyzer = CyclingDataAnalyzer()
                analyzer.load_fit_file(tmp_path, speed_source=speed_source)
                os.unlink(tmp_path)

                st.session_state["analyzer"]   = analyzer
                st.session_state["file_key"]   = file_key
                st.session_state["cda_params"] = None   # reset vorig resultaat

            except Exception as e:
                st.error(f"❌ Fout bij laden: {e}")
                st.code(traceback.format_exc())
                return

    analyzer: CyclingDataAnalyzer = st.session_state["analyzer"]
    df = analyzer.raw_data
    n  = len(df)

    # ── Bestandsinfo ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records",    f"{n:,}")
    c2.metric("Duration",   f"{df['time'].iloc[-1]:.0f} s")
    c3.metric("Avg speed",  f"{df['speed'].mean()*3.6:.1f} km/h")
    c4.metric("Avg power",  f"{df['power'].mean():.0f} W")

    # ── Volledig rit overzicht ─────────────────────────────────────────────
    with st.expander("📈 Full ride overview", expanded=False):
        fig_raw = make_raw_overview(analyzer)
        st.pyplot(fig_raw, use_container_width=True)
        plt.close(fig_raw)

    # ──────────────────────────────────────────────────────────────────────
    # SEGMENT SELECTIE
    # ──────────────────────────────────────────────────────────────────────
    st.subheader("✂️ Segment selectie")
    c1, c2 = st.columns(2)
    with c1:
        seg_start = st.slider("Start index", 0, n-2, 0)
    with c2:
        seg_end   = st.slider("End index",   1, n-1, n-1)

    if seg_start >= seg_end:
        st.warning("⚠️ Start moet kleiner zijn dan End.")
        return

    # Draai turnaround detectie voor het preview
    analyzer.set_segment(seg_start, seg_end)
    analyzer.detect_turnarounds()

    n_turns = len(analyzer.turnaround_indices)
    if n_turns:
        st.success(f"✅ {n_turns} keerpunt(en) automatisch gedetecteerd "
                   f"(indices: {analyzer.turnaround_indices})")
    else:
        st.warning("⚠️ Geen keerpunten gevonden – check de segmentselectie")

    with st.expander("🗺️ Segment overzicht", expanded=True):
        fig_seg = make_segment_overview(analyzer)
        st.pyplot(fig_seg, use_container_width=True)
        plt.close(fig_seg)

    # ──────────────────────────────────────────────────────────────────────
    # BEREKENING
    # ──────────────────────────────────────────────────────────────────────
    if calc_btn:
        with st.spinner("⚙️ Berekening CdA …"):
            try:
                analyzer.set_segment(seg_start, seg_end)
                cda = analyzer.calculate_cda(mass, crr, rho, wind_ms)
                st.session_state["cda_params"] = {
                    "cda": cda, "mass": mass, "crr": crr,
                    "rho": rho, "wind_ms": wind_ms,
                }
                st.success(f"✅ CdA = **{cda:.4f} m²**")
            except Exception as e:
                st.error(f"❌ Berekeningsfout: {e}")
                st.code(traceback.format_exc())

    # ──────────────────────────────────────────────────────────────────────
    # RESULTATEN
    # ──────────────────────────────────────────────────────────────────────
    if st.session_state.get("cda_params") and analyzer.cda_result is not None:
        p   = st.session_state["cda_params"]
        cda = analyzer.cda_result

        st.divider()
        st.subheader("📊 Resultaten")

        # Groot getal
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"""
            <div class="result-card">
                <div class="cda-value">{cda:.4f} m²</div>
                <p style="color:#aaa; margin-top:.4rem;">CdA  –  Chung Virtual Elevation</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Detail kolommen
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Parameters:**")
            st.table({
                "Parameter": ["Mass", "Crr", "Air density", "Wind"],
                "Value":     [f"{p['mass']:.1f} kg",
                              f"{p['crr']:.5f}",
                              f"{p['rho']:.3f} kg/m³",
                              f"{p['wind_ms']:+.1f} m/s"],
            })
        with c2:
            st.markdown("**VE Loop Closures:**")
            for k, lc in enumerate(analyzer.leg_closures):
                icon = "✅" if abs(lc) < 0.01 else ("⚠️" if abs(lc) < 0.05 else "❌")
                st.markdown(f"Pair {k+1}: `{lc:+.4f} m` {icon}")

            if analyzer.gps_closures:
                st.markdown("**GPS Closures (back→start):**")
                for k, gc in enumerate(analyzer.gps_closures):
                    if np.isnan(gc):
                        st.markdown(f"Pair {k+1}: geen GPS")
                    else:
                        icon = "✅" if gc < 20 else ("⚠️" if gc < 50 else "❌")
                        st.markdown(f"Pair {k+1}: `{gc:.1f} m` {icon}")

        # Resultatenplot
        with st.spinner("Plots genereren …"):
            fig_res = make_results_plot(
                analyzer, p['crr'], p['rho'], p['mass'], p['wind_ms'])
            st.pyplot(fig_res, use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────
        st.subheader("💾 Download")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        c1, c2 = st.columns(2)
        with c1:
            summary = {
                "timestamp":      ts,
                "cda_m2":         float(cda),
                "ve_closures_m":  [float(x) for x in analyzer.leg_closures],
                "gps_closures_m": [float(x) if not np.isnan(x) else None
                                   for x in analyzer.gps_closures],
                "n_legs":         len(analyzer.legs),
                "n_pairs":        len(analyzer.pairs),
                "wind_ms":        float(p['wind_ms']),
                "mass_kg":        float(p['mass']),
                "crr":            float(p['crr']),
                "rho_kg_m3":      float(p['rho']),
            }
            st.download_button(
                "📄 Download JSON samenvatting",
                data=json.dumps(summary, indent=2),
                file_name=f"cda_summary_{ts}.json",
                mime="application/json",
                use_container_width=True)

        with c2:
            buf = io.BytesIO()
            fig_res.savefig(buf, format="png", dpi=150,
                            bbox_inches="tight", facecolor=PANEL_BG)
            buf.seek(0)
            st.download_button(
                "🖼️ Download plot (PNG)",
                data=buf,
                file_name=f"cda_plot_{ts}.png",
                mime="image/png",
                use_container_width=True)

        plt.close(fig_res)


if __name__ == "__main__":
    main()