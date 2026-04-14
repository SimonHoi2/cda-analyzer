import numpy as np
import pandas as pd

try:
    from fitparse import FitFile
    FITPARSE_AVAILABLE = True
except ImportError:
    FITPARSE_AVAILABLE = False


def circular_mean(angles_deg):
    r = np.radians(np.asarray(angles_deg))
    return float(np.degrees(
        np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)

def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)



class CyclingDataAnalyzer:
    """Load FIT data, detect turnarounds, split legs, fit CdA."""

    from utils.constants import G

    def __init__(self):
        self.raw_data = None
        self.segment_data = None
        self.legs = []
        self.pairs = []
        self.turnaround_indices = []
        self.manual_turnaround_indices = None  # ← ADD THIS LINE
        self.cda_result = None
        self.virtual_elevation = None
        self.leg_closures = []
        self.gps_closures = []
        self.pair_cda_values = []
        self.cda_mean = None
        self.cda_std = None

    def load_fit_file(self, filepath, speed_source: str = 'sensor'):
        if not FITPARSE_AVAILABLE:
            raise ImportError("Install fitparse: pip install fitparse")

        rows = []
        for rec in FitFile(filepath).get_messages('record'):
            rows.append({f.name: f.value for f in rec})
        if not rows:
            raise ValueError("No record data in FIT file.")

        df = pd.DataFrame(rows)

        # ── Normalise column names ─────────────────────────────────────────────
        # Rename alternative field names to the canonical names this code uses.
        # Order matters: enhanced_* takes priority over base field if both exist.
        rename_map = {
            'position_lat': 'latitude',
            'position_long': 'longitude',
            'lat': 'latitude',
            'lng': 'longitude',
            'enhanced_altitude': 'altitude',
            'velocity_smooth': 'speed',
            'watts': 'power',
        }
        df = df.rename(columns=rename_map)

        # Handle enhanced_speed carefully: prefer it over the raw speed field
        # because it has higher precision, but avoid creating duplicate columns.
        if 'enhanced_speed' in df.columns:
            df['speed'] = df['enhanced_speed']
            df = df.drop(columns=['enhanced_speed'])

        # Drop any remaining duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # ── Timestamp ──────────────────────────────────────────────────────────
        if 'timestamp' not in df.columns:
            raise ValueError("No timestamp field.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # ── GPS coordinates ────────────────────────────────────────────────────
        # Coerce to float first so abs().max() always returns a scalar.
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            if df['latitude'].notna().any():
                lat_max = float(df['latitude'].abs().max())
                if lat_max > 180:
                    # Values are in FIT semicircles — convert to degrees
                    df['latitude'] = df['latitude'] * (180.0 / 2 ** 31)
                    df['longitude'] = df['longitude'] * (180.0 / 2 ** 31)

        # ── Speed ──────────────────────────────────────────────────────────────
        if speed_source == 'gps':
            if 'latitude' in df.columns and df['latitude'].notna().any():
                df['speed'] = self._speed_from_gps(
                    df['latitude'], df['longitude'], df['timestamp'])
            else:
                raise ValueError("GPS speed selected but no GPS data found.")
            df['speed'] = (pd.to_numeric(df['speed'], errors='coerce')
                           .rolling(5, center=True, min_periods=1)
                           .mean()
                           .clip(lower=0))
        else:
            if 'speed' not in df.columns or df['speed'].isna().all():
                if 'latitude' in df.columns and df['latitude'].notna().any():
                    df['speed'] = self._speed_from_gps(
                        df['latitude'], df['longitude'], df['timestamp'])
                else:
                    raise ValueError("No speed or GPS data.")
            else:
                df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
                spd_median = float(df['speed'].median())
                if spd_median > 50:
                    df['speed'] = df['speed'] / 1000.0
            df['speed'] = df['speed'].ffill().bfill().clip(lower=0)

        # ── Power ──────────────────────────────────────────────────────────────
        if 'power' not in df.columns:
            raise ValueError("No power data — a power meter is required.")
        df['power'] = (pd.to_numeric(df['power'], errors='coerce')
                       .ffill().bfill().clip(lower=0))

        # ── Altitude ───────────────────────────────────────────────────────────
        if 'altitude' not in df.columns:
            df['altitude'] = np.nan
        else:
            df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')

        # ── Distance ───────────────────────────────────────────────────────────
        if 'distance' not in df.columns or df['distance'].isna().all():
            if 'latitude' in df.columns and df['latitude'].notna().any():
                df['distance'] = self._dist_from_gps(
                    df['latitude'], df['longitude'])
            else:
                df['distance'] = self._dist_from_speed(
                    df['speed'], df['timestamp'])
        else:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
            # FIT distance is stored in cm (scale=100) — convert if needed
            dist_max = float(df['distance'].max())
            if dist_max > 1_000_000:
                df['distance'] = df['distance'] / 100.0

        # ── Heading ────────────────────────────────────────────────────────────
        if 'latitude' in df.columns and df['latitude'].notna().any():
            df['heading'] = self._heading_from_gps(
                df['latitude'].values, df['longitude'].values)
        else:
            df['heading'] = np.nan

        # ── Final cleanup ──────────────────────────────────────────────────────
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
                             .iloc[start_idx:end_idx + 1]
                             .copy()
                             .reset_index(drop=True))
        self.legs = []
        self.pairs = []
        self.turnaround_indices = []
        #self.manual_turnaround_indices = None  # ← ADD THIS LINE
        self.cda_result = None
        self.virtual_elevation = None
        self.leg_closures = []
        self.gps_closures = []
        self.pair_cda_values = []
        self.cda_mean = None
        self.cda_std = None

    def detect_turnarounds(self, speed_threshold=1.5):
        # ── Manual override: skip auto-detection if user set turnarounds ──
        if self.manual_turnaround_indices is not None:
            self.turnaround_indices = list(self.manual_turnaround_indices)
            return self.turnaround_indices

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

    # ── in CyclingDataAnalyzer ─────────────────────────────────────────────────

    def build_legs(self, min_trim: int = 5, balance: bool = True):
        """
        Build out-and-back legs with adaptive, distance-balanced turnaround trimming.

        Three-step improvement over the old fixed trim=10:

        Step 1 – Speed-based turn-zone detection:
            Walk away from each turnaround until speed recovers to ≥70 % of the
            segment median.  This captures the true invalidated zone, which varies
            by corner radius and rider behaviour.

        Step 2 – Distance-symmetric trim:
            The approach end-point and the departure start-point of every turnaround
            are made to lie at the same physical distance from the cone, by
            extending whichever side is shorter.  The two trim edges thus coincide
            geographically, minimising unused road.

        Step 3 – Pair distance balancing (when balance=True):
            Speed × time distance is computed for each (out, back) pair.  Any
            residual imbalance is corrected by trimming from the *far* (outer)
            boundary of the longer leg — the start of the out leg or the end of
            the back leg — leaving the turnaround region untouched.
        """
        if self.segment_data is None:
            return []

        turns = self.turnaround_indices or self.detect_turnarounds()
        bounds = [0] + turns + [len(self.segment_data) - 1]

        df = self.segment_data
        spd = df['speed'].values
        t = df['time'].values
        dt = np.diff(t, prepend=t[0])
        step_d = spd * dt  # metres per sample
        cum_dist = np.cumsum(step_d)
        n = len(df)

        # Segment-wide riding speed (used for recovery threshold)
        riding_spd = np.median(spd[spd > 3.0]) if (spd > 3.0).any() else 5.0
        recovery_thr = riding_spd * 0.90  # 90 % = "back to steady riding"

        # ── Step 1 & 2: adaptive, distance-symmetric trim per turnaround ────────
        def _adaptive_trim(ti: int, max_search: int = 150):
            """
            Return (trim_left, trim_right) with:
              - speed must recover to ≥ 90% of median riding speed
              - acceleration must drop below 0.3 m/s² for 3 consecutive samples
            """
            # Smoothed acceleration (used for stability check)
            acc = np.gradient(
                pd.Series(spd).rolling(5, center=True, min_periods=1).mean().values,
                t
            )

            def _stable(idx):
                """True if acceleration is settled at index idx."""
                if idx < 2 or idx >= n - 2:
                    return False
                window = acc[max(0, idx - 2): idx + 3]
                return np.all(np.abs(window) < 0.3)

            # ── approach side (walk backward from turnaround) ──
            tl = min_trim
            for i in range(ti - 1, max(0, ti - max_search) - 1, -1):
                if spd[i] >= recovery_thr and _stable(i):
                    tl = ti - i
                    break
            tl = max(min_trim, tl)

            # ── departure side (walk forward from turnaround) ──
            tr = min_trim
            for i in range(ti + 1, min(n, ti + max_search)):
                if spd[i] >= recovery_thr and _stable(i):
                    tr = i - ti
                    break
            tr = max(min_trim, tr)

            # ── distance-symmetric trim (Step 2 — unchanged) ──
            d_left = cum_dist[ti] - cum_dist[max(0, ti - tl)]
            d_right = cum_dist[min(n - 1, ti + tr)] - cum_dist[ti]
            target = max(d_left, d_right)

            if d_left < target:
                for i in range(ti - tl - 1, max(0, ti - max_search * 2) - 1, -1):
                    if cum_dist[ti] - cum_dist[max(0, i)] >= target:
                        tl = ti - i
                        break

            if d_right < target:
                for i in range(ti + tr + 1, min(n, ti + max_search * 2)):
                    if cum_dist[i] - cum_dist[ti] >= target:
                        tr = i - ti
                        break

            return max(min_trim, tl), max(min_trim, tr)

        # Trim table: trims[k] = (left_trim, right_trim) at bounds[k]
        # Protocol: rider is at speed at file start/end → minimal boundary trim.
        # Turnarounds get full adaptive trim.
        BOUNDARY_TRIM = 2

        trims = (
                [(0, BOUNDARY_TRIM)]
                + [_adaptive_trim(ti) for ti in turns]
                + [(BOUNDARY_TRIM, 0)]
        )

        # ── Slice legs ──────────────────────────────────────────────────────────
        legs = []
        for k in range(len(bounds) - 1):
            s = bounds[k] + trims[k][1]  # right trim of left boundary
            e = bounds[k + 1] - trims[k + 1][0]  # left  trim of right boundary
            if e - s < 20:
                continue
            leg = df.iloc[s:e + 1].copy().reset_index(drop=True)
            if len(leg) < 20 or leg['speed'].mean() < 2.0:
                continue
            legs.append(leg)

        # ── Step 3: pair distance balancing ─────────────────────────────────────
        def _leg_dist(lg):
            v = lg['speed'].values
            t_l = lg['time'].values
            dt_l = np.diff(t_l, prepend=t_l[0])
            return float(np.sum(v * dt_l))

        def _trim_start(lg, target_cut_m):
            """Remove ~target_cut_m metres from the START of a leg."""
            sd = lg['speed'].values * np.diff(lg['time'].values,
                                              prepend=lg['time'].values[0])
            cut = int(np.searchsorted(np.cumsum(sd), target_cut_m))
            cut = min(cut, len(lg) - 20)
            return lg.iloc[max(0, cut):].reset_index(drop=True)

        def _trim_end(lg, target_cut_m):
            """Remove ~target_cut_m metres from the END of a leg."""
            sd = lg['speed'].values * np.diff(lg['time'].values,
                                              prepend=lg['time'].values[0])
            cut = int(np.searchsorted(np.cumsum(sd[::-1]), target_cut_m))
            cut = min(cut, len(lg) - 20)
            return lg.iloc[:len(lg) - max(0, cut)].reset_index(drop=True)

        if balance and len(legs) >= 2:
            balanced = []
            for ki in range(0, len(legs) - 1, 2):
                out_l = legs[ki]
                back_l = legs[ki + 1]

                d_out = _leg_dist(out_l)
                d_back = _leg_dist(back_l)
                excess = d_out - d_back  # positive → out is longer

                if abs(excess) > 5:
                    # ── Chung method: both legs must cover the same road ──
                    # The turnaround side is the geographic anchor (don't touch).
                    # Trim from the OUTER edge of the LONGER leg only.
                    #   Out leg outer edge  = its START
                    #   Back leg outer edge = its END
                    # The shorter leg stays untouched — it defines the valid road.
                    if excess > 0:
                        # Out is longer → trim from START of out (outer edge)
                        out_l = _trim_start(out_l, abs(excess))
                    else:
                        # Back is longer → trim from END of back (outer edge)
                        back_l = _trim_end(back_l, abs(excess))

                balanced += [out_l, back_l]

            if len(legs) % 2 == 1:
                balanced.append(legs[-1])
            legs = balanced

        # ── Finalise ────────────────────────────────────────────────────────────
        self.legs = legs
        self.pairs = [(legs[k], legs[k + 1]) for k in range(0, len(legs) - 1, 2)]

        # Store distance imbalance per pair as a diagnostic attribute
        self.pair_dist_balance = []
        for out_l, back_l in self.pairs:
            self.pair_dist_balance.append(_leg_dist(out_l) - _leg_dist(back_l))

        # GPS closures (unchanged logic)
        self.gps_closures = []
        for out_leg, back_leg in self.pairs:
            if ('latitude' not in out_leg.columns
                    or out_leg['latitude'].isna().all()):
                self.gps_closures.append(np.nan)
                continue
            lat1, lon1 = out_leg['latitude'].iloc[0], out_leg['longitude'].iloc[0]
            lat2, lon2 = back_leg['latitude'].iloc[-1], back_leg['longitude'].iloc[-1]
            if any(pd.isna([lat1, lon1, lat2, lon2])):
                self.gps_closures.append(np.nan)
                continue
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
        legs = self.build_legs()

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

        # ── Per-pair CdA for confidence interval ─────────────────────────
        self.pair_cda_values = []
        for o, b in self.pairs:
            def _single_pair_obj(cda_test, _o=o, _b=b):
                return abs(
                    self._leg_delta(_o, cda_test, crr, mass, rho, +wind_ms) +
                    self._leg_delta(_b, cda_test, crr, mass, rho, -wind_ms)
                )

            sg = np.linspace(0.10, 0.70, 400)
            se = [_single_pair_obj(c) for c in sg]
            sb = sg[np.argmin(se)]
            sf = np.linspace(max(0.05, sb - 0.05), min(0.80, sb + 0.05), 400)
            sf_e = [_single_pair_obj(c) for c in sf]
            self.pair_cda_values.append(float(sf[np.argmin(sf_e)]))

        self.cda_mean = float(np.mean(self.pair_cda_values))
        self.cda_std = float(np.std(self.pair_cda_values, ddof=1)) if len(self.pair_cda_values) > 1 else 0.0

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
