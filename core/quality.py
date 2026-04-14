import numpy as np
import pandas as pd
from core.analyzer import CyclingDataAnalyzer, circular_mean, angle_diff

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
        analyzer.build_legs()

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

    # ── 8. Heading antiparallel (out vs back ≈ 180°) ────────────────────────
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

def _background_cda_closures(analyzer, mass, crr, rho, wind_ms=0.0):
    import copy
    try:
        probe = copy.copy(analyzer)
        probe.segment_data       = analyzer.segment_data
        probe.turnaround_indices = list(analyzer.turnaround_indices)
        probe.legs               = []
        probe.pairs              = []
        probe.leg_closures       = []
        probe.gps_closures       = []
        probe.cda_result         = None
        probe.virtual_elevation  = None

        probe.build_legs()                    # ← default min_trim=5, same as calculate_cda

        if not probe.pairs:
            return None

        grid = np.linspace(0.10, 0.70, 400)  # ← 400, not 200
        errs = [probe._objective(c, crr, mass, rho, wind_ms) for c in grid]
        best = grid[np.argmin(errs)]
        fine = np.linspace(max(0.05, best - 0.05), min(0.80, best + 0.05), 400)  # ← 400
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