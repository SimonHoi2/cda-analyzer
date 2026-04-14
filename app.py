
#!/usr/bin/env python3
"""
Cycling CdA Analyzer - Web Version (Streamlit)
Chung Virtual Elevation Method
"""

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

# Internal modules
from core.analyzer import CyclingDataAnalyzer
from core.quality import check_data_quality
from plots.diagnostic import make_diagnostic_plot
from plots.style import PANEL_BG, SPEED_CLR, POWER_CLR, style_ax
from plots.analysis import make_raw_overview, make_segment_overview, make_results_plot
from utils.history import add_entry, get_history, delete_entry, clear_history, reorder_history
# ===========================================================================
# PDF helpers
# ===========================================================================

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

def _cell(pdf, w, h, txt="", *args, **kwargs):
    return pdf.cell(w, h, _pdf_safe(txt), *args, **kwargs)

def _multi_cell(pdf, w, h, txt="", *args, **kwargs):
    return pdf.multi_cell(w, h, _pdf_safe(txt), *args, **kwargs)

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

    # Try platform-specific Unicode fonts, fall back to built-in Helvetica
    font_candidates = [
        # Windows
        (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
        # Linux (DejaVu is pre-installed on most distros / Render)
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]

    FONT_FAMILY = "Helvetica"  # safe default
    for font_regular, font_bold in font_candidates:
        if os.path.exists(font_regular) and os.path.exists(font_bold):
            pdf.add_font("UI", "", font_regular, uni=True)
            pdf.add_font("UI", "B", font_bold, uni=True)
            FONT_FAMILY = "UI"
            break

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
# UI helpers
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

    # Custom CSS — only for classes not covered by .streamlit/config.toml
    st.markdown("""
        <style>
        .cda-big {
            color: #3b8ed0;
            font-size: 2.6rem;
            font-weight: bold;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🚴 Cycling CdA Analyzer")
    st.caption("Chung Virtual Elevation Method — upload one or more FIT files, check quality, calculate CdA, compare configurations.")

    # ── Session state init ────────────────────────────────────────────────────
    if 'registry' not in st.session_state:
        st.session_state['registry'] = {}
    if 'crop_key' not in st.session_state:
        st.session_state['crop_key'] = None
    if 'report_key' not in st.session_state:
        st.session_state['report_key'] = None
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0
    if 'deleted_keys' not in st.session_state:
        st.session_state['deleted_keys'] = set()
    if 'selected_keys' not in st.session_state:
        st.session_state['selected_keys'] = []
    if 'active_panel' not in st.session_state:  # ← ADD
        st.session_state['active_panel'] = None  # ← ADD

    registry: dict = st.session_state['registry']

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🚴 CdA Analyzer")

        # ── Section 1: File Upload (top priority = top position) ──────────
        st.subheader("📁 Upload FIT Files")
        uploaded_files = st.file_uploader(
            "Drop one or more .fit files",
            type=["fit"],
            accept_multiple_files=True,
            key=f"fit_uploader_{st.session_state['uploader_key']}",
        )

        # ── Section 2: Speed Source ───────────────────────────────────────
        st.divider()
        spd_label = st.radio(
            "Speed source",
            ["Speed sensor", "GPS"],
            horizontal=True,
            help="**Speed sensor** = wheel/hub sensor (recommended). "
                 "**GPS** = derived from coordinates (less precise).",
        )
        speed_source = "sensor" if spd_label == "Speed sensor" else "gps"

        # ── Section 3: Physics Parameters (collapsed by default) ──────────
        with st.expander("⚙️ Physics Parameters", expanded=False):
            mass = st.number_input(
                "Total mass — rider + bike (kg)",
                value=95.0, step=0.5,
                min_value=40.0, max_value=200.0,
                help="Include rider, bike, clothing, bottles, etc."
            )
            crr = st.number_input(
                "Crr — rolling resistance",
                value=0.0031, step=0.0001,
                format="%.4f", min_value=0.001, max_value=0.010,
                help="Typical values:\n"
                     "- Road clincher: 0.003–0.005\n"
                     "- TT tubular: 0.002–0.003\n"
                     "- Velodrome: 0.001–0.002"
            )
            rho = st.number_input(
                "Air density (kg/m³)",
                value=1.225, step=0.001,
                format="%.3f", min_value=1.0, max_value=1.4,
                help="Depends on altitude, temperature, humidity:\n"
                     "- Sea level, 15 °C: 1.225\n"
                     "- Sea level, 30 °C: 1.165\n"
                     "- 500 m altitude, 20 °C: 1.167"
            )
            wind_ms = st.number_input(
                "Wind (m/s)",
                value=0.0, step=0.1,
                min_value=-20.0, max_value=20.0,
                help="Positive = headwind on outbound leg. "
                     "Leave at 0 if you don't know the wind."
            )

        # ── File sync: detect files removed via the ✕ button ─────────────
        if uploaded_files is not None:
            current_fkeys = {f"{uf.name}_{speed_source}" for uf in uploaded_files}
        else:
            current_fkeys = set()

        prev_fkeys = st.session_state.get('upload_fkeys', None)
        if prev_fkeys is not None:
            removed = prev_fkeys - current_fkeys
            for fkey in removed:
                registry.pop(fkey, None)
                st.session_state['deleted_keys'].discard(fkey)
        st.session_state['upload_fkeys'] = current_fkeys

        # ── Process newly uploaded files ──────────────────────────────────
        if uploaded_files:
            for uf in uploaded_files:
                fkey = f"{uf.name}_{speed_source}"
                if fkey in registry or fkey in st.session_state['deleted_keys']:
                    continue

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
                            'name': uf.name,
                            'speed_source': speed_source,
                            'analyzer': az,
                            'quality': qr,
                            'cda': None,
                            'cda_params': None,
                            'loaded_at': datetime.now().strftime('%H:%M:%S'),
                        }
                        st.success(f"✓ {uf.name}")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

        # ── Recalculate All button ────────────────────────────────────────
        st.divider()
        if registry:
            st.caption(f"📂 {len(registry)} file(s) loaded")
            if st.button("🔄 Recalculate All", use_container_width=True,
                         type="primary",
                         help="Re-run quality checks and CdA with the "
                              "current parameter values"):
                progress = st.progress(0, text="Recalculating…")
                keys = list(registry.keys())
                for i, fkey in enumerate(keys):
                    rec = registry[fkey]
                    az = rec['analyzer']

                    qr = check_data_quality(
                        az, mass=mass, crr=crr, rho=rho, wind_ms=wind_ms)
                    rec['quality'] = qr

                    if rec['cda'] is not None:
                        try:
                            az.set_segment(0, len(az.raw_data) - 1)
                            cda = az.calculate_cda(mass, crr, rho, wind_ms)
                            rec['cda'] = cda
                            rec['cda_params'] = {
                                'cda': cda, 'mass': mass, 'crr': crr,
                                'rho': rho, 'wind_ms': wind_ms,
                            }
                        except Exception as e:
                            st.warning(f"⚠️ {rec['name']}: {e}")
                            rec['cda'] = None
                            rec['cda_params'] = None

                    progress.progress((i + 1) / len(keys),
                                      text=f"Recalculated {rec['name']}")

                progress.empty()
                st.success(
                    f"✅ Recalculated {len(keys)} file(s) with: "
                    f"mass={mass}kg, Crr={crr}, ρ={rho}, wind={wind_ms}m/s"
                )
                st.rerun()
        else:
            st.caption("No files loaded yet.")

    # ── Welcome screen (no files loaded) ──────────────────────────────────
    if not registry:
        st.markdown("")

        hero_left, hero_right = st.columns([3, 2], gap="large")

        with hero_left:
            st.markdown("## Determine Your CdA From a FIT File")
            st.markdown(
                "Upload one or more `.fit` files from an out-and-back "
                "aerodynamic test. The app uses the **Chung Virtual Elevation "
                "method** to calculate your drag area (CdA) — no wind tunnel "
                "required."
            )
            st.info("👈 **Upload a .fit file** in the sidebar to get started.")

        with hero_right:
            st.markdown("#### ⏱ Quick Start")
            st.markdown(
                "**1️⃣ Upload** — Drop a FIT file → quality check runs instantly\n\n"
                "**2️⃣ Analyse** — Select the file → click **Calculate CdA**\n\n"
                "**3️⃣ Compare** — Select two files → get a head-to-head report\n\n"
                "**4️⃣ Export** — Download a PDF report at any time"
            )

        st.divider()

        # ── How to get a good test ────────────────────────────────────────
        tip_col1, tip_col2, tip_col3 = st.columns(3)

        with tip_col1:
            st.markdown("##### 🛣️ Route")
            st.markdown(
                "- Flat, straight road\n"
                "- 1–2 km each way\n"
                "- Minimal traffic\n"
                "- Low crosswind exposure"
            )

        with tip_col2:
            st.markdown("##### 🚴 Riding")
            st.markdown(
                "- Steady power (≥ 200 W)\n"
                "- Constant speed ≥ 25 km/h\n"
                "- Hold aero position\n"
                "- ≥ 3 out-and-back laps"
            )

        with tip_col3:
            st.markdown("##### 📡 Equipment")
            st.markdown(
                "- Power meter (required)\n"
                "- Speed sensor (recommended)\n"
                "- GPS head unit\n"
                "- Record at 1 s intervals"
            )

        st.divider()
        st.caption(
            "ℹ️ CdA is determined by minimising Virtual Elevation "
            "loop-closure error between outbound and return legs "
            "(Chung method). GPS altitude is not used — only speed, "
            "power, and heading."
        )
        return

    # =========================================================================
    # FILE TABLE
    # =========================================================================
    st.subheader("📋 Uploaded Files")

    # ── Status summary metrics ────────────────────────────────────────────
    n_total = len(registry)
    n_pass = sum(1 for r in registry.values()
                 if r['quality']['overall'] == 'PASS')
    n_warn = sum(1 for r in registry.values()
                 if r['quality']['overall'] == 'WARNING')
    n_fail = sum(1 for r in registry.values()
                 if r['quality']['overall'] == 'FAIL')
    n_cda = sum(1 for r in registry.values()
                if r['cda'] is not None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Files loaded", n_total)
    m2.metric("Quality ✅", n_pass,
              help="Files that passed all quality checks")
    m3.metric("Warnings / Fails", f"{n_warn} / {n_fail}",
              help="⚠️ = borderline data, ❌ = data issue blocks CdA calculation")
    m4.metric("CdA calculated", f"{n_cda} / {n_total}",
              help="Files with a CdA result. Select a file and click "
                   "Calculate CdA to analyse.")

    st.markdown("")

    # ── Build display dataframe ───────────────────────────────────────────
    rows = []
    for fkey, rec in registry.items():
        qr = rec['quality']
        az = rec['analyzer']
        df = az.raw_data

        rows.append({
            'Select': False,
            '_key': fkey,
            'File': rec['name'],
            'Loaded': rec['loaded_at'],
            'Duration': f"{df['time'].iloc[-1] / 60:.1f} min",
            'Avg speed': f"{df['speed'].mean() * 3.6:.1f} km/h",
            'Avg power': f"{df['power'].mean():.0f} W",
            'Quality': f"{_quality_badge(qr['overall'])}  {qr['score']:.0f}%",
            'CdA': f"{rec['cda']:.4f}" if rec['cda'] else "—",
        })

    display_df = pd.DataFrame(rows)

    edited = st.data_editor(
        display_df.drop(columns=['_key']),
        column_config={
            "Select": st.column_config.CheckboxColumn("✓", width="small",
                                                      help="Tick one file for analysis, or two for comparison"),
            "File": st.column_config.TextColumn("File", width="medium"),
            "Loaded": st.column_config.TextColumn("Loaded", width="small"),
            "Duration": st.column_config.TextColumn("Duration", width="small"),
            "Avg speed": st.column_config.TextColumn("Avg Speed", width="small"),
            "Avg power": st.column_config.TextColumn("Avg Power", width="small"),
            "Quality": st.column_config.TextColumn("Quality", width="small"),
            "CdA": st.column_config.TextColumn("CdA (m²)", width="small"),
        },
        disabled=["File", "Loaded", "Duration", "Avg speed",
                  "Avg power", "Quality", "CdA"],
        hide_index=True,
        use_container_width=True,
    )



    # ── Map selections back to file keys (persisted) ─────────────────────
    selected_mask = edited['Select'].values
    fresh_selected = [display_df['_key'].iloc[i]
                      for i, v in enumerate(selected_mask) if v]

    if fresh_selected:
        st.session_state['selected_keys'] = fresh_selected
    else:
        st.session_state['selected_keys'] = [
            k for k in st.session_state['selected_keys']
            if k in registry
        ]

    selected_keys = st.session_state['selected_keys']
    n_selected = len(selected_keys)

    # =========================================================================
    # ACTION BAR
    # =========================================================================
    st.markdown("---")

    act_col1, act_col2, act_col3, act_col4 = st.columns([3, 3, 1, 1])

    btn_calc = False
    btn_compare = False

    with act_col1:
        if n_selected == 0:
            st.button("☝️ Select one or two files above",
                      disabled=True, use_container_width=True)
        elif n_selected == 1:
            btn_calc = st.button("⚡ Calculate CdA",
                                 type="primary", use_container_width=True,
                                 help="Run the Chung method on the selected file")
        elif n_selected == 2:
            btn_compare = st.button("⚖️ Compare Two Files",
                                    type="primary", use_container_width=True,
                                    help="Head-to-head CdA and 40 km TT comparison")
        else:
            st.warning("Select **1** file for analysis or **2** for comparison.")

    btn_qreport = False
    btn_report = False
    btn_diag = False
    btn_crop = False

    with act_col2:
        if n_selected == 1:
            secondary_action = st.selectbox(
                "More actions",
                ["— Choose —",
                 "🔍 Quality Report",
                 "📑 Analysis Report",
                 "🔬 Diagnose",
                 "✂️ Crop"],
                label_visibility="collapsed",
                key="secondary_action_select",
            )
        else:
            secondary_action = "— Choose —"
            st.selectbox(
                "More actions",
                ["— Select 1 file —"],
                disabled=True,
                label_visibility="collapsed",
            )

    with act_col3:
        can_go = (n_selected == 1
                  and secondary_action != "— Choose —")
        btn_go = st.button("▶️", disabled=not can_go,
                           use_container_width=True,
                           help="Run the selected action")

    with act_col4:
        btn_delete = st.button("🗑️", disabled=(n_selected == 0),
                               use_container_width=True,
                               help="Remove selected file(s)")

    if btn_go and can_go:
        if secondary_action.startswith("🔍"):
            btn_qreport = True
            st.session_state['active_panel'] = 'quality'
            st.session_state['report_key'] = None
        elif secondary_action.startswith("📑"):
            btn_report = True
            st.session_state['active_panel'] = 'report'
        elif secondary_action.startswith("🔬"):
            btn_diag = True
            st.session_state['active_panel'] = 'diagnose'
            st.session_state['report_key'] = None
        elif secondary_action.startswith("✂️"):
            btn_crop = True
            st.session_state['active_panel'] = 'crop'
            st.session_state['report_key'] = None

    if btn_calc:
        st.session_state['active_panel'] = 'calc'
        st.session_state['report_key'] = None

    if btn_compare:
        st.session_state['active_panel'] = 'compare'
        st.session_state['report_key'] = None

    if btn_crop and n_selected == 1:
        st.session_state['crop_key'] = selected_keys[0]

    if (st.session_state.get('crop_key')
            and st.session_state['crop_key'] not in registry):
        st.session_state['crop_key'] = None

    # =========================================================================
    # DIAGNOSE  (1 file selected)
    # =========================================================================
    if btn_diag and n_selected == 1 and st.session_state.get('active_panel') == 'diagnose':
        fkey = selected_keys[0]
        rec = registry[fkey]
        az = rec['analyzer']

        st.divider()
        st.subheader(f"🔬 Diagnostic — {rec['name']}")

        az.set_segment(0, len(az.raw_data) - 1)

        # ── restore manual turnarounds ──
        _mk = f"manual_turns_{fkey}"
        if _mk in st.session_state and st.session_state[_mk] is not None:
            az.manual_turnaround_indices = list(st.session_state[_mk])
        # ────────────────────────────────

        az.detect_turnarounds()

        if not az.turnaround_indices:
            st.warning("⚠️ No turnarounds detected — nothing to diagnose.")
        else:
            az.build_legs()

            # Summary table — uses actual balanced legs
            rows_diag = []
            for k, leg in enumerate(az.legs):
                t_leg = leg['time'].values
                dt_leg = np.diff(t_leg, prepend=t_leg[0])
                d_m = float(np.sum(leg['speed'].values * dt_leg))
                dur_s = float(t_leg[-1] - t_leg[0])

                # Pair distance difference
                pair_idx = k + 1 if k % 2 == 0 else k - 1
                delta_str = ""
                if 0 <= pair_idx < len(az.legs):
                    t_p = az.legs[pair_idx]['time'].values
                    dt_p = np.diff(t_p, prepend=t_p[0])
                    d_p = float(np.sum(az.legs[pair_idx]['speed'].values * dt_p))
                    delta_str = f"{abs(d_m - d_p):.0f} m"

                rows_diag.append({
                    'Leg': f"Leg {k + 1} ({'out' if k % 2 == 0 else 'back'})",
                    'Start (s)': f"{t_leg[0]:.0f}",
                    'End (s)': f"{t_leg[-1]:.0f}",
                    'Duration': f"{dur_s:.0f} s",
                    'Distance': f"{d_m / 1000:.3f} km",
                    'Pair Δ': delta_str,
                })

            st.markdown("**Leg summary** (distances from v × dt, crop zones excluded)")
            st.dataframe(pd.DataFrame(rows_diag), hide_index=True,
                         use_container_width=True)

            with st.spinner("Rendering diagnostic plot…"):
                fig_diag = make_diagnostic_plot(az)
                st.pyplot(fig_diag, use_container_width=True)
                plt.close(fig_diag)

            st.caption(
                "Red dashed lines = detected turnarounds  |  "
                "Red shaded zones = data cropped out (±10 pts)  |  "
                "Coloured segments = usable leg data  |  "
                "Distances calculated from speed × time"
            )

    # =========================================================================
    # DELETE selected files
    # =========================================================================
    if btn_delete:
        for k in selected_keys:
            registry.pop(k, None)
            st.session_state['deleted_keys'].add(k)
        if st.session_state.get('report_key') in selected_keys:
            st.session_state['report_key'] = None
        st.session_state['selected_keys'] = []
        st.rerun()


    # =========================================================================
    # CROP EDITOR
    # =========================================================================
    crop_key = st.session_state.get('crop_key')
    if crop_key and crop_key in registry and st.session_state.get('active_panel') == 'crop':
        rec = registry[crop_key]
        az = rec['analyzer']
        df = az.raw_data

        st.divider()
        st.subheader(f"✂️ Crop Editor — {rec['name']}")
        st.caption(
            "Drag the sliders to remove messy start/end sections. "
            "The **grey area is discarded**, the **coloured area is kept**."
        )

        total_time = float(df['time'].iloc[-1])
        total_pts = len(df)

        # ── Quick-preset buttons ──────────────────────────────────────────
        st.markdown("**Quick presets:**")
        pre1, pre2, pre3, pre4 = st.columns(4)

        # Read current slider value (or default to full range)
        current_range = st.session_state.get(
            'crop_slider', (0.0, round(total_time, 1)))

        if pre1.button("Trim 30 s start", use_container_width=True):
            st.session_state['crop_slider'] = (
                30.0, current_range[1])
            st.rerun()
        if pre2.button("Trim 30 s end", use_container_width=True):
            st.session_state['crop_slider'] = (
                current_range[0], round(total_time - 30, 1))
            st.rerun()
        if pre3.button("Trim 60 s both", use_container_width=True):
            st.session_state['crop_slider'] = (
                60.0, round(total_time - 60, 1))
            st.rerun()
        if pre4.button("↩️ Reset to full", use_container_width=True):
            st.session_state['crop_slider'] = (
                0.0, round(total_time, 1))
            st.rerun()

        # ── Range slider (time-based) ─────────────────────────────────────
        t_min, t_max = st.slider(
            "Select time range to **keep** (seconds)",
            min_value=0.0,
            max_value=round(total_time, 1),
            value=current_range,
            step=0.5,
            format="%.1f s",
            key="crop_slider",
            help="Drag the handles to exclude warm-up, cool-down, or "
                 "messy sections. Only the data between the two handles "
                 "will be used for analysis."
        )

        # Map times back to indices
        idx_start = int(np.searchsorted(df['time'].values, t_min))
        idx_end = int(np.searchsorted(df['time'].values, t_max,
                                      side='right')) - 1
        idx_start = max(0, idx_start)
        idx_end = min(total_pts - 1, idx_end)
        kept = idx_end - idx_start + 1

        # ── Info row 1: point counts ──────────────────────────────────────
        ci1, ci2, ci3, ci4 = st.columns(4)
        ci1.metric("Original points", f"{total_pts:,}")
        ci2.metric("Keeping", f"{kept:,}",
                   help="Number of data points in the selected region")
        ci3.metric("Removing start",
                   f"{idx_start:,} pts  ({t_min:.1f} s)")
        ci4.metric("Removing end",
                   f"{total_pts - 1 - idx_end:,} pts  "
                   f"({total_time - t_max:.1f} s)")

        # ── Info row 2: live statistics of kept region ────────────────────
        kept_df = df.iloc[idx_start:idx_end + 1]
        kept_spd_avg = kept_df['speed'].mean() * 3.6
        kept_pwr_avg = kept_df['power'].mean()
        full_spd_avg = df['speed'].mean() * 3.6
        full_pwr_avg = df['power'].mean()

        ci5, ci6, ci7, ci8 = st.columns(4)
        ci5.metric(
            "Avg speed (kept)",
            f"{kept_spd_avg:.1f} km/h",
            delta=f"{kept_spd_avg - full_spd_avg:+.1f} vs full ride",
            help="Average speed in the selected region",
        )
        ci6.metric(
            "Avg power (kept)",
            f"{kept_pwr_avg:.0f} W",
            delta=f"{kept_pwr_avg - full_pwr_avg:+.0f} vs full ride",
            help="Average power in the selected region",
        )
        ci7.metric(
            "Duration (kept)",
            f"{(t_max - t_min) / 60:.1f} min",
        )
        ci8.metric(
            "% of ride kept",
            f"{kept / total_pts * 100:.0f}%",
        )

        # ── Preview plot ──────────────────────────────────────────────────
        fig_crop, (ax_s, ax_p) = plt.subplots(
            2, 1, figsize=(14, 5), sharex=True, facecolor=PANEL_BG
        )
        style_ax(ax_s)
        style_ax(ax_p)

        time_arr = df['time'].values
        speed_arr = df['speed'].values * 3.6
        power_arr = df['power'].values

        # Full ride in grey
        ax_s.fill_between(time_arr, speed_arr, color='#333333', alpha=0.4)
        ax_p.fill_between(time_arr, power_arr, color='#333333', alpha=0.4)

        # Kept section highlighted
        keep_mask = (time_arr >= t_min) & (time_arr <= t_max)
        ax_s.plot(time_arr[keep_mask], speed_arr[keep_mask],
                  color=SPEED_CLR, lw=1.5)
        ax_p.plot(time_arr[keep_mask], power_arr[keep_mask],
                  color=POWER_CLR, lw=1.5)

        # Crop boundaries
        for ax in (ax_s, ax_p):
            ax.axvline(t_min, color='#f44336', lw=2, ls='--', alpha=0.9)
            ax.axvline(t_max, color='#f44336', lw=2, ls='--', alpha=0.9)
            ax.axvspan(0, t_min, color='#f44336', alpha=0.08)
            ax.axvspan(t_max, total_time, color='#f44336', alpha=0.08)
            ax.grid(True, alpha=0.12)

        ax_s.set_ylabel("Speed (km/h)")
        ax_s.set_title("Crop Preview — red zones will be removed",
                       color="white", fontweight="bold", fontsize=10)
        ax_p.set_ylabel("Power (W)")
        ax_p.set_xlabel("Time (s)")

        fig_crop.tight_layout(pad=1.5)
        st.pyplot(fig_crop, use_container_width=True)
        plt.close(fig_crop)

        # ================================================================
        # TURNAROUND EDITOR  (inside Crop Editor)
        # ================================================================
        st.markdown("---")
        st.subheader("🔄 Turnaround Points")
        st.caption(
            "The algorithm detects turnaround points automatically. "
            "If they are wrong or missing, you can **adjust**, **add**, or "
            "**remove** them here, then press **Recrop** to rebuild the legs."
        )

        # ── Make sure segment + turnarounds exist ─────────────────────────
        az.set_segment(0, len(az.raw_data) - 1)

        manual_key = f"manual_turns_{crop_key}"
        if manual_key in st.session_state and st.session_state[manual_key] is not None:
            az.manual_turnaround_indices = list(st.session_state[manual_key])

        az.detect_turnarounds()
        seg_df = az.segment_data
        seg_time = seg_df['time'].values

        current_turn_indices = list(az.turnaround_indices)
        current_turn_times = [float(seg_time[ti]) for ti in current_turn_indices]

        is_manual = az.manual_turnaround_indices is not None
        st.info(
            f"{'🖐️ **Manual** turnarounds active' if is_manual else '🤖 **Auto-detected** turnarounds'}"
            f" — {len(current_turn_indices)} turnaround(s)"
        )

        # ── Store editable times in session state list ────────────────────
        edit_key = f"turn_times_{crop_key}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = list(current_turn_times)
        turn_times: list = st.session_state[edit_key]

        # Sync: if auto-detect changed (e.g. after crop) and no manual
        # override exists, refresh the list
        if not is_manual and turn_times != current_turn_times:
            turn_times = list(current_turn_times)
            st.session_state[edit_key] = turn_times

        # ── Render one number_input per turnaround ────────────────────────
        if turn_times:
            st.markdown("**Turnaround times** — adjust values directly:")
            remove_idx = None  # track which one to delete

            for i, tt in enumerate(turn_times):
                c1, c2 = st.columns([5, 1])
                with c1:
                    new_val = st.number_input(
                        f"Turnaround {i + 1}",
                        min_value=0.0,
                        max_value=float(seg_time[-1]),
                        value=float(tt),
                        step=0.5,
                        format="%.1f",
                        key=f"turn_inp_{crop_key}_{i}",
                    )
                    turn_times[i] = new_val  # update in-place
                with c2:
                    st.markdown("")  # vertical spacing
                    st.markdown("")
                    if st.button("🗑️", key=f"del_turn_{crop_key}_{i}",
                                 help=f"Remove turnaround {i + 1}"):
                        remove_idx = i

            if remove_idx is not None:
                turn_times.pop(remove_idx)
                st.session_state[edit_key] = turn_times
                st.rerun()
        else:
            st.warning("No turnarounds. Add one below ⬇️")

        # ── Add turnaround ────────────────────────────────────────────────
        st.markdown("")
        add_c1, add_c2, add_c3 = st.columns([2, 1, 1])

        with add_c1:
            new_turn_time = st.number_input(
                "New turnaround time (s)",
                min_value=0.0,
                max_value=float(seg_time[-1]),
                value=float(seg_time[-1]) / 2,
                step=0.5,
                format="%.1f",
                key=f"new_turn_time_{crop_key}",
            )
        with add_c2:
            st.markdown("")
            st.markdown("")
            if st.button("➕ Add", use_container_width=True,
                         key=f"add_turn_{crop_key}"):
                turn_times.append(new_turn_time)
                turn_times.sort()
                st.session_state[edit_key] = turn_times
                st.rerun()

        with add_c3:
            st.markdown("")
            st.markdown("")
            if st.button("🤖 Reset to Auto", use_container_width=True,
                         key=f"reset_turns_{crop_key}"):
                st.session_state.pop(manual_key, None)
                st.session_state.pop(edit_key, None)
                az.manual_turnaround_indices = None
                st.rerun()

        # ── RECROP button ─────────────────────────────────────────────────
        st.markdown("")
        if st.button("🔄 Recrop with These Turnarounds", type="primary",
                     use_container_width=True,
                     key=f"recrop_btn_{crop_key}"):

            if not turn_times:
                st.error("❌ Add at least one turnaround point.")
            else:
                # Times → sorted, unique indices
                new_indices = sorted(set(
                    int(np.clip(np.searchsorted(seg_time, t),
                                0, len(seg_time) - 1))
                    for t in turn_times
                ))

                # Persist as manual override
                st.session_state[manual_key] = new_indices
                az.manual_turnaround_indices = new_indices
                az.turnaround_indices = new_indices

                # Rebuild legs
                az.build_legs()

                # Clear stale CdA
                rec['cda'] = None
                rec['cda_params'] = None

                # Re-run quality (save + restore manual turns around the
                # call because check_data_quality calls set_segment which
                # clears manual_turnaround_indices)
                saved_manual = list(new_indices)
                qr = check_data_quality(
                    az, mass=mass, crr=crr, rho=rho, wind_ms=wind_ms)
                rec['quality'] = qr
                az.manual_turnaround_indices = saved_manual
                st.session_state[manual_key] = saved_manual

                st.success(
                    f"✅ Recropped with {len(new_indices)} turnaround(s) "
                    f"→ {len(az.legs)} legs, {len(az.pairs)} pairs")
                st.rerun()

        # ── Live preview plot ─────────────────────────────────────────────
        if turn_times:
            # Temporarily apply the current times so we can preview legs
            preview_indices = sorted(set(
                int(np.clip(np.searchsorted(seg_time, t),
                            0, len(seg_time) - 1))
                for t in turn_times
            ))
            az.turnaround_indices = preview_indices
            az.build_legs()

            if az.legs:
                st.markdown(
                    f"**Preview:** {len(az.legs)} legs, "
                    f"{len(az.pairs)} pair(s)")

                from plots.style import LEG_PALETTE, TURN_CLR, DARK_BG

                fig_ta, (ax_ta1, ax_ta2) = plt.subplots(
                    2, 1, figsize=(14, 5), sharex=True, facecolor=PANEL_BG)
                style_ax(ax_ta1);
                style_ax(ax_ta2)

                ax_ta1.plot(seg_time, seg_df['speed'].values * 3.6,
                            color='#444', lw=0.8)
                ax_ta2.plot(seg_time, seg_df['power'].values,
                            color='#444', lw=0.8)

                for k, leg in enumerate(az.legs):
                    col = LEG_PALETTE[k % len(LEG_PALETTE)]
                    tag = 'out' if k % 2 == 0 else 'back'
                    ax_ta1.plot(leg['time'].values,
                                leg['speed'].values * 3.6,
                                color=col, lw=1.8,
                                label=f'Leg {k + 1} ({tag})')
                    ax_ta2.plot(leg['time'].values,
                                leg['power'].values,
                                color=col, lw=1.8)

                for ti in preview_indices:
                    t_val = float(seg_time[ti])
                    ax_ta1.axvline(t_val, color=TURN_CLR, lw=2, ls='--')
                    ax_ta2.axvline(t_val, color=TURN_CLR, lw=2, ls='--')

                for i in range(len(az.legs) - 1):
                    gap_s = float(az.legs[i]['time'].iloc[-1])
                    gap_e = float(az.legs[i + 1]['time'].iloc[0])
                    if gap_e > gap_s:
                        ax_ta1.axvspan(gap_s, gap_e,
                                       color=TURN_CLR, alpha=0.15)
                        ax_ta2.axvspan(gap_s, gap_e,
                                       color=TURN_CLR, alpha=0.15)

                ax_ta1.set_ylabel("Speed (km/h)")
                ax_ta1.set_title(
                    "Turnaround Preview — dashed red = turnarounds, "
                    "shaded red = cropped zones",
                    color="white", fontweight="bold", fontsize=10)
                ax_ta1.legend(fontsize=7, facecolor=PANEL_BG,
                              labelcolor='white', ncol=4, loc='lower right')
                ax_ta1.grid(True, alpha=0.12)
                ax_ta2.set_ylabel("Power (W)")
                ax_ta2.set_xlabel("Time (s)")
                ax_ta2.grid(True, alpha=0.12)
                fig_ta.tight_layout(pad=1.5)
                st.pyplot(fig_ta, use_container_width=True)
                plt.close(fig_ta)

        st.markdown("---")

        # ── Action buttons ────────────────────────────────────────────────
        bc1, bc2, bc3 = st.columns([1, 1, 3])

        if bc1.button("💾 Save Crop", type="primary",
                      use_container_width=True):
            if kept < 100:
                st.error(
                    "❌ Selection too short — keep at least 100 data points.")
            else:
                with st.spinner(
                        "Applying crop and re-running quality checks…"):
                    # Crop the raw data
                    cropped_df = (df.iloc[idx_start:idx_end + 1]
                                  .copy()
                                  .reset_index(drop=True))

                    # Recalculate time from zero
                    cropped_df['time'] = (
                        (cropped_df['timestamp']
                         - cropped_df['timestamp'].iloc[0])
                        .dt.total_seconds()
                    )

                    # Recalculate cumulative distance from zero
                    if 'distance' in cropped_df.columns:
                        d0 = cropped_df['distance'].iloc[0]
                        cropped_df['distance'] = cropped_df['distance'] - d0

                    # Apply to analyzer
                    az.raw_data = cropped_df

                    # ── Remap manual turnarounds to cropped timeline ──
                    _mk = f"manual_turns_{crop_key}"
                    _ek = f"turn_times_{crop_key}"

                    old_times = st.session_state.get(_ek)
                    if old_times:
                        # Keep turnarounds strictly inside the kept window,
                        # then shift so the cropped file starts at t = 0
                        new_times = sorted(
                            t - t_min
                            for t in old_times
                            if t_min < t < t_max
                        )
                        if new_times:
                            new_seg_time = cropped_df['time'].values
                            new_indices = [
                                int(np.clip(
                                    np.searchsorted(new_seg_time, t),
                                    0, len(new_seg_time) - 1
                                ))
                                for t in new_times
                            ]
                            st.session_state[_ek] = new_times
                            st.session_state[_mk] = new_indices
                            az.manual_turnaround_indices = new_indices
                        else:
                            # All turnarounds fell outside the crop
                            st.session_state.pop(_mk, None)
                            st.session_state.pop(_ek, None)
                            az.manual_turnaround_indices = None
                    # ──────────────────────────────────────────────────

                    # Re-run quality checks (save/restore manual turns
                    # because check_data_quality calls set_segment which
                    # clears manual_turnaround_indices)
                    saved_manual = getattr(az, 'manual_turnaround_indices', None)
                    qr = check_data_quality(
                        az,
                        mass=mass, crr=crr, rho=rho, wind_ms=wind_ms,
                    )
                    az.manual_turnaround_indices = saved_manual

                    # Update registry — clear old CdA results
                    rec['quality'] = qr
                    rec['cda'] = None
                    rec['cda_params'] = None

                    # Close crop editor
                    st.session_state['crop_key'] = None
                    st.session_state['active_panel'] = None

                st.success(
                    f"✅ Cropped to {kept:,} points — quality re-checked.")
                st.rerun()

        if bc2.button("❌ Cancel", use_container_width=True):
            st.session_state['crop_key'] = None
            st.session_state['active_panel'] = None
            st.rerun()


    # =========================================================================
    # CALCULATE CdA  (1 file selected)
    # =========================================================================
    if btn_calc and n_selected == 1 and st.session_state.get('active_panel') == 'calc':
        fkey = selected_keys[0]  # ← fkey must stay a STRING
        rec = registry[fkey]
        az = rec['analyzer']
        qr = rec['quality']

        if qr['overall'] == 'FAIL':
            st.error("❌ Quality check FAILED — fix the data before calculating CdA.")
        else:
            if qr['overall'] == 'WARNING':
                st.warning("⚠️ Quality WARNING — proceed carefully.")

            with st.spinner("Calculating CdA…"):
                try:
                    az.set_segment(0, len(az.raw_data) - 1)

                    # ── restore manual turnarounds ──
                    _mk = f"manual_turns_{fkey}"
                    if _mk in st.session_state and st.session_state[_mk] is not None:
                        az.manual_turnaround_indices = list(st.session_state[_mk])
                    # ────────────────────────────────

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
    if btn_qreport and n_selected == 1 and st.session_state.get('active_panel') == 'quality':
        fkey = selected_keys[0]
        rec  = registry[fkey]
        qr   = rec['quality']

        st.divider()
        st.subheader(f"🔍 Quality Report — {rec['name']}")
        st.caption("Each check verifies a condition required for reliable "
                   "CdA measurement. Fix any ❌ FAIL items before calculating.")

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
    if btn_report and n_selected == 1 and st.session_state.get('active_panel') == 'report':
        if registry[selected_keys[0]]['cda'] is None:
            st.warning("⚠️ Calculate CdA for this file first.")
            st.session_state['active_panel'] = None
        else:
            st.session_state['report_key'] = selected_keys[0]

    if (st.session_state.get('report_key')
            and st.session_state['report_key'] in registry
            and st.session_state.get('active_panel') == 'report'):
        fkey = st.session_state['report_key']  # ← this is a STRING
        rec = registry[fkey]

        if rec['cda'] is None:
            st.session_state['report_key'] = None
        else:
            az = rec['analyzer']
            p = rec['cda_params']
            qr = rec['quality']
            cda = rec['cda']

            az.set_segment(0, len(az.raw_data) - 1)
            az.cda_result = cda

            # ── restore manual turnarounds ──
            _mk = f"manual_turns_{fkey}"
            if _mk in st.session_state and st.session_state[_mk] is not None:
                az.manual_turnaround_indices = list(st.session_state[_mk])
            # ────────────────────────────────

            az.detect_turnarounds()
            az.build_legs()

            az.leg_closures = []
            az.pair_cda_values = []
            for o, b in az.pairs:
                d_o = az._leg_delta(o, cda, p['crr'], p['mass'], p['rho'], +p['wind_ms'])
                d_b = az._leg_delta(b, cda, p['crr'], p['mass'], p['rho'], -p['wind_ms'])
                az.leg_closures.append(d_o + d_b)

                # Per-pair CdA
                def _single_pair_obj(cda_test, _o=o, _b=b):
                    return abs(
                        az._leg_delta(_o, cda_test, p['crr'], p['mass'], p['rho'], +p['wind_ms']) +
                        az._leg_delta(_b, cda_test, p['crr'], p['mass'], p['rho'], -p['wind_ms'])
                    )

                sg = np.linspace(0.10, 0.70, 400)
                se = [_single_pair_obj(c) for c in sg]
                sb = sg[np.argmin(se)]
                sf = np.linspace(max(0.05, sb - 0.05), min(0.80, sb + 0.05), 400)
                sf_e = [_single_pair_obj(c) for c in sf]
                az.pair_cda_values.append(float(sf[np.argmin(sf_e)]))

            az.cda_mean = float(np.mean(az.pair_cda_values))
            az.cda_std = float(np.std(az.pair_cda_values, ddof=1)) if len(az.pair_cda_values) > 1 else 0.0

            st.divider()
            rpt_col1, rpt_col2 = st.columns([5, 1])
            rpt_col1.subheader(f"📊 Analysis Report — {rec['name']}")
            if rpt_col2.button("✕ Close", key="close_report"):
                st.session_state['report_key'] = None
                st.session_state['active_panel'] = None
                st.rerun()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CdA", f"{cda:.4f} m²",
                      help="Optimised drag area from the Chung Virtual "
                           "Elevation method. Lower = more aerodynamic.")
            if az.cda_std is not None and az.cda_std > 0:
                c2.metric("Confidence", f"± {az.cda_std:.4f} m²",
                          help="Standard deviation across individual pair "
                               "CdA values. Below ±0.005 = excellent. "
                               "Above ±0.015 = conditions may have varied.")
            else:
                c2.metric("Confidence", "— (need ≥2 pairs)",
                          help="Need at least 2 out-and-back pairs to "
                               "calculate a confidence interval.")
            c3.metric("Pairs", len(az.pairs),
                      help="Number of out-and-back pairs used. "
                           "3+ pairs recommended for reliable results.")
            c4.metric("Quality", f"{qr['score']:.0f}%  {_quality_badge(qr['overall'])}",
                      help="Data quality score from the automated checks. "
                           "100% = all checks passed.")

            # VE closures + per-pair CdA
            st.markdown("**Per-Pair Breakdown**")
            pair_rows = []
            for k, lc in enumerate(az.leg_closures):
                status = '✅ Excellent' if abs(lc) < 0.01 else (
                    '⚠️ Acceptable' if abs(lc) < 0.05 else '❌ Poor')
                pair_rows.append({
                    'Pair': f"Pair {k + 1}",
                    'CdA (m²)': f"{az.pair_cda_values[k]:.4f}" if k < len(az.pair_cda_values) else "—",
                    'VE Closure': f"{lc:+.4f} m",
                    'Status': status,
                })

            st.dataframe(
                pd.DataFrame(pair_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Pair": st.column_config.TextColumn("Pair", width="small"),
                    "CdA (m²)": st.column_config.TextColumn("CdA (m²)", width="small"),
                    "VE Closure": st.column_config.TextColumn("VE Closure", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="medium"),
                },
            )

            if len(az.pair_cda_values) >= 2:
                spread = max(az.pair_cda_values) - min(az.pair_cda_values)
                if spread < 0.005:
                    st.success(f"🎯 Excellent consistency — spread: {spread * 10000:.1f} cm²")
                elif spread < 0.015:
                    st.info(f"👍 Good consistency — spread: {spread * 10000:.1f} cm²")
                else:
                    st.warning(f"⚠️ High spread: {spread * 10000:.1f} cm² — conditions may have changed between pairs")

                # ── Save to History ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("**💾 Save to Trend History**")
                h_col1, h_col2 = st.columns([1, 2])
                label = h_col1.text_input(
                    "Label",
                    key="history_label",
                    placeholder="e.g. Baseline, New helmet, Skinsuit v2",
                    help="A short name for this configuration. "
                         "This appears on the trend chart X-axis.")
                notes = h_col2.text_input(
                    "Notes (optional)",
                    key="history_notes",
                    placeholder="Weather, equipment details...",
                    help="Any extra context: wind conditions, temperature, "
                         "equipment used, etc.")

                if st.button("💾 Save Result to History", type="primary"):
                    add_entry(
                        file_name=rec['name'],
                        cda=cda,
                        cda_std=az.cda_std,
                        pair_cda_values=list(az.pair_cda_values),
                        params=p,
                        quality_score=qr['score'],
                        n_pairs=len(az.pairs),
                        label=label,
                        notes=notes,
                    )
                    st.success(f"✅ Saved to history: **{label or rec['name']}**")




            # Plots in expander (optional, not forced on user)
            with st.expander("📈 Analysis plots", expanded=True):
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
    if btn_compare and n_selected == 2 and st.session_state.get('active_panel') == 'compare':
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
                "Reference TT power (W) — your expected 40 km effort",
                min_value=100, max_value=600,
                value=int(avg_pwr), step=5,
                help="Enter the average power you'd hold for a 40 km TT. "
                     "This is used to estimate time savings. "
                     "The default is the average power from both files."
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

            # ── Visual delta banner ───────────────────────────────────────
            delta_cm2 = (cda_b - cda_a) * 10000
            if delta_cm2 < 0:
                bar_colour = "#00e676"
                direction = "lower (better) ✅"
            elif delta_cm2 > 0:
                bar_colour = "#ff5252"
                direction = "higher (worse) ❌"
            else:
                bar_colour = "#aaaaaa"
                direction = "identical"

            st.markdown(f"""
                        <div style="background: #333; border-radius: 8px; padding: 16px;
                                    text-align: center; margin: 12px 0;">
                            <span style="font-size: 2.2rem; font-weight: bold; color: {bar_colour};">
                                {delta_cm2:+.1f} cm²
                            </span>
                            <br>
                            <span style="color: #aaa;">B is {direction} than A</span>
                        </div>
                        """, unsafe_allow_html=True)


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

            # ── Plots in expander (side by side) ─────────────────────────────
            with st.expander("📈 View plots for both files", expanded=True):
                # Restore analyzer state before plotting
                for _rec, _cda_val in [(rec_a, cda_a), (rec_b, cda_b)]:
                    _az = _rec['analyzer']
                    _az.set_segment(0, len(_az.raw_data) - 1)
                    _az.cda_result = _cda_val

                    # ── restore manual turnarounds ──
                    _fk = [k for k, v in registry.items() if v is _rec]
                    if _fk:
                        _mk = f"manual_turns_{_fk[0]}"
                        if _mk in st.session_state and st.session_state[_mk] is not None:
                            _az.manual_turnaround_indices = list(st.session_state[_mk])
                    # ────────────────────────────────

                    _az.detect_turnarounds()
                    _az.build_legs()

                plot_col_a, plot_col_b = st.columns(2)
                with plot_col_a:
                    st.markdown(f"**A — {rec_a['name']}**")
                    p_a = rec_a['cda_params']
                    fig_a = make_results_plot(
                        rec_a['analyzer'], p_a['crr'], p_a['rho'],
                        p_a['mass'], p_a['wind_ms'])
                    st.pyplot(fig_a, use_container_width=True)
                    plt.close(fig_a)
                with plot_col_b:
                    st.markdown(f"**B — {rec_b['name']}**")
                    p_b = rec_b['cda_params']
                    fig_b = make_results_plot(
                        rec_b['analyzer'], p_b['crr'], p_b['rho'],
                        p_b['mass'], p_b['wind_ms'])
                    st.pyplot(fig_b, use_container_width=True)
                    plt.close(fig_b)

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

    # =========================================================================
    # TREND HISTORY
    # =========================================================================
    st.markdown("---")
    st.subheader("📈 CdA Trend History")

    history = get_history()

    if not history:
        st.info("No saved results yet. Analyse a file and click "
                "**Save to History** in the Analysis Report to start tracking.")
    else:
        # ── Quick stats (always visible) ──────────────────────────────────
        if len(history) >= 2:
            best = min(history, key=lambda e: e['cda'])
            worst = max(history, key=lambda e: e['cda'])
            latest = history[-1]
            delta_vs_best = (latest['cda'] - best['cda']) * 10000
            total_range = abs(best['cda'] - worst['cda']) * 10000

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Latest", f"{latest['cda']:.4f} m²",
                      delta=f"{delta_vs_best:+.0f} cm² vs best",
                      delta_color="inverse",
                      help=latest.get('label', latest['file_name']))
            s2.metric("Best ever", f"{best['cda']:.4f} m²",
                      help=best.get('label', best['file_name']))
            s3.metric("Worst", f"{worst['cda']:.4f} m²",
                      help=worst.get('label', worst['file_name']))
            s4.metric("Total range", f"{total_range:.0f} cm²",
                      help="Difference between best and worst CdA")

        # ── Trend chart (always visible if ≥ 2 entries) ───────────────────
        if len(history) >= 2:
            fig_trend, ax_trend = plt.subplots(figsize=(12, 4),
                                               facecolor=PANEL_BG)
            style_ax(ax_trend)

            cda_vals = [e['cda'] for e in history]
            labels = [e.get('label', '') or e['file_name'] for e in history]
            x_pos = list(range(len(history)))
            yerr = [e.get('cda_std') or 0 for e in history]

            ax_trend.errorbar(x_pos, cda_vals, yerr=yerr,
                              fmt='o-', color='#4fc3f7', lw=2,
                              markersize=8, capsize=5, capthick=1.5,
                              ecolor='#90caf9',
                              markerfacecolor='#4fc3f7')

            ax_trend.axhline(cda_vals[0], color='#666', ls='--', lw=1,
                             label=f"Baseline: {cda_vals[0]:.4f}")

            for i, (x, cda_v) in enumerate(zip(x_pos, cda_vals)):
                delta_cm2 = (cda_v - cda_vals[0]) * 10000
                color = '#00e676' if delta_cm2 < 0 else (
                    '#ff5252' if delta_cm2 > 0 else '#aaa')
                if i > 0:
                    ax_trend.annotate(
                        f"{delta_cm2:+.0f} cm²",
                        xy=(x, cda_v), xytext=(0, 15),
                        textcoords='offset points', ha='center',
                        color=color, fontsize=9, fontweight='bold',
                    )

            ax_trend.set_xticks(x_pos)
            ax_trend.set_xticklabels(labels, rotation=30, ha='right',
                                     fontsize=8)
            ax_trend.set_ylabel("CdA (m²)")
            ax_trend.set_title("CdA Trend — lower is better",
                               color="white", fontweight="bold")
            ax_trend.legend(fontsize=8, facecolor=PANEL_BG,
                            labelcolor='white')
            ax_trend.grid(True, alpha=0.15)
            fig_trend.tight_layout(pad=2)

            st.pyplot(fig_trend, use_container_width=True)
            plt.close(fig_trend)

        # ── Detail table + management (collapsible) ───────────────────────
        with st.expander("📋 Full history table & management", expanded=False):
            trend_rows = []
            for idx, e in enumerate(history):
                trend_rows.append({
                    'Order': idx + 1,
                    'ID': e['id'],
                    'Date': e['timestamp'][:16].replace('T', '  '),
                    'Label': e.get('label', '') or '—',
                    'File': e['file_name'],
                    'CdA': f"{e['cda']:.4f}",
                    '±': f"{e['cda_std']:.4f}" if e.get('cda_std') else "—",
                    'Pairs': e['n_pairs'],
                    'Quality': f"{e['quality_score']:.0f}%",
                    'Params': f"{e['mass']}kg / {e['crr']:.4f} / {e['rho']:.3f}",
                    'Notes': e.get('notes', ''),
                })

            trend_df = pd.DataFrame(trend_rows)

            edited_trend = st.data_editor(
                trend_df.drop(columns=['ID']),
                column_config={
                    "Order": st.column_config.NumberColumn(
                        "Order", min_value=1, max_value=len(history), step=1,
                        help="Change numbers to reorder, then click Save Order"),
                },
                disabled=["Date", "Label", "File", "CdA", "±", "Pairs",
                          "Quality", "Params", "Notes"],
                hide_index=True,
                use_container_width=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])

            if btn_col1.button("💾 Save Order", use_container_width=True):
                new_order_vals = edited_trend['Order'].values
                id_order_pairs = list(zip(trend_df['ID'].tolist(), new_order_vals))
                id_order_pairs.sort(key=lambda x: x[1])
                new_id_order = [pair[0] for pair in id_order_pairs]
                reorder_history(new_id_order)
                st.success("✅ Order saved!")
                st.rerun()

            if btn_col2.button("🗑️ Clear All History", use_container_width=True):
                clear_history()
                st.rerun()

if __name__ == "__main__":
    main()