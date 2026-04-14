# utils/history.py

import json
import os
from datetime import datetime

HISTORY_FILE = "cda_history.json"


def _load_history():
    """Load history from JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_history(history):
    """Save history to JSON file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def add_entry(file_name, cda, cda_std, pair_cda_values, params, quality_score,
              n_pairs, label="", notes=""):
    """Add a CdA result to the history."""
    history = _load_history()
    entry = {
        'id':               len(history) + 1,
        'timestamp':        datetime.now().isoformat(),
        'file_name':        file_name,
        'label':            label,
        'notes':            notes,
        'cda':              round(cda, 5),
        'cda_std':          round(cda_std, 5) if cda_std else None,
        'pair_cda_values':  [round(v, 5) for v in pair_cda_values],
        'n_pairs':          n_pairs,
        'quality_score':    round(quality_score, 1),
        'mass':             params.get('mass'),
        'crr':              params.get('crr'),
        'rho':              params.get('rho'),
        'wind_ms':          params.get('wind_ms'),
    }
    history.append(entry)
    _save_history(history)
    return entry


def get_history():
    """Return all history entries."""
    return _load_history()


def delete_entry(entry_id):
    """Delete an entry by ID."""
    history = _load_history()
    history = [e for e in history if e.get('id') != entry_id]
    # Re-number IDs
    for i, e in enumerate(history):
        e['id'] = i + 1
    _save_history(history)


def clear_history():
    """Delete all history."""
    _save_history([])

def reorder_history(new_order):
    """
    Reorder history entries.
    new_order: list of entry IDs in the desired display order.
    """
    history = _load_history()
    id_map = {e['id']: e for e in history}
    reordered = []
    for oid in new_order:
        if oid in id_map:
            reordered.append(id_map[oid])
    # Append any entries not in new_order (safety net)
    seen = set(new_order)
    for e in history:
        if e['id'] not in seen:
            reordered.append(e)
    # Re-number IDs
    for i, e in enumerate(reordered):
        e['id'] = i + 1
    _save_history(reordered)