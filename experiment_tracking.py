"""
Run provenance and experiment tracking (R1 of architecture_review.md).

Two jobs:

1. **One self-describing file per run.** Every tracking run writes a single
   ``run.json`` = ``{provenance, config, summary}`` into its output directory,
   replacing the older split ``params.json`` + ``summary.json``. ``provenance``
   records the git commit, a dirty-tree flag, a timestamp and library versions,
   so a result on disk says *which code* produced it — the gap the review named.
   ``read_run`` / ``read_summary`` / ``read_config`` read it back and fall back to
   the legacy split files, so runs already sitting in ``results/`` still load.

2. **A local MLflow mirror.** ``mlflow_log_run`` logs the same config, provenance
   and metrics (plus the run artifacts) to a local MLflow store, so runs are
   browsable/comparable in the MLflow UI. It is fully graceful: a missing
   ``mlflow`` package or an unreachable server degrades to a one-line warning and
   never breaks a run.

This module imports nothing from the pipeline (``evaluation`` imports *it*), so
the report scripts can read run records without pulling the heavy network stack.

MLflow configuration (environment variables)
--------------------------------------------
- ``MLFLOW_TRACKING_URI`` : where to log. Unset -> a repo-local SQLite backend at
  ``./mlflow.db`` (view with ``mlflow ui --backend-store-uri sqlite:///mlflow.db``).
  Set it to e.g. ``http://127.0.0.1:5000`` to log to a running ``mlflow server``.
- ``IM_MLFLOW=0``         : disable MLflow logging entirely.
- ``IM_MLFLOW_EXPERIMENT``: MLflow experiment name (default ``interacting_maps``).
"""

import os
import sys
import json
import subprocess
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

RUN_FILE = 'run.json'
_LEGACY_SUMMARY = 'summary.json'
_LEGACY_PARAMS = 'params.json'


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _git(*args):
    """Run a git command at the repo root; return stripped stdout or None."""
    try:
        out = subprocess.run(['git', '-C', ROOT, *args],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def get_provenance():
    """Describe the code and environment that produced a run.

    All fields degrade to None/False rather than raising, so a run outside a git
    checkout (or without git installed) still records a timestamp and versions.
    """
    commit = _git('rev-parse', 'HEAD')
    status = _git('status', '--porcelain')
    prov = {
        'git_commit': commit,
        'git_commit_short': commit[:7] if commit else None,
        'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        # dirty = tracked changes not committed when the run started; a dirty run
        # is not exactly reproducible from the commit alone, so flag it loudly.
        'git_dirty': bool(status) if status is not None else None,
        'timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'python': sys.version.split()[0],
    }
    for name in ('numpy', 'scipy', 'cv2', 'mlflow'):
        try:
            mod = __import__(name)
            prov[f'{name}_version'] = getattr(mod, '__version__', None)
        except Exception:
            prov[f'{name}_version'] = None
    return prov


# ---------------------------------------------------------------------------
# The merged per-run record
# ---------------------------------------------------------------------------

def save_run(output_dir, config, summary=None, provenance=None):
    """Write the merged run.json. Called once early (summary=None, so the config
    and provenance survive a crash) and again at the end with the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    record = {
        'provenance': provenance if provenance is not None else get_provenance(),
        'config': config,
        'summary': summary,
    }
    with open(os.path.join(output_dir, RUN_FILE), 'w') as f:
        json.dump(record, f, indent=2)
    return record


def read_run(output_dir):
    """Load a run record as ``{provenance, config, summary}``.

    Prefers run.json; falls back to the legacy split files so runs written before
    this change still load. Missing pieces come back as {}."""
    run_path = os.path.join(output_dir, RUN_FILE)
    if os.path.exists(run_path):
        with open(run_path) as f:
            rec = json.load(f)
        rec.setdefault('provenance', {})
        rec.setdefault('config', {})
        rec.setdefault('summary', {})
        return rec
    # Legacy fallback.
    rec = {'provenance': {}, 'config': {}, 'summary': {}}
    p = os.path.join(output_dir, _LEGACY_PARAMS)
    s = os.path.join(output_dir, _LEGACY_SUMMARY)
    if os.path.exists(p):
        with open(p) as f:
            rec['config'] = json.load(f)
    if os.path.exists(s):
        with open(s) as f:
            rec['summary'] = json.load(f)
    return rec


def read_summary(output_dir):
    """The metrics dict for a run (new run.json or legacy summary.json)."""
    return read_run(output_dir)['summary'] or {}


def read_config(output_dir):
    """The config dict for a run (new run.json or legacy params.json)."""
    return read_run(output_dir)['config'] or {}


# ---------------------------------------------------------------------------
# MLflow mirror
# ---------------------------------------------------------------------------

def _flatten_for_mlflow(config, provenance):
    """MLflow params are flat scalars; expand nested config/params and tag with
    the git provenance so runs are filterable by commit in the UI."""
    flat = {}
    for k, v in (config or {}).items():
        if k == 'params' and isinstance(v, dict):
            for pk, pv in v.items():
                flat[pk] = pv
        elif k == 'initial_R':
            flat['initial_R'] = str(v)
        elif isinstance(v, (list, tuple)):
            flat[k] = str(v)
        else:
            flat[k] = v
    for k in ('git_commit_short', 'git_branch', 'git_dirty'):
        if provenance and provenance.get(k) is not None:
            flat[k] = provenance[k]
    return flat


def mlflow_log_run(config, summary, provenance=None, output_dir=None,
                   run_name=None, artifact_names=None):
    """Mirror a run to a local MLflow store. Graceful: never raises.

    Returns the MLflow run_id on success, or None if logging was disabled,
    skipped, or failed (with a one-line warning).
    """
    if os.environ.get('IM_MLFLOW', '1') == '0':
        return None
    try:
        import mlflow
    except Exception:
        print("  [mlflow] package not available -- skipping (pip install mlflow).")
        return None

    try:
        if provenance is None:
            provenance = get_provenance()
        # Default to a repo-local SQLite backend. MLflow 3.x deprecated the bare
        # file store, and SQLite is the recommended local backend: no server
        # process needed, and `mlflow server/ui --backend-store-uri
        # sqlite:///mlflow.db` reads the very same DB. MLFLOW_TRACKING_URI (e.g. a
        # running server) overrides it and mlflow reads that env var itself.
        if not os.environ.get('MLFLOW_TRACKING_URI'):
            db = os.path.join(ROOT, 'mlflow.db').replace('\\', '/')
            mlflow.set_tracking_uri('sqlite:///' + db)
        mlflow.set_experiment(os.environ.get('IM_MLFLOW_EXPERIMENT',
                                             'interacting_maps'))

        if run_name is None and config:
            run_name = (f"{config.get('dataset')}/{config.get('model')}/"
                        f"{config.get('segment_id', config.get('segment', ''))}")

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(_flatten_for_mlflow(config, provenance))
            if provenance.get('git_commit'):
                mlflow.set_tag('git_commit', provenance['git_commit'])
            if config and config.get('out_root'):
                # 'results' = reported runs; 'experiments/...' = throwaway sweeps.
                mlflow.set_tag('out_root', config['out_root'])
            metrics = {k: float(v) for k, v in (summary or {}).items()
                       if isinstance(v, (int, float)) and v == v}  # drop NaN
            if metrics:
                mlflow.log_metrics(metrics)
            if output_dir:
                names = artifact_names or (
                    RUN_FILE, 'tracking.csv', 'tracking_plot.png')
                for name in names:
                    path = os.path.join(output_dir, name)
                    if os.path.exists(path):
                        mlflow.log_artifact(path)
            return run.info.run_id
    except Exception as e:
        print(f"  [mlflow] logging skipped ({type(e).__name__}: {e}).")
        return None
