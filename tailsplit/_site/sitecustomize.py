"""Loaded at interpreter start by the servers this tool launches (their PYTHONPATH starts with
this directory). Does nothing unless TAILSPLIT_MODE is 'baseline' or 'patched'."""
import os

if os.environ.get('TAILSPLIT_MODE') in ('baseline', 'patched'):
    from tailsplit.loader import install
    install()
