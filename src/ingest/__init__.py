"""Dialygo ingest: institutional fistulography handover -> de-identified, patient-grouped frames.

Phase 1 (scan) is read-only inventory; later phases de-identify and extract frames. Every module
here is gated by src.ingest.clearance.require_clearance: nothing touches real patient data until
the institutional agreement (B5) and the IP/engagement agreement (B9) have both executed.

Modules import torch-free, cv2-free and pydicom-free at module level (repo convention); heavy
imports live inside the functions that need them.
"""
