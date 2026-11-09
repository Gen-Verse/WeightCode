"""Fitting references and matched random controls (contract sections 3 to 6, PREREG sections 5 and 6).

Fitting references show how far plain label fitting on the same 64 evidence images gets: ``grad_last``,
``grad_sparse`` and ``grad_full`` (module ``grad``) and ``ridge_last`` (module ``ridge``). Matched random controls
(module ``controls``) keep a reference patch's per-tensor changed count and norm but move where the edit lands.

Every method sees exactly what an editor sees (model, forward.py, model_spec.json, evidence) plus a counted
patch-check oracle with 8 queries. All candidates are generated before the first query, every candidate is written in
the shared patch format and re-applied with the patch executor, and the final patch follows contract 5.2.
"""
