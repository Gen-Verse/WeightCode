"""Example reusable analytic head correction. Not a reported paper program."""

import numpy as np

HYPOTHESIS = "A regularized head correction can fit new examples while anchoring logits on old examples."
PREDICTED_EFFECT = "Improve the requested class decisions with a bounded head edit; validate retention separately."


def update(student, experience):
    params = student["params"]
    x = np.concatenate([experience["x_new"], experience["x_old"]])
    # forward.py exports features; access the already loaded adapter without relying on sys.path.
    features = student["forward"].__globals__["features"]
    f = features(params, x)
    f = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    weight = np.concatenate([params["fc.weight"].astype(np.float64),
                             params["fc.bias"][:, None].astype(np.float64)], axis=1)
    logits = f @ weight.T
    n = len(experience["y_new"])
    residual = np.zeros_like(logits)
    rows = np.arange(n)
    labels = experience["y_new"]
    # Request a unit winning margin on new examples; old-example logits are anchors.
    margin = np.maximum(0.0, logits[:n].max(axis=1) - logits[rows, labels] + 1.0)
    residual[rows, labels] = margin
    gram = f @ f.T
    ridge = max(1e-6, float(np.trace(gram)) / len(f) * 0.01)
    delta = (f.T @ np.linalg.solve(gram + ridge * np.eye(len(f)), residual)).T
    # Bound the matrix's relative Frobenius change; no random search or gradients.
    limit = 0.1 * np.linalg.norm(weight[:, :-1])
    magnitude = np.linalg.norm(delta[:, :-1])
    if magnitude > limit and magnitude > 0:
        delta *= limit / magnitude
    return {"fc.weight": delta[:, :-1], "fc.bias": delta[:, -1]}
