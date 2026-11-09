"""Public workflow boundaries and real ResNet-18 artifact integration (no live LLM calls)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from weightcode.application.bundles import (compatibility, copy_model, dataset, model_bundle, pack_program,
                                           prepare_task, program_bundle, read, sha)
from weightcode.application.data import prepare_data
from weightcode.application.editor import EditorTools
from weightcode.application.resnet import import_checkpoint, load_buffers
from weightcode.application.runtime import Application, apply_program, prepare_run
from weightcode.common.identity import params_sha256

SANDBOX = Path(__file__).resolve().parents[1] / ".venv-sandbox/bin/python"
PROGRAM = '''import numpy as np
HYPOTHESIS = "Integration fixture"
PREDICTED_EFFECT = "Predict class one"
def update(student, experience):
    return {"fc.bias": np.array([0., 2.])}
'''


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from torchvision.models import resnet18
    torch.set_num_threads(2)
    torch.manual_seed(120)
    root = tmp_path_factory.mktemp("application-model")
    module = resnet18(weights=None, num_classes=2)
    with torch.no_grad():
        module.fc.weight.zero_()
        module.fc.bias.copy_(torch.tensor([1., 0.]))
        module.bn1.num_batches_tracked.fill_(17)
    torch.save(module.state_dict(), root / "source.pt")
    import_checkpoint(root / "source.pt", ["zero", "one"], root / "model")
    return root / "model"


def make_task(root, model, *, selection_old=1, final_label=1):
    paths = []
    for index, name in enumerate(("experience", "selection", "final")):
        rng = np.random.default_rng(730 + index)
        ys = [1, 1] if name == "experience" else [1, selection_old] if name == "selection" else [final_label] * 2
        data = {f"x_{role}": rng.integers(0, 256, (1, 3, 224, 224), dtype=np.uint8) for role in ("new", "old")}
        data.update({f"y_{role}": np.array([label], dtype=np.int64) for role, label in zip(("new", "old"), ys)})
        path = root / f"{name}.npz"
        np.savez(path, **data)
        paths.append(path)
    prepare_task(model, *paths, root / "task", goal="Synthetic integration test", retention_drop=0, min_new_gain=1)
    return root / "task"


def require_sandbox():
    if not SANDBOX.is_file():
        pytest.skip("NumPy-only sandbox venv not configured")


def test_import_parity_and_buffers(model):
    manifest, params = model_bundle(model)
    assert manifest["parity"]["float64_max_rel_diff"] <= manifest["parity"]["float64_tolerance"]
    assert manifest["params_sha256"] == params_sha256(params)
    assert load_buffers(model)["bn1.num_batches_tracked"].item() == 17


def test_import_refuses_wrapped_checkpoint(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "wrapped.pt")
    with pytest.raises(ValueError, match="plain torchvision"):
        import_checkpoint(tmp_path / "wrapped.pt", ["zero", "one"], tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_import_legacy_counter_matches_torch(model, tmp_path):
    source = torch.load(model.parent / "source.pt", weights_only=True)
    del source["bn1.num_batches_tracked"]
    source["conv1.weight"].requires_grad_(True)  # Also present in official legacy weight files.
    torch.save(source, tmp_path / "legacy.pt")
    imported = import_checkpoint(tmp_path / "legacy.pt", ["zero", "one"], tmp_path / "model")
    assert imported["initialized_missing_counters"] == ["bn1.num_batches_tracked"]
    assert load_buffers(tmp_path / "model")["bn1.num_batches_tracked"].item() == 0
    assert imported["params_sha256"] == read(model / "manifest.json")["params_sha256"]


def test_model_parity_code_cannot_be_replaced(model, tmp_path):
    copy_model(model, tmp_path / "model")
    root = tmp_path / "model"
    (root / "forward.py").write_text("raise RuntimeError('untrusted')")
    m = read(root / "manifest.json")
    m["files"]["forward.py"] = sha(root / "forward.py")
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="supported adapter"):
        model_bundle(root)


def test_program_requires_same_class_order(model, tmp_path):
    source = tmp_path / "p.py"
    source.write_text(PROGRAM)
    pack_program(source, model, tmp_path / "program")
    m, params = model_bundle(model)
    m["labels"] = ["one", "zero"]
    m["compatibility"] = compatibility(params, m["labels"])
    with pytest.raises(ValueError, match="incompatible"):
        program_bundle(tmp_path / "program", m)
    (tmp_path / "program/program.py").write_text(PROGRAM + "\n# changed\n")
    with pytest.raises(ValueError, match="changed bundle"):
        program_bundle(tmp_path / "program", read(model / "manifest.json"))


def test_task_disallows_role_overlap(model, tmp_path):
    task = make_task(tmp_path, model)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_task(model, task / "experience.npz", task / "experience.npz", task / "final.npz",
                     tmp_path / "bad-task", goal="Bad split", retention_drop=0, min_new_gain=0)
    assert not (tmp_path / "bad-task").exists()


@pytest.mark.parametrize("key,value", [("y_new", np.array([2], dtype=np.int64)),
                                      ("y_old", np.array([1.0])),
                                      ("x_new", np.zeros((1, 3, 32, 32), dtype=np.uint8))])
def test_dataset_contract(tmp_path, key, value):
    data = {f"x_{r}": np.zeros((1, 3, 224, 224), dtype=np.uint8) for r in ("new", "old")}
    data.update({f"y_{r}": np.ones(1, dtype=np.int64) for r in ("new", "old")})
    data[key] = value
    np.savez(tmp_path / "bad.npz", **data)
    with pytest.raises(ValueError):
        dataset(tmp_path / "bad.npz", 2)


def test_image_preprocessing_matches_reference(model, tmp_path):
    from PIL import Image
    from torchvision.models import ResNet18_Weights
    image = Image.fromarray(np.random.default_rng(51).integers(0, 256, (260, 310, 3), dtype=np.uint8))
    for role in ("new", "old"):
        folder = tmp_path / role / "one"
        folder.mkdir(parents=True)
        image.save(folder / "x.png")
    prepare_data(model, tmp_path / "new", tmp_path / "old", tmp_path / "images.npz")
    data = dataset(tmp_path / "images.npz", 2)
    cropped = torch.from_numpy(data["x_new"][0]).to(torch.float32) / 255.0
    normalized = (cropped - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    assert torch.equal(normalized, ResNet18_Weights.IMAGENET1K_V1.transforms()(image))
    before = sha(tmp_path / "images.npz")
    with pytest.raises(FileExistsError):
        prepare_data(model, tmp_path / "new", tmp_path / "old", tmp_path / "images.npz")
    assert sha(tmp_path / "images.npz") == before


def test_invalid_check_consumes_budget_across_reload(model, tmp_path):
    require_sandbox()
    task = make_task(tmp_path, model)
    prepare_run(task, tmp_path / "run", SANDBOX, checks=1, cells=1, wall_s=120)
    app = Application(tmp_path / "run")
    (app.paths.patches / "invalid.py").write_text("bad syntax (")
    result = app.check("invalid.py")
    assert result["status"] == "rejected"
    reloaded = Application(app.paths.root)
    with pytest.raises(ValueError, match="budget exhausted"):
        reloaded.check("invalid.py")
    assert reloaded.status()["checks_used"] == 1
    tools = EditorTools(reloaded)
    assert tools.call("check", {"program_file": "x.py", "extra": 1}).is_error


def test_sandbox_paths_do_not_allow_symlink_program(model, tmp_path):
    require_sandbox()
    task = make_task(tmp_path, model)
    prepare_run(task, tmp_path / "run", SANDBOX, checks=1, cells=1, wall_s=120)
    app = Application(tmp_path / "run")
    source = tmp_path / "outside.py"
    source.write_text(PROGRAM)
    (app.paths.patches / "link.py").symlink_to(source)
    with pytest.raises(ValueError, match="regular file"):
        app.check("link.py")
    assert app.status()["checks_used"] == 0


@pytest.mark.parametrize("selection_old,final_label,status", [(1, 0, "final-rejected"), (0, 1, "base-kept")])
def test_rejected_update_releases_original(model, tmp_path, selection_old, final_label, status):
    require_sandbox()
    task = make_task(tmp_path, model, selection_old=selection_old, final_label=final_label)
    source = tmp_path / "program.py"
    source.write_text(PROGRAM)
    pack_program(source, model, tmp_path / "program")
    report = apply_program(task, tmp_path / "program", tmp_path / "run", SANDBOX, wall_s=120)
    assert report["status"] == status
    assert not report["output_is_updated_model"]
    assert report["export"]["params_sha256"] == read(model / "manifest.json")["params_sha256"]
    state = torch.load(tmp_path / "run/final/model.pt", weights_only=True)
    assert state["bn1.num_batches_tracked"].item() == 17
    app = Application(tmp_path / "run")
    with pytest.raises(ValueError, match="frozen"):
        app.check("program.py")
    before = sha(tmp_path / "run/final/model.pt")
    with pytest.raises(FileExistsError):
        app.finalize()
    assert sha(tmp_path / "run/final/model.pt") == before
