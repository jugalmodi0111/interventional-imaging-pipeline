"""TDD for the torch-free helpers in src.train.train_seg.

The whole point: this module (and thus these tests) must import on a box with NO torch /
nnU-Net / coremltools. Every heavy import in train_seg lives inside train()/eval helpers.
"""
from src.train import train_seg as T

CFG = {
    "task": "coronary_vessel_segmentation",
    "preprocess": {"clahe": True, "size": 512},
    "teacher": {"name": "nnunetv2", "config": "2d", "preset": "ResEncM", "folds": 5},
    "student": {"name": "tiny_unet", "base_ch": 16, "depth": 4},
    "distill": {"alpha": 0.5, "temperature": 2.0, "feature_kd": True},
    "train": {"epochs": 200, "batch": 8, "lr": 1e-3, "amp": True},
    "metrics": ["dice", "cldice", "hd95"],
    "target": {"dice": 0.75},
    "export": {"onnx": True, "int8": True, "coreml": True},
}


# ---- student_kwargs ---------------------------------------------------------
def test_student_kwargs_maps_base_ch_and_depth():
    assert T.student_kwargs(CFG) == {"base": 16, "depth": 4}


def test_student_kwargs_uses_config_values():
    assert T.student_kwargs({"student": {"base_ch": 32, "depth": 5}}) == {"base": 32, "depth": 5}


def test_student_kwargs_defaults_when_missing():
    assert T.student_kwargs({}) == {"base": 16, "depth": 4}


# ---- distill_kwargs ---------------------------------------------------------
def test_distill_kwargs_maps_temperature_to_T_and_pulls_train_fields():
    assert T.distill_kwargs(CFG) == {"alpha": 0.5, "T": 2.0, "epochs": 200, "lr": 1e-3, "amp": True}


def test_distill_kwargs_drops_keys_distill_does_not_accept():
    d = T.distill_kwargs(CFG)
    for k in ("temperature", "batch", "feature_kd"):
        assert k not in d


def test_distill_kwargs_defaults_when_missing():
    assert T.distill_kwargs({}) == {"alpha": 0.5, "T": 2.0, "epochs": 200, "lr": 1e-3, "amp": True}


# ---- dataset_id_and_name ----------------------------------------------------
def test_dataset_id_and_name_matches_dca1_converter():
    # dca1_to_nnunet.py hard-codes "Dataset001_Coronary"; we must agree with it.
    assert T.dataset_id_and_name(CFG) == (1, "Dataset001_Coronary")


def test_dataset_id_and_name_is_deterministic():
    assert T.dataset_id_and_name(CFG) == T.dataset_id_and_name(CFG)


def test_dataset_id_and_name_zero_pads_id_into_name():
    did, name = T.dataset_id_and_name(CFG)
    assert isinstance(did, int) and name.startswith(f"Dataset{did:03d}_")


# ---- nnunet_train_cmd -------------------------------------------------------
def test_nnunet_train_cmd_is_pure_string_argv():
    cmd = T.nnunet_train_cmd(1, CFG)
    assert isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd)


def test_nnunet_train_cmd_shape():
    cmd = T.nnunet_train_cmd(1, CFG)
    assert cmd[0] == "nnUNetv2_train"
    assert cmd[1] == "1" and cmd[2] == "2d"            # dataset id + config


def test_nnunet_train_cmd_fold_defaults_to_zero_and_is_selectable():
    assert T.nnunet_train_cmd(1, CFG)[3] == "0"
    assert T.nnunet_train_cmd(1, CFG, fold=3)[3] == "3"


def test_nnunet_train_cmd_resenc_preset_selects_plans():
    cmd = T.nnunet_train_cmd(1, CFG)
    assert cmd[cmd.index("-p") + 1] == "nnUNetResEncUNetMPlans"


def test_nnunet_train_cmd_default_preset_omits_plans_flag():
    assert "-p" not in T.nnunet_train_cmd(1, {"teacher": {"config": "2d"}})


# ---- nnunet_predict_cmd -----------------------------------------------------
def test_nnunet_predict_cmd_is_pure_string_argv():
    cmd = T.nnunet_predict_cmd(1, "in", "out", CFG)
    assert isinstance(cmd, list) and all(isinstance(x, str) for x in cmd)
    assert cmd[0] == "nnUNetv2_predict"


def test_nnunet_predict_cmd_wires_io_and_dataset():
    cmd = T.nnunet_predict_cmd(1, "in_dir", "out_dir", CFG)
    assert cmd[cmd.index("-i") + 1] == "in_dir"
    assert cmd[cmd.index("-o") + 1] == "out_dir"
    assert cmd[cmd.index("-d") + 1] == "1"
    assert cmd[cmd.index("-c") + 1] == "2d"


def test_nnunet_predict_cmd_saves_probabilities():
    # --save_probabilities is what makes the teacher cache TeacherCacheDataset needs.
    assert "--save_probabilities" in T.nnunet_predict_cmd(1, "in", "out", CFG)


def test_nnunet_predict_cmd_lists_all_folds():
    cmd = T.nnunet_predict_cmd(1, "in", "out", CFG)
    i = cmd.index("-f")
    assert cmd[i + 1:i + 6] == ["0", "1", "2", "3", "4"]


def test_nnunet_predict_cmd_resenc_preset_selects_plans():
    cmd = T.nnunet_predict_cmd(1, "in", "out", CFG)
    assert cmd[cmd.index("-p") + 1] == "nnUNetResEncUNetMPlans"


# ---- qualifies --------------------------------------------------------------
def test_qualifies_true_above_target():
    assert T.qualifies({"dice": 0.80, "cldice": 0.7}, CFG) is True


def test_qualifies_true_on_exact_target():
    assert T.qualifies({"dice": 0.75}, CFG) is True


def test_qualifies_false_below_target():
    assert T.qualifies({"dice": 0.60}, CFG) is False


def test_import_train_seg_without_torch():
    # Guardrail: importing the driver must not drag in torch/nnU-Net/coremltools.
    import importlib, sys
    assert "torch" not in sys.modules
    importlib.import_module("src.train.train_seg")
    assert "torch" not in sys.modules
