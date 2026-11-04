"""Validate unchanged exported actor bytes and semantics; no ROS imports."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .contract import (ACTION_DIM,ACTION_SCALE,CONTROL_DT,DEFAULT_JOINT_POS,JOINT_NAMES,
 OBSERVATION_DIM,SOURCE_CHECKPOINT_SHA256,TRAINING_JOINT_NAMES,MODEL_SHA256,CONTRACT_SHA256)
def load_export_contract(model_path: Path) -> dict:
    contract_path = model_path.with_name("proprio_hold.contract.json")
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError("Contract bytes differ from frozen export")
    if model_path.suffix != ".onnx" or hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Expected the unchanged frozen ONNX actor")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {
        "schema": "team15-proprio-hold-actor-v1",
        "task_schema": "k1-proprioceptive-forearm-hold-v1",
        "source_observation_schema": "k1-imu-proprio-last-action-72-v1",
        "no_vision": True,
        "actor_uses_payload_state": False,
        "mass_observed": False,
        "actor_observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "normalization_included": True,
        "action_output_semantics": "clipped_normalized_joint_position_offset",
        "action_scale_included": False,
        "default_joint_offset_included": False,
        "hardware_ready": False,
        "control_dt_s": CONTROL_DT,
        "history_frames": 1,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"Incompatible export contract {key}: {contract.get(key)!r}")
    if contract.get("action_clip") != [-1, 1]:
        raise ValueError("Expected export action clip [-1, 1]")
    if contract.get("input") != {"name": "obs", "dtype": "float32", "shape": [1, OBSERVATION_DIM]}:
        raise ValueError("Expected a single float32[1,72] obs input")
    if contract.get("output") != {"name": "actions", "dtype": "float32", "shape": [1, ACTION_DIM]}:
        raise ValueError("Expected a single float32[1,22] actions output")
    mapping = contract.get("joint_mapping", {})
    for key, expected in (("training_joint_names", TRAINING_JOINT_NAMES),
                          ("booster_deploy_joint_names", JOINT_NAMES)):
        if mapping.get(key) != list(expected):
            raise ValueError(f"Export canonical joint mapping differs: {key}")
    for key, expected in (("default_joint_pos_rad", DEFAULT_JOINT_POS),
                          ("action_scale_rad", ACTION_SCALE)):
        values = np.asarray(mapping.get(key, []), dtype=float)
        if values.shape != (ACTION_DIM,) or not np.allclose(values, expected, rtol=0., atol=1e-12):
            raise ValueError(f"Export canonical joint mapping differs: {key}")
    hash_key = "onnx_sha256" if model_path.suffix == ".onnx" else "torchscript_sha256"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != contract.get(hash_key):
        raise ValueError("Export file SHA-256 does not match its contract")
    return contract


