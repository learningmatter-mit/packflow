"""PackFlow: flow matching for molecular crystal structure prediction.

This package provides:

- ``packflow.processing`` -- mmCIF/SMILES -> training-ready crystal tensors.
- ``packflow.data``       -- datamodule / dataset / collate utilities.
- ``packflow.models``     -- the Cartesian crystal transformer + flow-matching wrapper.
- ``packflow.utils``      -- crystal/lattice/SO(3) helper functions.
- ``packflow.grpo``       -- GRPO (Group Relative Policy Optimization) post-training.
- ``packflow.relaxation`` -- UMA relaxation, lattice-energy and hydrogen-addition tools.
- ``packflow.inference``  -- high-level ``load_checkpoint`` / ``predict`` / ``run_inference`` API.

The canonical model architecture used for every paper checkpoint (base 2M/20M/60M
and the GRPO-finetuned "PA" model) is
``packflow.models.model.CrystalTransformerEncoder``, wrapped by
``CrystalFlowMatching``.

Top-level imports are guarded: if an optional dependency for a given subsystem is
not installed, the corresponding names are set to ``None`` (and the import error is
recorded in ``packflow._import_errors``) instead of breaking ``import packflow``.
"""

# Records (subsystem -> exception) for any subsystem whose optional deps are missing.
_import_errors = {}


def _safe(subsystem, importer):
    try:
        return importer()
    except Exception as exc:  # pragma: no cover - depends on installed extras
        _import_errors[subsystem] = exc
        return None


# --- Processing: raw structures -> tensors (needs gemmi, rdkit, pymatgen) ------
def _imp_processing():
    from .processing.final_crystal_processor import (
        parse_mmcif_to_crystal_dict,
        process_single_crystal,
        process_single_mmcif_file,
        process_directory,
    )
    return (parse_mmcif_to_crystal_dict, process_single_crystal,
            process_single_mmcif_file, process_directory)


_p = _safe("processing", _imp_processing)
(parse_mmcif_to_crystal_dict, process_single_crystal,
 process_single_mmcif_file, process_directory) = _p if _p else (None, None, None, None)


# --- Data ----------------------------------------------------------------------
def _imp_data():
    from .data.crystal_datamodule import CrystalDataset, CrystalDataModule, collate_fn
    return CrystalDataset, CrystalDataModule, collate_fn


_d = _safe("data", _imp_data)
CrystalDataset, CrystalDataModule, collate_fn = _d if _d else (None, None, None)


# --- Utils ---------------------------------------------------------------------
def _imp_utils():
    from .utils.batch_processor import (
        batch_process_crystals, save_processed_data, load_processed_data,
        split_processed_data, create_dataset_info,
    )
    return (batch_process_crystals, save_processed_data, load_processed_data,
            split_processed_data, create_dataset_info)


_u = _safe("utils", _imp_utils)
(batch_process_crystals, save_processed_data, load_processed_data,
 split_processed_data, create_dataset_info) = _u if _u else (None,) * 5


# --- Models (canonical = Cartesian transformer + flow matching) ----------------
def _imp_models():
    from .models.model import (
        CrystalTransformerEncoder, CrystalFlowMatching,
    )
    return CrystalTransformerEncoder, CrystalFlowMatching


_m = _safe("models", _imp_models)
CrystalTransformerEncoder, CrystalFlowMatching = _m if _m else (None, None)


# --- High-level inference API --------------------------------------------------
def _imp_inference():
    from .inference import (
        load_checkpoint, run_inference, predict, build_sampling_template,
        generate, list_models, to_structure, write_cif,
    )
    return (load_checkpoint, run_inference, predict, build_sampling_template,
            generate, list_models, to_structure, write_cif)


_i = _safe("inference", _imp_inference)
(load_checkpoint, run_inference, predict, build_sampling_template, generate,
 list_models, to_structure, write_cif) = _i if _i else (None,) * 8


# --- Checkpoint zoo (no heavy deps) --------------------------------------------
def _imp_checkpoints():
    from .checkpoints import download_checkpoints, resolve
    return download_checkpoints, resolve


_c = _safe("checkpoints", _imp_checkpoints)
download_checkpoints, resolve_checkpoint = _c if _c else (None, None)


# --- Training ------------------------------------------------------------------
def _imp_training():
    from .training import train, TrainConfig
    return train, TrainConfig


_t = _safe("training", _imp_training)
train, TrainConfig = _t if _t else (None, None)


# --- Evaluation ----------------------------------------------------------------
def _imp_evaluation():
    from .evaluation import evaluate
    return (evaluate,)


_e = _safe("evaluation", _imp_evaluation)
(evaluate,) = _e if _e else (None,)


# --- Relaxation (UMA) ----------------------------------------------------------
def _imp_relaxation():
    from .relaxation import relax_crystal
    return (relax_crystal,)


_r = _safe("relaxation", _imp_relaxation)
(relax,) = _r if _r else (None,)


__all__ = [
    # processing
    "parse_mmcif_to_crystal_dict", "process_single_crystal",
    "process_single_mmcif_file", "process_directory",
    # data
    "CrystalDataset", "CrystalDataModule", "collate_fn",
    # utils
    "batch_process_crystals", "save_processed_data", "load_processed_data",
    "split_processed_data", "create_dataset_info",
    # models
    "CrystalTransformerEncoder", "CrystalFlowMatching",
    # inference
    "load_checkpoint", "run_inference", "predict", "build_sampling_template",
    "generate", "list_models", "to_structure", "write_cif",
    # checkpoints
    "download_checkpoints", "resolve_checkpoint",
    # training / evaluation / relaxation
    "train", "TrainConfig", "evaluate", "relax",
]
