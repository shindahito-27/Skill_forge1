import json
import importlib.util
import os
import uuid
from functools import lru_cache
from pathlib import Path

from ..utils.parser import build_structured_response

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTPARK_DIR = PROJECT_ROOT
PIPELINE_OUTPUT_DIR = ARTPARK_DIR / "output"
DEFAULT_JD_PATH = ARTPARK_DIR / "Machine-Learning-Engineer.pdf"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "backend" / "uploads"

DEFAULT_JD_CONTENT = b"%PDF-1.4\n%\xc3\xa2\xc3\xa3\xc3\x8f\xc3\x93\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 62 >>\nstream\nBT /F1 24 Tf 100 700 Td (No JD provided) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \n0000000189 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n273\n%%EOF\n"


def _write_default_jd(path: Path) -> None:
    path.write_bytes(DEFAULT_JD_CONTENT)


def _clear_previous_outputs() -> None:
    if not PIPELINE_OUTPUT_DIR.exists():
        return
    for path in PIPELINE_OUTPUT_DIR.rglob("*"):
        if path.is_file():
            path.unlink()


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Expected output file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_run_pipeline():
    pipeline_script = ARTPARK_DIR / "run_pipeline.py"
    if not pipeline_script.exists():
        raise RuntimeError(f"Pipeline script not found: {pipeline_script}")
    spec = importlib.util.spec_from_file_location("artpark_run_pipeline", pipeline_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load pipeline module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_pipeline = getattr(module, "run_pipeline", None)
    if run_pipeline is None:
        raise RuntimeError("run_pipeline function not found in run_pipeline.py")
    return run_pipeline


@lru_cache(maxsize=1)
def _get_run_pipeline():
    return _load_run_pipeline()


def analyze_resume(
    filename: str,
    file_bytes: bytes,
    jd_filename: str | None = None,
    jd_file_bytes: bytes | None = None,
) -> dict:
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = TEMP_UPLOADS_DIR / unique_name
    upload_path.write_bytes(file_bytes)
    jd_upload_path: Path | None = None
    if jd_file_bytes is not None and jd_filename:
        jd_unique_name = f"{uuid.uuid4().hex}_{jd_filename}"
        jd_upload_path = TEMP_UPLOADS_DIR / jd_unique_name
        jd_upload_path.write_bytes(jd_file_bytes)

    created_default_jd = False
    if jd_upload_path is not None:
        jd_path = jd_upload_path
    elif DEFAULT_JD_PATH.exists():
        jd_path = DEFAULT_JD_PATH
    else:
        jd_path = TEMP_UPLOADS_DIR / f"default_jd_{uuid.uuid4().hex}.pdf"
        _write_default_jd(jd_path)
        created_default_jd = True

    old_cwd = Path.cwd()
    try:
        _clear_previous_outputs()
        os.chdir(ARTPARK_DIR)
        run_pipeline = _get_run_pipeline()
        run_pipeline(str(upload_path), str(jd_path), "output")
    except Exception as exc:  # pragma: no cover - surfaced through API
        raise RuntimeError(f"Pipeline execution failed: {exc}") from exc
    finally:
        os.chdir(old_cwd)
        if upload_path.exists():
            upload_path.unlink()
        if jd_upload_path and jd_upload_path.exists():
            jd_upload_path.unlink()
        if created_default_jd and jd_path.exists():
            jd_path.unlink()

    gap_data = _read_json(PIPELINE_OUTPUT_DIR / "module_4" / "gapengine_output.json")
    mapping_data = _read_json(PIPELINE_OUTPUT_DIR / "module_5" / "profession_mapping_output.json")
    roadmap_data = _read_json(PIPELINE_OUTPUT_DIR / "module_6" / "adaptive_path_output.json")
    resources_data = _read_json(PIPELINE_OUTPUT_DIR / "module_7" / "learning_resources_output.json")
    resume_skill_data = _read_json(
        PIPELINE_OUTPUT_DIR / "resume" / "module_2" / "Module_2_combined.json"
    )

    return build_structured_response(
        filename=filename,
        gap_data=gap_data,
        mapping_data=mapping_data,
        roadmap_data=roadmap_data,
        resources_data=resources_data,
        resume_skill_data=resume_skill_data,
    )
