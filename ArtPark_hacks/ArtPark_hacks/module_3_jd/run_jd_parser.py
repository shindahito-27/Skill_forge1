from __future__ import annotations

import argparse
import json
from pathlib import Path

from main_extraction import parse_job_description


DEFAULT_INPUT = Path("ArtPark_hacks/module_3_jd/Machine-Learning-Engineer.pdf")
DEFAULT_TXT_OUT = Path(__file__).resolve().parent / "jd_resulting_text.txt"
DEFAULT_JSON_OUT = Path(__file__).resolve().parent / "jd_parsed_output.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command JD parser wrapper.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to JD file (pdf/txt/docx-supported by parser).")
    parser.add_argument("--txt-out", default=str(DEFAULT_TXT_OUT), help="Path for resulting JD text output.")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="Path for parsed JD JSON output.")
    args = parser.parse_args()

    jd_path = Path(args.input)
    txt_out = Path(args.txt_out)
    json_out = Path(args.json_out)

    parsed = parse_job_description(str(jd_path))

    txt_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    txt_out.write_text(parsed.get("resulting_text", ""), encoding="utf-8")
    json_out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"JD resulting text written to: {txt_out}")
    print(f"JD parsed JSON written to: {json_out}")


if __name__ == "__main__":
    main()
