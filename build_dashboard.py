"""
Bake real analysis results into the frontend
==============================================
Reads dashboard_data.json (produced by run_all.py) and injects it into
dashboard_template.html, producing a final, self-contained dashboard.html
with real data baked directly into the page. Baking the data in (rather than
fetching the JSON at runtime) means the file can just be double-clicked and
opened in a browser -- no local server, no CORS issues with file:// URLs.

Run: python build_dashboard.py
"""

import json
import re

TEMPLATE_PATH = "dashboard_template.html"
DATA_PATH = "dashboard_data.json"
OUTPUT_PATH = "dashboard.html"


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    data["_demo"] = False  # real data -> hide the demo banner

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    marker_start = "const DATA = /*__DASHBOARD_DATA__*/"
    start_idx = template.find(marker_start)
    if start_idx == -1:
        raise RuntimeError("Could not find data marker in template. Has dashboard_template.html changed?")

    # The embedded demo object runs from just after the marker to the first
    # "\n};\n" that precedes the "// Helpers" comment block.
    object_start = start_idx + len(marker_start)
    end_marker = "\n};\n\n// ---------------------------------------------------------------------------\n// Helpers"
    end_idx = template.find(end_marker, object_start)
    if end_idx == -1:
        raise RuntimeError("Could not find end-of-data marker in template. Has dashboard_template.html changed?")

    # end_idx points at the "\n" just before the template's own closing "};".
    # new_json already supplies a complete, self-closed object (starts with
    # "{" and ends with "}"), so we must skip PAST the template's redundant
    # "}" and resume right at the ";" -- otherwise we'd end up with "}};"
    # (the template's leftover brace plus new_json's own closing brace).
    new_json = json.dumps(data, indent=2)
    new_content = (
        template[:start_idx]
        + "const DATA = "
        + new_json
        + template[end_idx + 2:]
    )

    with open(OUTPUT_PATH, "w") as f:
        f.write(new_content)

    print(f"Wrote {OUTPUT_PATH} with real data from {DATA_PATH}. Open it directly in a browser.")


if __name__ == "__main__":
    main()
