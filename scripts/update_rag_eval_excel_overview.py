from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from render_rag_eval_excel import CATEGORY_NAMES, write_xlsx


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the overview sheet using manually corrected review values in the workbook."
    )
    parser.add_argument(
        "--workbook",
        required=True,
        help="Path to the manually reviewed Excel workbook",
    )
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    lower_weight = upper - index
    upper_weight = index - lower
    return sorted_values[lower] * lower_weight + sorted_values[upper] * upper_weight


def col_to_index(ref: str) -> int:
    value = 0
    for char in ref:
        if char.isalpha():
            value = value * 26 + ord(char.upper()) - 64
    return value - 1


def load_workbook_rows(path: Path) -> dict[str, list[list[str]]]:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("main:si", NS):
                shared_strings.append("".join(t.text or "" for t in si.iterfind(".//main:t", NS)))

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        workbook_rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in workbook_rels_root.findall("rel:Relationship", NS)
        }

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            if cell_type == "s":
                value = cell.find("main:v", NS)
                return shared_strings[int(value.text)] if value is not None and value.text else ""
            if cell_type == "inlineStr":
                return "".join(t.text or "" for t in cell.iterfind(".//main:t", NS))
            value = cell.find("main:v", NS)
            return value.text if value is not None and value.text is not None else ""

        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook_root.findall("main:sheets/main:sheet", NS):
            name = sheet.attrib["name"]
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rel_map[rel_id]
            if not target.startswith("worksheets/"):
                continue

            sheet_root = ET.fromstring(zf.read(f"xl/{target}"))
            rows: list[list[str]] = []
            for row in sheet_root.findall(".//main:row", NS):
                values_by_index: dict[int, str] = {}
                max_index = -1
                for cell in row.findall("main:c", NS):
                    idx = col_to_index(cell.attrib["r"])
                    values_by_index[idx] = cell_value(cell)
                    max_index = max(max_index, idx)
                rows.append([values_by_index.get(index, "") for index in range(max_index + 1)])
            sheets[name] = rows

    return sheets


def parse_ratio(value: str) -> float:
    text = value.strip()
    if not text:
        return 0.0
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def parse_float(value: str) -> float:
    text = value.strip()
    if text.endswith("s"):
        text = text[:-1]
    return float(text)


def build_manual_overview(review_rows: list[list[str]], existing_summary_rows: list[list[str]]) -> list[list[str]]:
    header = review_rows[0]
    index = {name: idx for idx, name in enumerate(header)}

    rows = review_rows[1:]
    total = len(rows)
    point_rates = [parse_ratio(row[index["要点命中率"]]) for row in rows]
    latencies = [parse_float(row[index["响应时延(s)"]]) for row in rows]
    hit_at_3 = [parse_float(row[index["Hit@3"]]) for row in rows]
    hit_at_5 = [parse_float(row[index["Hit@5"]]) for row in rows]
    reciprocal_ranks = [parse_float(row[index["Reciprocal Rank"]]) for row in rows]
    retrieval_ranks = [row[index["检索排名"]].strip() for row in rows]

    base_info = {}
    for row in existing_summary_rows[1:14]:
        if len(row) >= 2 and row[0]:
            base_info[row[0]] = row[1]

    summary_rows = [
        ["项目", "值"],
        ["数据集", base_info.get("数据集", "")],
        ["Base URL", base_info.get("Base URL", "")],
        ["检索 Top-K", base_info.get("检索 Top-K", "")],
        ["题目数", str(total)],
        ["Hit@3", pct(sum(hit_at_3) / max(total, 1))],
        ["Hit@5", pct(sum(hit_at_5) / max(total, 1))],
        ["MRR", f"{sum(reciprocal_ranks) / max(total, 1):.4f}"],
        ["平均要点命中率（人工）", pct(sum(point_rates) / max(total, 1))],
        ["平均时延", f"{sum(latencies) / max(total, 1):.2f}s"],
        ["P95 时延", f"{percentile(latencies, 0.95):.2f}s"],
        ["漏召样本数", str(sum(1 for rank in retrieval_ranks if rank == '漏召' or not rank))],
        ["要点命中率 < 50%（人工）", str(sum(1 for rate in point_rates if rate < 0.5))],
        ["满分样本数（人工）", str(sum(1 for rate in point_rates if rate == 1.0))],
        ["", ""],
    ]

    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row, point_rate in zip(rows, point_rates):
        grouped[row[index["类别"]]].append((point_rate, row[index["检索排名"]].strip()))

    category_rows = [["类别", "题数", "平均要点命中率（人工）", "漏召数", "0分样本数", "低分样本数(<50%)"]]
    ordered_categories = [CATEGORY_NAMES[key] for key in ("cpu", "disk", "mem", "slow", "svc")]
    for category in ordered_categories:
        items = grouped.get(category, [])
        if not items:
            continue
        rates = [item[0] for item in items]
        ranks = [item[1] for item in items]
        category_rows.append(
            [
                category,
                str(len(items)),
                pct(sum(rates) / len(rates)),
                str(sum(1 for rank in ranks if rank == "漏召" or not rank)),
                str(sum(1 for rate in rates if rate == 0.0)),
                str(sum(1 for rate in rates if rate < 0.5)),
            ]
        )

    return summary_rows + category_rows


def main() -> None:
    args = parse_args()
    workbook_path = Path(args.workbook)
    sheets = load_workbook_rows(workbook_path)

    overview_rows = sheets["概览"]
    review_rows = sheets["人工评审"]
    retrieval_rows = sheets["召回详情"]

    refreshed_overview = build_manual_overview(review_rows, overview_rows)

    write_xlsx(
        workbook_path,
        [
            ("概览", refreshed_overview, [24, 28]),
            ("人工评审", review_rows, [12, 10, 20, 28, 46, 10, 8, 8, 12, 12, 12, 24, 30, 30, 24, 50, 16, 16, 24, 24]),
            ("召回详情", retrieval_rows, [10, 42, 8, 18, 24, 28, 55]),
        ],
    )
    print(f"updated: {workbook_path}")


if __name__ == "__main__":
    main()
