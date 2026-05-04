from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]

CATEGORY_NAMES = {
    "cpu": "CPU 告警",
    "mem": "内存告警",
    "disk": "磁盘告警",
    "slow": "慢响应",
    "svc": "服务不可用",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render RAG eval JSON into an Excel workbook.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "docs" / "rag_eval_results.json"),
        help="Path to rag eval result json",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "docs" / "rag_eval_results_review.xlsx"),
        help="Path to output xlsx",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def score_label(point_hit_rate: float, retrieval_rank: int | None) -> str:
    if retrieval_rank is None:
        return "高优先级：漏召"
    if point_hit_rate == 1.0:
        return "通过"
    if point_hit_rate >= 0.5:
        return "可接受，建议人工复核"
    return "高优先级：答案不足"


def review_focus(item: dict) -> str:
    point_hit_rate = float(item["point_hit_rate"])
    retrieval_rank = item["retrieval_rank"]
    latency = float(item["latency_seconds"])

    focus: list[str] = []
    if retrieval_rank is None:
        focus.append("先看检索是否漏召")
    elif retrieval_rank and retrieval_rank > 3:
        focus.append("先看相关文档排序是否偏后")

    if point_hit_rate == 0:
        focus.append("重点看答案是否答偏/幻觉扩写")
    elif point_hit_rate < 0.5:
        focus.append("重点看关键要点缺失")
    elif point_hit_rate < 1.0:
        focus.append("重点看是否遗漏部分要点")

    if latency >= 8:
        focus.append("顺带关注时延偏高")

    return "；".join(focus) if focus else "整体表现稳定，可抽样复核"


def build_summary_rows(payload: dict) -> list[list[str]]:
    summary = payload["summary"]
    results = payload["results"]
    retrieval_miss = sum(1 for item in results if item["retrieval_rank"] is None)
    low_point = sum(1 for item in results if float(item["point_hit_rate"]) < 0.5)
    perfect = sum(1 for item in results if float(item["point_hit_rate"]) == 1.0)

    rows = [
        ["项目", "值"],
        ["数据集", str(payload["dataset"])],
        ["Base URL", str(payload["base_url"])],
        ["检索 Top-K", str(payload["retrieval_k"])],
        ["题目数", str(summary["total"])],
        ["Hit@3", pct(float(summary["hit_at_3"]))],
        ["Hit@5", pct(float(summary["hit_at_5"]))],
        ["MRR", f"{float(summary['mrr']):.4f}"],
        ["平均要点命中率", pct(float(summary["avg_point_hit_rate"]))],
        ["平均时延", f"{float(summary['avg_latency_seconds']):.2f}s"],
        ["P95 时延", f"{float(summary['p95_latency_seconds']):.2f}s"],
        ["漏召样本数", str(retrieval_miss)],
        ["要点命中率 < 50%", str(low_point)],
        ["满分样本数", str(perfect)],
    ]
    return rows


def build_category_rows(results: list[dict]) -> list[list[str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        grouped[item["id"].split("_")[0]].append(item)

    rows = [["类别", "题数", "平均要点命中率", "漏召数", "0分样本数", "低分样本数(<50%)"]]
    for key in sorted(grouped):
        items = grouped[key]
        avg = sum(float(item["point_hit_rate"]) for item in items) / len(items)
        miss = sum(1 for item in items if item["retrieval_rank"] is None)
        zero = sum(1 for item in items if float(item["point_hit_rate"]) == 0)
        low = sum(1 for item in items if float(item["point_hit_rate"]) < 0.5)
        rows.append(
            [
                CATEGORY_NAMES.get(key, key),
                str(len(items)),
                pct(avg),
                str(miss),
                str(zero),
                str(low),
            ]
        )
    return rows


def join_lines(values: list[str]) -> str:
    return "\n".join(values) if values else ""


def build_review_rows(results: list[dict]) -> list[list[str]]:
    headers = [
        "类别",
        "ID",
        "自动结论",
        "人工优先级",
        "问题",
        "检索排名",
        "Hit@3",
        "Hit@5",
        "Reciprocal Rank",
        "要点命中率",
        "响应时延(s)",
        "Gold 文档",
        "Top-3 召回",
        "Gold Points",
        "Matched Points",
        "模型回答",
        "人工结论",
        "是否可直接上线",
        "主要问题",
        "修订建议",
    ]
    rows = [headers]

    sorted_results = sorted(
        results,
        key=lambda item: (float(item["point_hit_rate"]), -(float(item["latency_seconds"]))),
    )

    for item in sorted_results:
        retrieved_docs = item.get("retrieved_docs", [])
        top_docs = [
            f"#{doc.get('rank', '-')} {doc.get('file_name', '-')} | {' / '.join(doc.get('headers', [])) or '-'}"
            for doc in retrieved_docs[:3]
        ]
        rows.append(
            [
                CATEGORY_NAMES.get(item["id"].split("_")[0], item["id"].split("_")[0]),
                item["id"],
                score_label(float(item["point_hit_rate"]), item["retrieval_rank"]),
                review_focus(item),
                item["question"],
                "漏召" if item["retrieval_rank"] is None else str(item["retrieval_rank"]),
                str(item["hit_at_3"]),
                str(item["hit_at_5"]),
                f"{float(item['reciprocal_rank']):.2f}",
                pct(float(item["point_hit_rate"])),
                f"{float(item['latency_seconds']):.3f}",
                join_lines(item.get("gold_docs", [])),
                join_lines(top_docs),
                join_lines([f"- {point}" for point in item.get("gold_answer_points", [])]),
                join_lines([f"- {point}" for point in item.get("matched_points", [])]),
                item.get("answer", ""),
                "待评审",
                "待定",
                "",
                "",
            ]
        )
    return rows


def build_retrieval_rows(results: list[dict]) -> list[list[str]]:
    rows = [[
        "ID",
        "问题",
        "rank",
        "file_name",
        "headers",
        "queries",
        "preview",
    ]]
    for item in results:
        docs = item.get("retrieved_docs", [])
        if not docs:
            rows.append([item["id"], item["question"], "漏召", "", "", "", ""])
            continue
        for doc in docs[:5]:
            rows.append(
                [
                    item["id"],
                    item["question"],
                    str(doc.get("rank", "")),
                    str(doc.get("file_name", "")),
                    " / ".join(doc.get("headers", [])),
                    "；".join(doc.get("queries", [])),
                    str(doc.get("preview", "")),
                ]
            )
    return rows


def column_letter(index: int) -> str:
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def build_shared_strings(sheets: list[list[list[str]]]) -> tuple[list[str], dict[str, int]]:
    mapping: dict[str, int] = {}
    unique: list[str] = []
    for sheet in sheets:
        for row in sheet:
            for value in row:
                if value not in mapping:
                    mapping[value] = len(unique)
                    unique.append(value)
    return unique, mapping


def xml_header() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def shared_strings_xml(strings: list[str]) -> str:
    body = "".join(f"<si><t xml:space=\"preserve\">{escape(value)}</t></si>" for value in strings)
    return (
        f"{xml_header()}"
        f"<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        f"count=\"{len(strings)}\" uniqueCount=\"{len(strings)}\">{body}</sst>"
    )


def styles_xml() -> str:
    return (
        f"{xml_header()}"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"2\">"
        "<font><sz val=\"11\"/><name val=\"Calibri\"/></font>"
        "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/></font>"
        "</fonts>"
        "<fills count=\"2\">"
        "<fill><patternFill patternType=\"none\"/></fill>"
        "<fill><patternFill patternType=\"gray125\"/></fill>"
        "</fills>"
        "<borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        "<cellXfs count=\"3\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
        "<xf numFmtId=\"0\" fontId=\"1\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyFont=\"1\" applyAlignment=\"1\">"
        "<alignment horizontal=\"center\" vertical=\"center\" wrapText=\"1\"/>"
        "</xf>"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyAlignment=\"1\">"
        "<alignment vertical=\"top\" wrapText=\"1\"/>"
        "</xf>"
        "</cellXfs>"
        "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>"
        "</styleSheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets_xml = "".join(
        f"<sheet name=\"{escape(name)}\" sheetId=\"{index}\" r:id=\"rId{index}\"/>"
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        f"{xml_header()}"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<sheets>{sheets_xml}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for index in range(1, sheet_count + 1):
        rels.append(
            f"<Relationship Id=\"rId{index}\" "
            f"Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
            f"Target=\"worksheets/sheet{index}.xml\"/>"
        )
    rels.append(
        f"<Relationship Id=\"rId{sheet_count + 1}\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/>"
    )
    rels.append(
        f"<Relationship Id=\"rId{sheet_count + 2}\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings\" "
        "Target=\"sharedStrings.xml\"/>"
    )
    return (
        f"{xml_header()}"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        f"{''.join(rels)}"
        "</Relationships>"
    )


def root_rels_xml() -> str:
    return (
        f"{xml_header()}"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"xl/workbook.xml\"/>"
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        "<Override PartName=\"/xl/workbook.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
        "<Override PartName=\"/xl/styles.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>",
        "<Override PartName=\"/xl/sharedStrings.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml\"/>",
    ]
    for index in range(1, sheet_count + 1):
        overrides.append(
            f"<Override PartName=\"/xl/worksheets/sheet{index}.xml\" "
            "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        )
    return (
        f"{xml_header()}"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        f"{''.join(overrides)}"
        "</Types>"
    )


def worksheet_xml(
    rows: list[list[str]],
    shared_string_map: dict[str, int],
    widths: list[float] | None = None,
) -> str:
    max_columns = max((len(row) for row in rows), default=0)
    columns_xml = ""
    if widths:
        cols = []
        for index, width in enumerate(widths, start=1):
            cols.append(
                f"<col min=\"{index}\" max=\"{index}\" width=\"{width}\" customWidth=\"1\"/>"
            )
        columns_xml = f"<cols>{''.join(cols)}</cols>"

    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            cell_ref = f"{column_letter(col_index)}{row_index}"
            style_id = "1" if row_index == 1 else "2"
            string_id = shared_string_map[value]
            cells.append(
                f"<c r=\"{cell_ref}\" t=\"s\" s=\"{style_id}\"><v>{string_id}</v></c>"
            )
        sheet_rows.append(f"<row r=\"{row_index}\">{''.join(cells)}</row>")

    last_column = column_letter(max_columns or 1)
    auto_filter = f"<autoFilter ref=\"A1:{last_column}1\"/>" if rows else ""
    freeze = (
        "<sheetViews><sheetView workbookViewId=\"0\">"
        "<pane ySplit=\"1\" topLeftCell=\"A2\" activePane=\"bottomLeft\" state=\"frozen\"/>"
        "<selection pane=\"bottomLeft\" activeCell=\"A2\" sqref=\"A2\"/>"
        "</sheetView></sheetViews>"
    )
    dimension = f"A1:{last_column}{len(rows) if rows else 1}"

    return (
        f"{xml_header()}"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        f"<dimension ref=\"{dimension}\"/>"
        f"{freeze}"
        f"{columns_xml}"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        f"{auto_filter}"
        "</worksheet>"
    )


def write_xlsx(output_path: Path, sheets: list[tuple[str, list[list[str]], list[float] | None]]) -> None:
    sheet_rows = [rows for _, rows, _ in sheets]
    shared_strings, shared_map = build_shared_strings(sheet_rows)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        workbook.writestr("_rels/.rels", root_rels_xml())
        workbook.writestr("xl/workbook.xml", workbook_xml([name for name, _, _ in sheets]))
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        workbook.writestr("xl/sharedStrings.xml", shared_strings_xml(shared_strings))
        workbook.writestr("xl/styles.xml", styles_xml())

        for index, (_, rows, widths) in enumerate(sheets, start=1):
            workbook.writestr(
                f"xl/worksheets/sheet{index}.xml",
                worksheet_xml(rows, shared_map, widths),
            )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = load_payload(input_path)
    results = payload["results"]

    sheets = [
        (
            "概览",
            build_summary_rows(payload) + [["", ""], *build_category_rows(results)],
            [24, 28],
        ),
        (
            "人工评审",
            build_review_rows(results),
            [12, 10, 20, 28, 46, 10, 8, 8, 12, 12, 12, 24, 30, 30, 24, 50, 16, 16, 24, 24],
        ),
        (
            "召回详情",
            build_retrieval_rows(results),
            [10, 42, 8, 18, 24, 28, 55],
        ),
    ]

    write_xlsx(output_path, sheets)
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
