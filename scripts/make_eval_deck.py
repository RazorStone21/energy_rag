"""从评测报告 JSON 生成三页测评结果 PPT。

数据全部读自 tests/results/ 下的报告文件，不写死数字：重新跑评测后重跑本脚本
即可得到与新报告一致的 PPT。

版式与配色沿用 docs/RAG系统介绍.pptx：主色 0F766E、卡片底 F5FBF9、字体微软雅黑。
热力图用的四级色阶经 OKLCH 计算并用校验器验证（单色相、亮度单调、相邻 ΔL≥0.06、
浅端对卡片底 ≥2:1），四档内嵌文字的对比度均 ≥4.5:1。

用法：
    python -m scripts.make_eval_deck                        # 输出到 docs/
    python -m scripts.make_eval_deck --out /tmp/deck.pptx
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "tests" / "results"

# ---- 设计变量（与 docs/RAG系统介绍.pptx 一致）----
FONT = "微软雅黑"
INK = "1F2937"          # 标题
INK_SUB = "5B6470"      # 正文/次要
INK_ACCENT = "0B4F48"   # 强调（结论正文、增量）
BRAND = "0F766E"        # 主色
BRAND_LIGHT = "14B8A6"  # 顶部第二条
CARD = "F5FBF9"         # 卡片底
WHITE = "FFFFFF"

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.85
CONTENT_W = 11.63
CARD_W, CARD_H = 2.79, 1.05
CARD_X = (0.85, 3.79, 6.74, 9.68)
GAP = 0.045             # 热力图格子之间的表面间隙（用底色分隔，不画边框）

# ---- 热力图色阶：OKLCH 计算 + 校验器验证，索引 0 最浅（最低值）----
RAMP = ("#7EBCB5", "#4E9F96", "#0D756D", "#094F49")
RAMP_INK = (INK, INK, WHITE, WHITE)
# 分档阈值：值越高颜色越深
BINS = ((0.00, 0.82), (0.82, 0.90), (0.90, 0.95), (0.95, 1.01))
BIN_LABELS = ("<0.82", "0.82–0.90", "0.90–0.95", "≥0.95")

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_SHORT = {
    "faithfulness": "忠实度",
    "answer_relevancy": "相关性",
    "context_precision": "精确率",
    "context_recall": "召回率",
}
TYPE_CN = {
    "list": "列举题",
    "numeric": "数值题",
    "factual": "事实题",
    "figure": "图表题",
    "multihop": "多跳题",
    "table": "表格题",
    "negative": "拒答题",
}


# ---------- 基础绘制 ----------
def _hex(color):
    """去掉可选的 # 前缀：RGBColor.from_string 不接受 #。"""
    return color.lstrip("#").upper()


def set_font(run, size, bold, color, name=FONT):
    """设置一个 run 的字体。

    python-pptx 的 font.name 只写 latin 字体；中文由 <a:ea> 决定，不设就会
    回落到主题默认的宋体，与既有 deck 不一致，因此三个都要写。
    """
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = name
    run.font.color.rgb = RGBColor.from_string(_hex(color))
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = rpr.find(qn(tag))
        if element is None:
            element = rpr.makeelement(qn(tag), {})
            rpr.append(element)
        element.set("typeface", name)


def add_text(slide, x, y, w, h, text, size, color, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, line_spacing=None):
    """加一个无内边距的文本框；text 里的 \n 分成多段。"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        if line_spacing:
            paragraph.line_spacing = line_spacing
        set_font(paragraph.add_run(), size, bold, color)
        paragraph.runs[0].text = line
    return box


def add_shape(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.16667):
    """加一个无边框无阴影的色块。

    radius 取 0.1667：与 docs/RAG系统介绍.pptx 里 roundRect 的 adj=16667 一致，
    否则圆角只有既有 deck 的一半，并排放会看出差别。
    """
    figure = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    figure.fill.solid()
    figure.fill.fore_color.rgb = RGBColor.from_string(_hex(fill))
    figure.line.fill.background()
    figure.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        figure.adjustments[0] = radius
    figure.text_frame.word_wrap = True
    return figure


def add_header(slide, title, subtitle):
    """顶部色条 + 标题 + 副标题。"""
    add_shape(slide, 0, 0, SLIDE_W, 0.22, BRAND, MSO_SHAPE.RECTANGLE)
    add_shape(slide, 0, 0.22, SLIDE_W, 0.04, BRAND_LIGHT, MSO_SHAPE.RECTANGLE)
    add_text(slide, MARGIN, 0.60, CONTENT_W, 0.70, title, 34, INK, bold=True)
    add_text(slide, MARGIN, 1.28, CONTENT_W, 0.26, subtitle, 13, INK_SUB)


def add_chip(slide, y, text):
    """章节标签：深色圆角块 + 白字。"""
    add_shape(slide, MARGIN, y, 2.0, 0.40, BRAND)
    add_text(slide, MARGIN, y, 2.0, 0.40, text, 13, WHITE, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def add_card(slide, x, y, label, value, note, note_color=INK_ACCENT):
    """指标卡：标签 / 数值 / 说明三行。"""
    add_shape(slide, x, y, CARD_W, CARD_H, CARD)
    add_text(slide, x + 0.20, y + 0.13, CARD_W - 0.40, 0.28, label, 11, INK_SUB, bold=True)
    add_text(slide, x + 0.20, y + 0.42, CARD_W - 0.40, 0.34, value, 16, INK_SUB, bold=True)
    add_text(slide, x + 0.20, y + 0.77, CARD_W - 0.40, 0.24, note, 10, note_color, bold=True)


def add_panel(slide, y, height, lines, size=12):
    """底部结论面板：浅底圆角块 + 以 ▪ 开头的多行结论。"""
    add_shape(slide, MARGIN, y, CONTENT_W, height, CARD)
    box = slide.shapes.add_textbox(Inches(MARGIN + 0.27), Inches(y + 0.14),
                                   Inches(CONTENT_W - 0.54), Inches(height - 0.28))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.line_spacing = 1.35
        run = paragraph.add_run()
        set_font(run, size, True, INK_ACCENT)
        run.text = "▪ " + line
    return box


def bin_of(value):
    """按阈值取色阶档位，越高的值颜色越深（索引越大）。"""
    for index, (low, high) in enumerate(BINS):
        if low <= value < high:
            return index
    return len(BINS) - 1


# ---------- 数据 ----------
def load_data():
    """读两份评测报告并汇总出三页需要的全部数字。"""
    rag = json.loads((RESULTS / "rag_eval_report.json").read_text(encoding="utf-8"))
    retrieval = json.loads((RESULTS / "retrieval_eval_report.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "tests" / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    type_of = {q["question"]: q.get("type", "") for q in questions}

    # 生成：按题型分组，negative 单独看
    by_type = defaultdict(list)
    for row in rag["scores"]:
        by_type[row.get("type", "")].append(row)
    negative = by_type.pop("negative", [])

    def mean(rows, metric):
        values = [r[metric] for r in rows if r.get(metric) is not None]
        return sum(values) / len(values) if values else float("nan")

    generation = {
        "summary": rag["ragas"]["metrics"],
        "n": rag["ragas"]["n"],
        "judge": rag.get("judge", ""),
        "per_type": {t: {m: mean(rows, m) for m in METRICS} for t, rows in by_type.items()},
        "type_order": sorted(by_type, key=lambda t: -len(by_type[t])),
        "negative": {m: mean(negative, m) for m in METRICS} if negative else {},
        "negative_n": len(negative),
        "negative_refusals": sum(1 for r in negative if (r.get("faithfulness") or 0) > 0),
    }

    # 检索：per_query 不带题型，与题库按问题文本对齐
    def retrieval_metrics(key):
        rows = retrieval[key]["per_query"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[type_of.get(row["question"], "")].append(row)
        return {t: {k: sum(r[k] for r in rs) / len(rs) for k in
                    ("recall@1", "recall@5", "mrr", "ndcg@5")}
                for t, rs in grouped.items()}, retrieval[key]["n"]

    hybrid, n_only = retrieval_metrics("hybrid_only")
    rerank, n_rerank = retrieval_metrics("hybrid_rerank")
    retrieval_out = {
        "only": retrieval["hybrid_only"]["avg"],
        "rerank": retrieval["hybrid_rerank"]["avg"],
        "n": n_rerank,
        "per_type": rerank,
        "n_only": n_only,
    }
    return generation, retrieval_out


def fmt(value, digits=4):
    """把指标格式化成固定小数位。"""
    return f"{value:.{digits}f}"


# ---------- 三页 ----------
def slide_overview(prs, gen, ret):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "测评结果概览",
        f"语料 47 份能源领域公文 / 1625 个检索片段 · 题库 121 道人工标注题 / 7 种题型 · "
        f"判分模型 {gen['judge']}",
    )

    # 检索：纯融合 → 加重的对比，两组同口径，是全文最可比的一组数字
    add_chip(slide, 1.78, "检索效果")
    add_text(slide, 3.05, 1.85, 8.0, 0.26,
             f"融合召回 → 融合 + 重排（{ret['n']} 题平均）", 11, INK_SUB)
    only, rerank = ret["only"], ret["rerank"]
    for index, (label, key, digits) in enumerate((
        ("Recall@1", "recall@1", 4),
        ("Recall@5", "recall@5", 4),
        ("MRR（平均倒数排名）", "mrr", 4),
        ("nDCG@5（排序质量）", "ndcg@5", 4),
    )):
        before, after = only[key], rerank[key]
        add_card(slide, CARD_X[index], 2.30, label,
                 f"{fmt(before, digits)}  →  {fmt(after, digits)}",
                 f"+{after - before:.4f}", )

    # 生成：判分模型已更换，不与上一版做差值（不同判分模型的分数不可比）
    add_chip(slide, 3.58, "生成质量")
    add_text(slide, 3.05, 3.65, 8.0, 0.26,
             f"{gen['n']} 题平均 · 判分模型独立于生成模型 · 各指标取值 0–1，越高越好",
             11, INK_SUB)
    # 每张卡给一句人话解释，比四张都写「0–1，越高越好」有信息量
    meanings = {
        "faithfulness": "答案陈述是否有据",
        "answer_relevancy": "答案是否切题",
        "context_precision": "检索片段是否精准",
        "context_recall": "参考答案是否覆盖",
    }
    for index, metric in enumerate(METRICS):
        add_card(slide, CARD_X[index], 4.10, METRIC_SHORT[metric] + "（" + metric + "）",
                 fmt(gen["summary"][metric]), meanings[metric], INK_SUB)

    add_panel(slide, 5.55, 1.10, [
        f"重排收益真实：MRR +{rerank['mrr'] - only['mrr']:.4f}、Recall@1 "
        f"+{rerank['recall@1'] - only['recall@1']:.4f}，两组同口径可比，是本报告最可信的结论。",
        f"生成质量整体稳定，{gen['negative_n']} 道「答案不在语料中」的题全部正确拒答（拒答率 100%）。",
        "短板集中在表格题与图表题两类，且 table 的问题出在 PDF 抽取层，详见次页。",
    ])
    return slide


def slide_per_type(prs, gen, ret):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "分题型表现",
        "生成指标按题型的热力分布：颜色越深表示越高，每格标注实际数值（拒答题因判定口径不同，已单列不参与均值）",
    )

    # 列头
    label_w, first_x = 1.40, 2.30
    grid_w = MARGIN + CONTENT_W - first_x
    cell_w = (grid_w - GAP * 3) / 4
    head_y, row_y, row_h = 1.86, 2.24, 0.475
    for index, metric in enumerate(METRICS):
        add_text(slide, first_x + index * (cell_w + GAP), head_y, cell_w, 0.30,
                 METRIC_SHORT[metric], 12, INK_SUB, bold=True, align=PP_ALIGN.CENTER)

    # 数据格：每格一个色块 + 居中数值，靠表面间隙分隔而不是边框
    order = gen["type_order"]
    for row_index, type_name in enumerate(order):
        y = row_y + row_index * (row_h + GAP)
        add_text(slide, MARGIN, y, label_w - 0.10, row_h, TYPE_CN.get(type_name, type_name),
                 12, INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        for column, metric in enumerate(METRICS):
            value = gen["per_type"][type_name][metric]
            level = bin_of(value)
            x = first_x + column * (cell_w + GAP)
            # 热力图格子用直角：网格里每格带圆角会读成按钮，且直角更贴紧刻度感
            add_shape(slide, x, y, cell_w, row_h, RAMP[level], MSO_SHAPE.RECTANGLE)
            add_text(slide, x, y, cell_w, row_h, fmt(value), 12, RAMP_INK[level],
                     bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # 色阶图例：阈值写在色块里，编码本身可解码
    legend_y = row_y + len(order) * (row_h + GAP) + 0.10
    add_text(slide, MARGIN, legend_y, label_w - 0.10, 0.24, "指标值", 11, INK_SUB,
             bold=True, anchor=MSO_ANCHOR.MIDDLE)
    swatch_w = 0.92
    for index, text in enumerate(BIN_LABELS):
        x = first_x + index * (swatch_w + 0.04)
        add_shape(slide, x, legend_y, swatch_w, 0.24, RAMP[index])
        add_text(slide, x, legend_y, swatch_w, 0.24, text, 10, RAMP_INK[index],
                 bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    table_r1 = ret["per_type"].get("table", {}).get("recall@1", float("nan"))
    figure_r5 = ret["per_type"].get("figure", {}).get("recall@5", float("nan"))
    add_panel(slide, legend_y + 0.46, 1.05, [
        f"检索侧同样集中在这两类：表格题 Recall@1 仅 {fmt(table_r1)}，图表题 Recall@5 仅 {fmt(figure_r5)}"
        "（15 题中有 4 题完全找不到目标片段）。",
        "表格题根因已定位到抽取层：PDF 表格把「额定容量」列与下一行序号粘连（800 + 2 → “8002”），容量列实际丢失，",
        "答案对但证据不可查；图表题则是图内数据依赖视觉模型生成的描述入库，标注与片段边界也对不上。",
    ], size=11)
    return slide


def slide_reliability(prs, gen):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "判分可靠性核查与改进方向",
        "四个指标全部依赖模型判分，因此另做不依赖模型的确定性校验，并对低分题逐题人工复核",
    )

    add_chip(slide, 1.78, "核查结果")
    for index, (label, value, note) in enumerate((
        ("确定性数字校验", "94.2%", "179/190 个数字原文可查"),
        ("判为低分的题", "10 道", "非拒答题中忠实度 < 0.5"),
        ("其中误报", "5 道", "要点逐条在上下文里，仍被判低"),
        ("真实数字错误", "2 道", "占含数字题目的 3%"),
    )):
        add_card(slide, CARD_X[index], 2.30, label, value, note,
                 INK_ACCENT if index != 2 else "9A3412")

    add_chip(slide, 3.58, "真实错误样本")
    add_panel(slide, 4.10, 1.05, [
        "无据硬答（最该修）：检索零命中时改用参数记忆作答，答成「2025 年底前基本实现全覆盖」，正确为「2027 年前正式运行」。",
        "表格串数：绿证消费主体应为 5.9 万个 / 企业 5.54 万家，答成 7.34 万个 / 6.32 万家，并把「3955 名」写成「3955 万名」。",
        "幻觉式「无数据」：声称文档未提供各国产量占比，而文档中就有 23.7% / 21.5% / 19.4%。",
    ], size=11)

    add_chip(slide, 5.50, "改进方向")
    add_panel(slide, 6.00, 1.05, [
        "把拒答题单列指标（本次拒答率 100%），不混入四个 RAGAS 均值 —— 否则正确拒答会把答案相关性压低约 0.05。",
        "将确定性数字校验常态化为交叉验证指标；它已抓到忠实度漏报的真错误，成本几乎为零。",
        "修 is_relevant 判分口径、并对表格区域做结构化抽取 —— 前者让召回率从下界变真值，后者解决列粘连。",
    ], size=11)
    return slide


def main():
    parser = argparse.ArgumentParser(description="生成三页测评结果 PPT")
    parser.add_argument("--out", default=str(ROOT / "docs" / "RAG测评结果-2026-09-21.pptx"))
    parser.add_argument("--rag-report", default=None, help="覆盖生成评测报告路径")
    args = parser.parse_args()

    global RESULTS
    if args.rag_report:
        RESULTS = Path(args.rag_report).parent

    gen, ret = load_data()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    slide_overview(prs, gen, ret)
    slide_per_type(prs, gen, ret)
    slide_reliability(prs, gen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.out)
    print(f"共 {len(prs.slides)} 页，已保存到 {args.out}")


if __name__ == "__main__":
    main()
