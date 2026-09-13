"""前端 markdown 渲染的回归测试；通过 node 调用真实模块，未安装 node 时跳过。

列表编号是最容易出错的地方：模型常在各项之间空一行，而拆成多个 ol 会让每一项都显示成「1.」。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_JS = ROOT / "front" / "assets" / "markdown.js"

# 用占位符注入参数，避免在 Python 字符串里转义 JavaScript 的花括号。
NODE_SCRIPT = """
import { renderMarkdown } from __MODULE__;
const cases = __CASES__;
const summary = {};
for (const [name, input] of Object.entries(cases)) {
  const html = renderMarkdown(input);
  summary[name] = {
    ol: (html.match(/<ol/g) || []).length,
    ul: (html.match(/<ul/g) || []).length,
    li: (html.match(/<li>/g) || []).length,
    starts: [...html.matchAll(/<ol class="md-list"( start="(\\d+)")?>/g)].map((m) => m[2] || '1'),
    html,
  };
}
console.log(JSON.stringify(summary));
"""


def render_with_node(tmp_path, cases):
    """用 node 调用前端的 markdown 模块，返回每个用例的渲染结果。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("未安装 node，跳过前端渲染测试")
    # 模块要复制成 .mjs 才能被 node 当成 ES 模块导入。
    module = tmp_path / "markdown.mjs"
    module.write_text(MARKDOWN_JS.read_text(encoding="utf-8"), encoding="utf-8")
    script = tmp_path / "render.mjs"
    script.write_text(
        NODE_SCRIPT.replace("__MODULE__", json.dumps(str(module))).replace(
            "__CASES__", json.dumps(cases, ensure_ascii=False)
        ),
        encoding="utf-8",
    )
    completed = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_blank_lines_between_items_keep_one_list(tmp_path):
    """验证各项之间空一行时仍是一个列表，而不是每项各起一个从 1 开始的列表。"""
    rendered = render_with_node(
        tmp_path,
        {
            "spaced": "包括：\n\n1. 甲\n\n2. 乙\n\n3. 丙",
            "dense": "1. 甲\n2. 乙\n3. 丙",
            "bullets": "- 甲\n\n- 乙\n\n- 丙",
        },
    )
    assert rendered["spaced"]["ol"] == 1
    assert rendered["spaced"]["li"] == 3
    assert rendered["dense"]["ol"] == 1 and rendered["dense"]["li"] == 3
    assert rendered["bullets"]["ul"] == 1 and rendered["bullets"]["li"] == 3


def test_blank_line_only_continues_the_same_list(tmp_path):
    """验证列表遇到普通段落会正常结束，末尾的空行留给后面的块。"""
    rendered = render_with_node(
        tmp_path,
        {
            "paragraph": "1. 甲\n\n2. 乙\n\n综上，总结如下。",
            "next_list": "1. 甲\n2. 乙\n\n- 丙",
        },
    )
    assert rendered["paragraph"]["li"] == 2 and rendered["paragraph"]["ol"] == 1
    assert '<p class="md-paragraph">综上，总结如下。</p>' in rendered["paragraph"]["html"]
    # 换成无序列表要另起一个 ul，而不是并进前面的 ol。
    assert rendered["next_list"]["ol"] == 1 and rendered["next_list"]["ul"] == 1


def test_ordered_list_keeps_its_starting_number(tmp_path):
    """验证从中间接着编号的列表不会被重新从 1 开始。"""
    rendered = render_with_node(
        tmp_path,
        {"from_three": "3. 第三项\n4. 第四项", "from_one": "1. 第一项\n2. 第二项"},
    )
    assert rendered["from_three"]["starts"] == ["3"]
    assert rendered["from_one"]["starts"] == ["1"]


def test_chinese_enumerated_list_is_recognized(tmp_path):
    """验证中文顿号编号且后面不空格的列表能被识别。"""
    rendered = render_with_node(tmp_path, {"chinese": "1、甲\n2、乙\n3、丙"})
    assert rendered["chinese"]["ol"] == 1
    assert rendered["chinese"]["li"] == 3


def test_decimal_number_is_not_a_list(tmp_path):
    """验证「1.5 亿元」这类小数不会被当成列表项。"""
    rendered = render_with_node(
        tmp_path, {"decimal": "1.5 亿元，同比增长。", "dotted": "2026.10 发布"}
    )
    assert rendered["decimal"]["ol"] == 0 and rendered["decimal"]["li"] == 0
    assert rendered["dotted"]["ol"] == 0 and rendered["dotted"]["li"] == 0


def test_model_output_is_escaped_before_markup(tmp_path):
    """验证模型输出先转义再转换，标签只会以文字出现。"""
    rendered = render_with_node(
        tmp_path,
        {
            "html": "<img src=x onerror=alert(1)>",
            "link": "[点我](javascript:alert(1))",
            "code": "`<b>粗体</b>`",
        },
    )
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered["html"]["html"]
    assert "<img" not in rendered["html"]["html"]
    # 危险协议的链接按普通文字显示，不生成 a 标签。
    assert "<a " not in rendered["link"]["html"]
    assert "&lt;b&gt;粗体&lt;/b&gt;" in rendered["code"]["html"]
