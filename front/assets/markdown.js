/**
 * 轻量 Markdown 渲染：先把文本整体做 HTML 转义，再做白名单转换。
 *
 * 因此模型输出里的任何标签都只会以文字出现，注入脚本在构造上就不可能发生，
 * 不需要再引入额外的净化库。支持标题、粗体、斜体、行内代码、代码块、
 * 有序与无序列表、引用、分隔线、管道表格和链接。
 *
 * 两条渲染规则的边界必须遵守：
 * - 模型输出走本模块（先转义再转换）；
 * - 检索到的原文只使用 textContent 赋值，不走这里。
 *
 * 已知取舍：嵌套列表按单层渲染；表格不做列对齐。
 */

const CODE_PLACEHOLDER = '\u0001';

/** 转义 HTML 特殊字符，是后续所有转换的安全前提。 */
export function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** 只允许常见协议和站内地址，其余链接按普通文字显示，避免 javascript: 之类的地址。 */
function safeHref(href) {
  const value = String(href ?? '').trim();
  if (/^(https?:\/\/|mailto:|\/|#)/i.test(value)) return value;
  return '';
}

/** 转换行内元素，输入必须是已经转义过的文本。 */
export function renderInline(escaped) {
  const codes = [];
  // 行内代码先取出，其中的星号不应再被当成强调标记。
  let text = escaped.replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(code);
    return `${CODE_PLACEHOLDER}${codes.length - 1}${CODE_PLACEHOLDER}`;
  });
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/(^|[^*\w])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
    const target = safeHref(href);
    if (!target) return match;
    return `<a href="${target}" target="_blank" rel="noreferrer noopener">${label}</a>`;
  });
  return text.replace(new RegExp(`${CODE_PLACEHOLDER}(\\d+)${CODE_PLACEHOLDER}`, 'g'), (_, index) => {
    return `<code class="md-inline">${codes[Number(index)]}</code>`;
  });
}

const HEADING = /^(#{1,6})\s+(.*)$/;
const FENCE = /^\s*```/;
const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
// 有序列表：半角点号和右括号后面必须跟空格，否则 "1.5 亿元" 会被当成列表项；
// 中文顿号和全角句点在中文排版里本来就不加空格，所以紧跟正文也算。
const NUMBERED = /^\s*\d+(?:[.)]\s+|、\s*|．\s*)(.*)$/;
const QUOTE = /^\s*>\s?(.*)$/;
// 起始编号的合理范围：超出就别写进 HTML 属性，交给浏览器从 1 开始。
const MAX_LIST_START = 999;

/** 判断一行是否是表格行：至少两个竖线分隔的单元格。 */
function isTableRow(line) {
  const trimmed = line.trim();
  return trimmed.startsWith('|') && trimmed.slice(1).includes('|');
}

/** 判断一行是否是表格的分隔行，例如 | --- | :--: |。 */
function isTableDivider(line) {
  return /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(line) && line.includes('-');
}

/** 把一行表格拆成单元格，去掉首尾竖线。 */
function splitRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim());
}

/** 判断某一行是否会开启一个新的块，用于结束段落收集。 */
function startsBlock(line) {
  return (
    line.trim() === '' ||
    FENCE.test(line) ||
    RULE.test(line) ||
    HEADING.test(line) ||
    BULLET.test(line) ||
    NUMBERED.test(line) ||
    QUOTE.test(line) ||
    isTableRow(line)
  );
}

/** 把转义后的多行文本按块渲染成 HTML。 */
export function renderMarkdown(text) {
  const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.trim() === '') {
      index += 1;
      continue;
    }
    if (FENCE.test(line)) {
      const language = escapeHtml(line.trim().replace(/^```/, '').trim());
      const body = [];
      index += 1;
      while (index < lines.length && !FENCE.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1;
      const attribute = language ? ` data-language="${language}"` : '';
      out.push(`<pre class="md-code"><code${attribute}>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }
    if (RULE.test(line)) {
      out.push('<hr class="md-rule">');
      index += 1;
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level} class="md-heading">${renderInline(escapeHtml(heading[2]))}</h${level}>`);
      index += 1;
      continue;
    }
    if (isTableRow(line) && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
      const header = splitRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && isTableRow(lines[index])) {
        rows.push(splitRow(lines[index]));
        index += 1;
      }
      const head = header.map((cell) => `<th>${renderInline(escapeHtml(cell))}</th>`).join('');
      const body = rows
        .map((row) => {
          const cells = row.map((cell) => `<td>${renderInline(escapeHtml(cell))}</td>`).join('');
          return `<tr>${cells}</tr>`;
        })
        .join('');
      out.push(
        `<div class="md-table-wrap"><table class="md-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`,
      );
      continue;
    }
    if (QUOTE.test(line)) {
      const body = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        body.push(QUOTE.exec(lines[index])[1]);
        index += 1;
      }
      out.push(`<blockquote class="md-quote">${renderInline(escapeHtml(body.join('\n')))}</blockquote>`);
      continue;
    }
    if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = NUMBERED.test(line);
      const pattern = ordered ? NUMBERED : BULLET;
      const items = [];
      while (index < lines.length) {
        const matched = pattern.exec(lines[index]);
        if (matched) {
          items.push(`<li>${renderInline(escapeHtml(matched[1]))}</li>`);
          index += 1;
          continue;
        }
        // 空行不结束列表：模型常在各项之间空一行，CommonMark 也把它们算同一个列表，
        // 拆成多个 ol 会让每一项都显示成「1.」。只有下一行仍是同类列表项时才跨过空行。
        const next = lines[index + 1];
        if (lines[index].trim() === '' && next !== undefined && pattern.test(next)) {
          index += 1;
          continue;
        }
        break;
      }
      const tag = ordered ? 'ol' : 'ul';
      // 按首项编号设置起始值：模型从中间接着编号时，不能悄悄改回从 1 开始。
      const start = ordered ? Number.parseInt(line, 10) : 1;
      const startAttribute =
        Number.isInteger(start) && start !== 1 && start >= 0 && start <= MAX_LIST_START
          ? ` start="${start}"`
          : '';
      out.push(`<${tag} class="md-list"${startAttribute}>${items.join('')}</${tag}>`);
      continue;
    }
    const paragraph = [];
    while (index < lines.length && !startsBlock(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    out.push(`<p class="md-paragraph">${renderInline(escapeHtml(paragraph.join('\n')))}</p>`);
  }
  return out.join('');
}

/**
 * 返回可以安全渲染的前缀长度：最后一个位于已闭合代码块之外的空白行之后。
 *
 * 流式输出时只把这一段追加到页面，尾部不完整的部分用纯文本显示，
 * 避免每来一个词元就重新解析全文。
 */
export function findCommittedLength(text) {
  const value = String(text ?? '').replace(/\r\n?/g, '\n');
  let fenceOpen = false;
  let committed = 0;
  let offset = 0;
  for (const line of value.split('\n')) {
    if (FENCE.test(line)) fenceOpen = !fenceOpen;
    offset += line.length + 1;
    if (!fenceOpen && line.trim() === '') committed = Math.min(offset, value.length);
  }
  return committed;
}
