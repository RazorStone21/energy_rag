/**
 * 内联 SVG 图标与品牌标识。
 * 图标集中放在这里，避免在页面和脚本里散落大段路径数据。
 */

const STROKE = 'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"';

/** 图标名称到路径数据的映射，路径按 24×24 视窗绘制。 */
export const ICONS = {
  plus: `<path ${STROKE} d="M12 5v14M5 12h14"/>`,
  send: `<path ${STROKE} d="M4.5 12h13M12 5.5l6.5 6.5-6.5 6.5"/>`,
  stop: `<rect ${STROKE} x="7" y="7" width="10" height="10" rx="2"/>`,
  copy: `<rect ${STROKE} x="9" y="9" width="10" height="10" rx="2"/><path ${STROKE} d="M5 15V7a2 2 0 0 1 2-2h8"/>`,
  check: `<path ${STROKE} d="M5 12.5l4.5 4.5L19 7.5"/>`,
  sun: `<circle ${STROKE} cx="12" cy="12" r="4"/><path ${STROKE} d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6L17 7M7 17l-1.4 1.4"/>`,
  moon: `<path ${STROKE} d="M20 13.5A8 8 0 0 1 10.5 4a8 8 0 1 0 9.5 9.5z"/>`,
  chevron: `<path ${STROKE} d="M8 10l4 4 4-4"/>`,
  close: `<path ${STROKE} d="M6 6l12 12M18 6L6 18"/>`,
  menu: `<path ${STROKE} d="M4 7h16M4 12h16M4 17h16"/>`,
  doc: `<path ${STROKE} d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path ${STROKE} d="M14 3v5h5"/>`,
  table: `<rect ${STROKE} x="4" y="5" width="16" height="14" rx="2"/><path ${STROKE} d="M4 10h16M10 10v9"/>`,
  image: `<rect ${STROKE} x="4" y="5" width="16" height="14" rx="2"/><circle ${STROKE} cx="9" cy="10" r="1.5"/><path ${STROKE} d="M5 17l4.5-4.5L13 16l2.5-2.5L19 17"/>`,
  prompt: `<path ${STROKE} d="M5 7h14M5 12h9M5 17h11"/>`,
  clock: `<circle ${STROKE} cx="12" cy="12" r="7.5"/><path ${STROKE} d="M12 8v4.5l3 1.8"/>`,
  refresh: `<path ${STROKE} d="M19 12a7 7 0 1 1-2.1-5M19 5v4h-4"/>`,
  alert: `<path ${STROKE} d="M12 8v5M12 16.5v.5"/><circle ${STROKE} cx="12" cy="12" r="8"/>`,
  sparkle: `<path ${STROKE} d="M12 4l1.4 4.1L17.5 9.5l-4.1 1.4L12 15l-1.4-4.1L6.5 9.5l4.1-1.4z"/><path ${STROKE} d="M18 15.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/>`,
};

/**
 * 把页面里带 data-icon 的元素替换成对应的内联图标。
 * 传入的元素自身带 data-icon 时也会处理；找不到名称时保留原内容，方便排查拼写错误。
 */
export function mountIcons(root = document) {
  const targets = [];
  if (root instanceof Element && root.matches('[data-icon]')) targets.push(root);
  targets.push(...root.querySelectorAll('[data-icon]'));
  for (const node of targets) {
    const paths = ICONS[node.dataset.icon];
    if (!paths) continue;
    node.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths}</svg>`;
  }
}
