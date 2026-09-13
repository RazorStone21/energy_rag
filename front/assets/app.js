/**
 * 能源文档问答前端：会话管理、SSE 流式渲染、参考来源面板与索引状态。
 *
 * 分节顺序：状态、接口、SSE、消息渲染、来源与面板、状态卡、会话、初始化。
 * 模型输出的渲染统一走 markdown.js（先转义再转换）；检索到的原文只用 textContent。
 */

import { mountIcons } from './icons.js';
import { findCommittedLength, renderMarkdown } from './markdown.js';

const THEME_KEY = 'energy-rag-theme';
// 会话记录由服务端保存成 markdown，浏览器只留主题和检索开关。

const SCORE_LABELS = { dense: '向量', bm25: 'BM25', rrf: 'RRF', rerank: '重排' };
const TIMING_LABELS = {
  rewrite: '改写',
  retrieval: '检索',
  rerank: '重排',
  context: '上下文',
  generation: '生成',
};

const SUGGESTIONS = [
  '到 2030 年新型储能的装机目标是多少？',
  '绿电交易和绿证核发如何衔接？',
  '全国统一电力市场的评价指标有哪些？',
  '跨省跨区输电价格是怎么确定的？',
];

const state = {
  conversations: [],   // 服务端返回的会话摘要，不含正文
  messages: [],        // 当前会话的正文
  activeId: null,      // 当前会话 id；为空表示还没落盘的新对话
  streaming: false,
  status: null,
  memory: null,
  controller: null,
  settings: { hybrid: true, rerank: true },
  statusTimer: null,
  listSeq: 0,          // 会话列表请求序号，防止旧响应覆盖新列表
};

/** 按 id 取元素，避免在脚本里反复写 document.getElementById。 */
const $ = (id) => document.getElementById(id);

/* ---------- 接口 ---------- */

/** 发送 JSON 请求并解析返回，失败时抛出带中文说明的错误。 */
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `请求失败（${response.status}）`);
  }
  return response.json();
}

/* ---------- SSE ---------- */

/** 解析一条 SSE 报文：忽略注释行，多行 data 用换行拼接。 */
function parseFrame(frame) {
  let name = 'message';
  const data = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue;
    if (line.startsWith('event:')) name = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
  }
  if (!data.length) return null;
  try {
    return { name, data: JSON.parse(data.join('\n')) };
  } catch {
    return null;
  }
}

/**
 * 用 fetch 读取 POST 返回的事件流。
 * 浏览器的 EventSource 只支持 GET，所以这里手动按空行切帧；
 * 解码必须用 stream 模式，否则被切开的多字节汉字会变成乱码。
 */
async function streamRequest(path, body, onEvent, signal) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `请求失败（${response.status}）`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      let index = buffer.indexOf('\n\n');
      while (index !== -1) {
        const event = parseFrame(buffer.slice(0, index));
        buffer = buffer.slice(index + 2);
        if (event) onEvent(event.name, event.data);
        index = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

/* ---------- 消息渲染 ---------- */

/** 创建一个流式渲染器：已完结的段落转成 HTML 追加，未完结的尾部保持纯文本。 */
function createStreamer(content) {
  let raw = '';
  let committed = 0;
  let scheduled = false;
  const tail = document.createElement('div');
  content.classList.add('stream-cursor');
  content.appendChild(tail);

  /** 把新增的已完成段落追加到页面，并更新尾部纯文本。 */
  function paint() {
    scheduled = false;
    const boundary = findCommittedLength(raw);
    if (boundary > committed) {
      // 插入到尾部节点之前，尾部节点始终保持未完结文本，供下一帧继续覆盖。
      tail.insertAdjacentHTML('beforebegin', renderMarkdown(raw.slice(committed, boundary)));
      committed = boundary;
    }
    tail.textContent = raw.slice(committed);
  }

  /** 用完整文本替换内容区，结束流式状态。 */
  function settle(text) {
    raw = typeof text === 'string' ? text : raw;
    content.classList.remove('stream-cursor');
    tail.remove();
    content.innerHTML = renderMarkdown(raw);
    return raw;
  }

  return {
    push(text) {
      raw += text;
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(paint);
    },
    finish(answer) {
      return settle(answer);
    },
    stop() {
      return settle(null);
    },
    text() {
      return raw;
    },
  };
}

/** 按各字段量纲格式化分数；None 表示该阶段没有产生分数，不显示为 0。 */
function formatScore(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const magnitude = Math.abs(value);
  if (magnitude === 0) return '0';
  if (magnitude < 0.01) return value.toFixed(4);
  if (magnitude < 1) return value.toFixed(3);
  return value.toFixed(2);
}

/** 生成一个小按钮，内容为图标加文字。 */
function actionButton(icon, label, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'action';
  button.innerHTML = `<span class="icon" data-icon="${icon}"></span><span></span>`;
  button.lastChild.textContent = label;
  button.addEventListener('click', onClick);
  mountIcons(button);
  return button;
}

/** 在消息正文之后追加一条错误提示。 */
function renderError(container, message) {
  const box = document.createElement('div');
  box.className = 'msg__error';
  box.innerHTML = '<span class="icon" data-icon="alert"></span><span></span>';
  box.lastChild.textContent = message;
  container.appendChild(box);
  mountIcons(box);
}

/* ---------- 来源与面板 ---------- */

/** 生成一张来源卡片，正文只用 textContent 赋值。 */
function sourceCard(source) {
  const card = document.createElement('article');
  card.className = 'source-card';

  const head = document.createElement('div');
  head.className = 'source-card__head';

  const index = document.createElement('span');
  index.className = 'source-card__index';
  index.textContent = String(source.index ?? '');

  const main = document.createElement('div');
  main.className = 'source-card__main';

  const title = document.createElement('div');
  title.className = 'source-card__title';
  title.textContent = source.label || source.source || '未知来源';
  if (source.type_label) {
    const type = document.createElement('span');
    type.className = 'source-card__type';
    type.textContent = source.type_label;
    title.appendChild(type);
  }
  main.appendChild(title);

  // 重新打开的会话只存了来源的位置标签，没有分数也没有片段正文，
  // 这两块要按实际存在与否渲染，而不是渲染成一排“—”。
  if (source.scores) {
    const scores = document.createElement('div');
    scores.className = 'source-card__scores';
    for (const [key, label] of Object.entries(SCORE_LABELS)) {
      const item = document.createElement('span');
      item.className = 'score';
      item.innerHTML = '<span></span><span class="score__value"></span>';
      item.firstChild.textContent = label;
      item.lastChild.textContent = formatScore(source.scores[key]);
      scores.appendChild(item);
    }
    main.appendChild(scores);
  }

  head.append(index, main);
  card.appendChild(head);
  if (source.content) {
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'source-card__toggle';
    toggle.textContent = '展开';
    const text = document.createElement('p');
    text.className = 'source-card__text';
    text.hidden = true;
    text.textContent = source.content;
    toggle.addEventListener('click', () => {
      text.hidden = !text.hidden;
      toggle.textContent = text.hidden ? '展开' : '收起';
    });
    head.appendChild(toggle);
    card.appendChild(text);
  }
  return card;
}

/** 生成一个展开面板，包含标题和内容容器。 */
function createPanel(title) {
  const panel = document.createElement('div');
  panel.className = 'msg__panel';
  panel.hidden = true;
  const head = document.createElement('div');
  head.className = 'msg__panel-head';
  const label = document.createElement('span');
  label.textContent = title;
  head.appendChild(label);
  const body = document.createElement('div');
  body.className = 'msg__panel-body';
  panel.append(head, body);
  return { panel, body, label };
}

/** 渲染来源、提示词和耗时三个可展开面板；按钮放进操作行，面板挂在消息下方。 */
function renderPanels(message, actions, container) {
  const sources = message.sources || [];
  const buttons = [];

  const sourcePanel = createPanel(`参考来源 · ${sources.length}`);
  const list = document.createElement('div');
  list.className = 'source-list';
  sources.forEach((source) => list.appendChild(sourceCard(source)));
  if (!sources.length) {
    const empty = document.createElement('p');
    empty.className = 'hint';
    empty.textContent =
      '本次没有检索到相关片段，因此没有调用生成模型。追问时如果指代不明确也可能出现这种情况，可以试着把问题写完整。';
    list.appendChild(empty);
  }
  sourcePanel.body.appendChild(list);
  buttons.push({
    icon: 'doc',
    label: `参考来源 · ${sources.length}`,
    panel: sourcePanel.panel,
  });

  if (message.prompt) {
    const promptPanel = createPanel('实际使用的 Prompt');
    const box = document.createElement('pre');
    box.className = 'prompt-box';
    box.textContent = message.prompt;
    promptPanel.body.appendChild(box);
    buttons.push({ icon: 'prompt', label: '查看 Prompt', panel: promptPanel.panel });
  }

  if (message.timings) {
    const timingPanel = createPanel('各阶段耗时');
    const list = document.createElement('div');
    list.className = 'timing-list';
    for (const [key, label] of Object.entries(TIMING_LABELS)) {
      const value = message.timings[key];
      const item = document.createElement('span');
      item.className = 'timing';
      item.innerHTML = '<span></span><span class="timing__value"></span>';
      item.firstChild.textContent = key === 'generation' ? `${label}（含首次加载）` : label;
      const seconds = typeof value === 'number' && Number.isFinite(value) ? value : null;
      item.lastChild.textContent = seconds === null ? '—' : `${seconds.toFixed(2)} s`;
      list.appendChild(item);
    }
    timingPanel.body.appendChild(list);
    buttons.push({ icon: 'clock', label: '耗时', panel: timingPanel.panel });
  }

  for (const item of buttons) {
    item.panel.hidden = true;
    const button = actionButton(item.icon, item.label, () => {
      const opening = item.panel.hidden;
      container.querySelectorAll('.msg__panel').forEach((panel) => {
        panel.hidden = true;
      });
      actions.querySelectorAll('.action[data-panel]').forEach((other) => {
        other.classList.remove('is-active');
      });
      item.panel.hidden = !opening;
      button.classList.toggle('is-active', opening);
    });
    button.dataset.panel = 'true';
    actions.appendChild(button);
    container.appendChild(item.panel);
  }
}

/** 渲染一条助手消息的操作行和展开面板。 */
function renderAssistantExtras(message, messageEl) {
  const actions = document.createElement('div');
  actions.className = 'msg__actions';
  messageEl.appendChild(actions);

  if (message.content) {
    const copy = actionButton('copy', '复制', async () => {
      const label = copy.querySelector('span:last-child');
      let text = '已复制';
      try {
        await navigator.clipboard.writeText(message.content);
      } catch {
        // 剪贴板需要安全上下文和授权，失败时如实提示并恢复按钮文字。
        text = '复制失败';
      }
      label.textContent = text;
      setTimeout(() => {
        label.textContent = '复制';
      }, 1500);
    });
    actions.appendChild(copy);
  }

  if (message.demo) {
    const badge = document.createElement('span');
    badge.className = 'action action--demo';
    badge.textContent = '演示数据';
    actions.appendChild(badge);
  }

  if (message.sources || message.prompt || message.timings) {
    renderPanels(message, actions, messageEl);
  }
}

/* ---------- 消息与会话 ---------- */

/** 把消息区滚动到底部；用户向上翻阅时不打断。 */
function scrollToBottom(force = false) {
  const chat = $('chat');
  const nearBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
  if (force || nearBottom) chat.scrollTop = chat.scrollHeight;
}

/** 渲染一条用户消息。 */
function appendUserMessage(text) {
  const article = document.createElement('article');
  article.className = 'msg msg--user';
  const bubble = document.createElement('div');
  bubble.className = 'msg__bubble';
  bubble.textContent = text;
  article.appendChild(bubble);
  $('messages').appendChild(article);
}

/** 创建一条助手消息的空壳；闪烁光标只在真正流式生成时才由 createStreamer 加上。 */
function appendAssistantMessage() {
  const article = document.createElement('article');
  article.className = 'msg msg--assistant';
  const avatar = document.createElement('div');
  avatar.className = 'msg__avatar';
  const body = document.createElement('div');
  body.className = 'msg__body';
  const content = document.createElement('div');
  content.className = 'msg__content';
  body.appendChild(content);
  article.append(avatar, body);
  $('messages').appendChild(article);
  return { article, body, content };
}

/** 拉取会话列表；用序号丢弃过期响应，避免旧列表盖掉新的。 */
async function refreshConversations() {
  const seq = ++state.listSeq;
  try {
    const payload = await api('/api/conversations');
    if (seq !== state.listSeq) return;
    state.conversations = payload.items || [];
    renderHistory();
  } catch (error) {
    if (seq === state.listSeq) {
      $('history').textContent = `读取会话记录失败：${error.message}`;
    }
  }
}

/** 删除一份会话记录；正在生成回答的那一次服务端会拒绝。 */
async function removeConversation(id) {
  try {
    await api(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
  } catch (error) {
    window.alert(`删除失败：${error.message}`);
    return;
  }
  if (state.activeId === id) newConversation();
  await refreshConversations();
}

/** 打开一份服务端保存的会话，按记录重绘全部消息。 */
async function openConversation(id) {
  try {
    const detail = await api(`/api/conversations/${encodeURIComponent(id)}`);
    state.activeId = detail.id;
    state.messages = detail.messages || [];
    renderMessages();
    renderHistory();
  } catch (error) {
    window.alert(`打开会话失败：${error.message}`);
  }
}

/** 按 state.messages 重绘消息区。 */
function renderMessages() {
  const host = $('messages');
  host.replaceChildren();
  $('welcome').hidden = state.messages.length > 0;
  for (const message of state.messages) {
    if (message.role === 'user') {
      appendUserMessage(message.content);
      continue;
    }
    const { body, content } = appendAssistantMessage();
    content.innerHTML = renderMarkdown(message.content || '');
    if (message.error) renderError(body, message.error);
    renderAssistantExtras(message, body);
  }
  scrollToBottom(true);
}

/** 开一段新对话：先只在本地留空，第一条提问时才在服务端建文件。 */
function newConversation() {
  state.activeId = null;
  state.messages = [];
  $('messages').replaceChildren();
  $('welcome').hidden = false;
  renderHistory();
  $('input').focus();
}

/** 返回当前会话 id；还没有记录时先让服务端建一份。 */
async function ensureConversation(question) {
  if (state.activeId) return state.activeId;
  const created = await api('/api/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: question.slice(0, 24) }),
  });
  state.activeId = created.id;
  await refreshConversations();
  return created.id;
}

/** 按创建时间给会话分组，与常见聊天界面的分组方式一致。 */
function groupLabel(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (date.getTime() >= startOfToday) return '今天';
  if (date.getTime() >= startOfToday - 6 * 86400000) return '7 天内';
  return '更早';
}

/** 重绘侧边栏的会话列表。 */
function renderHistory() {
  const host = $('history');
  host.replaceChildren();
  if (!state.conversations.length) {
    const empty = document.createElement('p');
    empty.className = 'history__empty';
    empty.textContent = '还没有对话记录';
    host.appendChild(empty);
    return;
  }
  let currentGroup = null;
  for (const conversation of state.conversations) {
    const label = groupLabel(conversation.updated_at_ms);
    if (label !== currentGroup) {
      currentGroup = label;
      const heading = document.createElement('p');
      heading.className = 'history__label';
      heading.textContent = label;
      host.appendChild(heading);
    }
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'history__item';
    if (conversation.id === state.activeId) item.classList.add('is-active');
    const title = document.createElement('span');
    title.className = 'history__title';
    title.textContent = conversation.title || '新对话';
    if (conversation.parse_error) {
      // 手工改坏的文件仍然列出来，但标一下，不让它悄悄消失。
      title.textContent = `${title.textContent}（无法解析）`;
    }
    const remove = document.createElement('span');
    remove.className = 'history__delete';
    remove.dataset.icon = 'close';
    mountIcons(remove);
    remove.addEventListener('click', async (event) => {
      event.stopPropagation();
      if (state.streaming) return;
      await removeConversation(conversation.id);
    });
    item.append(title, remove);
    item.addEventListener('click', () => {
      // 流式输出期间不切换：回答会被画进已经卸载的消息节点里，凭空消失。
      if (state.streaming || conversation.id === state.activeId) return;
      openConversation(conversation.id);
      $('app').classList.remove('is-sidebar-open');
    });
    host.appendChild(item);
  }
}

/* ---------- 状态卡 ---------- */

/** 把状态接口的返回渲染到侧边栏底部。 */
function renderStatus(status) {
  state.status = status;
  const dot = $('index-dot');
  dot.className = 'index-card__dot';
  if (status.busy || status.building) dot.classList.add('is-busy');
  else if (status.ready) dot.classList.add('is-ready');

  $('demo-badge').hidden = !status.demo;
  $('model-name').textContent = status.models?.generation
    ? status.models.generation.split('/').pop()
    : '未知模型';

  const meta = $('index-meta');
  meta.replaceChildren();
  const lines = [];
  if (status.ready) {
    lines.push(['已入库文件', `${status.sources.length} 个`]);
    lines.push(['片段总数', `${status.chunk_count} 条`]);
    if (status.built_at_ms) {
      lines.push(['构建时间', new Date(status.built_at_ms).toLocaleString('zh-CN')]);
    }
  } else {
    lines.push(['状态', status.reason || '索引不可用']);
  }
  if (status.building) lines.push(['当前', '正在构建索引']);
  else if (status.busy) lines.push(['当前', '正在回答提问']);
  for (const [name, value] of lines) {
    const row = document.createElement('span');
    row.className = 'index-card__line';
    const key = document.createElement('span');
    key.textContent = `${name}：`;
    const strong = document.createElement('b');
    strong.textContent = value;
    row.append(key, strong);
    meta.appendChild(row);
  }
  if (status.last_build) {
    const note = document.createElement('p');
    note.className = 'hint';
    const failed = Object.keys(status.last_build.failed || {}).length;
    note.textContent = `上次构建：成功 ${status.last_build.processed.length}，失败 ${failed}`;
    meta.appendChild(note);
  }

  $('build-button').disabled = Boolean(status.busy || status.building);
  updateSendState();
}

/** 拉取一次索引状态。 */
async function refreshStatus() {
  try {
    renderStatus(await api('/api/status'));
  } catch (error) {
    $('index-meta').textContent = `读取索引状态失败：${error.message}`;
  }
}

/** 根据是否有内容、是否在流式中决定发送按钮状态，并锁住流式期间的会话切换。 */
function updateSendState() {
  const hasText = $('input').value.trim().length > 0;
  const busy = Boolean(state.status?.busy || state.status?.building);
  $('send').disabled = state.streaming || !hasText || busy;
  $('send').hidden = state.streaming;
  $('stop').hidden = !state.streaming;
  $('history').classList.toggle('is-locked', state.streaming);
}

/* ---------- 长期记忆 ---------- */

/** 拉取长期记忆并填充面板；功能关闭时隐藏入口。 */
async function loadMemory() {
  try {
    state.memory = await api('/api/memory');
  } catch (error) {
    state.memory = { enabled: false, content: '', max_chars: 0, error: error.message };
  }
  renderMemory();
}

/** 把长期记忆画到面板里；正在编辑时不覆盖用户已经敲进去的内容。 */
function renderMemory() {
  const memory = state.memory;
  $('memory-toggle').hidden = !memory || !memory.enabled;
  if (!memory || !memory.enabled) {
    $('memory-panel').hidden = true;
    return;
  }
  const textarea = $('memory-text');
  if (document.activeElement !== textarea) textarea.value = memory.content || '';
  updateMemoryCounter();
}

/** 更新“已用 N / 上限”提示，超出上限时标红。 */
function updateMemoryCounter() {
  const memory = state.memory || {};
  const used = $('memory-text').value.length;
  const limit = memory.max_chars || 0;
  const counter = $('memory-counter');
  counter.textContent = limit ? `已用 ${used} / ${limit} 字（超出的部分不会进入提示词）` : '';
  counter.classList.toggle('is-over', limit > 0 && used > limit);
}

/** 保存长期记忆；失败时保留编辑内容并提示原因。 */
async function saveMemory() {
  const button = $('memory-save');
  const status = $('memory-status');
  button.disabled = true;
  status.classList.remove('is-error');
  status.textContent = '正在保存…';
  try {
    state.memory = await api('/api/memory', {
      method: 'PUT',
      body: JSON.stringify({ content: $('memory-text').value }),
    });
    status.textContent = '已保存，下一轮提问就会带上';
  } catch (error) {
    status.textContent = `保存失败：${error.message}`;
    status.classList.add('is-error');
  } finally {
    button.disabled = false;
    renderMemory();
  }
}

/* ---------- 发送 ---------- */

/** 加一条说明文字到回答下方，用于解释等待中的状态。 */
function addHint(context, text) {
  const hint = document.createElement('p');
  hint.className = 'hint';
  hint.textContent = text;
  context.hints.push(hint);
  context.body.appendChild(hint);
  scrollToBottom();
  return hint;
}

/** 清掉等待中的说明文字，第一个词元到达或本轮结束时调用。 */
function clearHints(context) {
  for (const hint of context.hints) hint.remove();
  context.hints.length = 0;
}

/** 处理一条问答事件。 */
function handleChatEvent(name, data, context) {
  const { streamer, message, body } = context;
  if (name === 'queued') {
    if (data.busy && !context.hints.length) {
      // 另有请求正占用模型时会先收到这一条，提示用户不是在卡住。
      addHint(context, '正在等待模型空闲，前面还有一次问答正在进行…');
    }
    return;
  }
  if (name === 'sources') {
    message.sources = Array.isArray(data) ? data : [];
    return;
  }
  if (name === 'prompt') {
    message.prompt = data.prompt || '';
    return;
  }
  if (name === 'token') {
    clearHints(context);
    streamer.push(data.text || '');
    scrollToBottom();
    return;
  }
  if (name === 'done') {
    clearHints(context);
    message.content = streamer.finish(data.answer ?? streamer.text());
    message.sources = Array.isArray(data.sources) ? data.sources : message.sources || [];
    message.timings = data.timings || null;
    message.demo = Boolean(data.demo);
    context.settled = true;
    return;
  }
  if (name === 'error') {
    clearHints(context);
    message.error = data.message || '问答失败';
    message.content = streamer.stop();
    context.settled = true;
  }
}

/** 发送一次提问并处理流式返回。 */
async function sendMessage(text) {
  const question = text.trim();
  if (!question || state.streaming) return;

  let conversationId;
  try {
    conversationId = await ensureConversation(question);
  } catch (error) {
    window.alert(`无法新建会话记录：${error.message}`);
    return;
  }

  state.messages.push({ role: 'user', content: question });
  appendUserMessage(question);
  $('welcome').hidden = true;

  const { body, content } = appendAssistantMessage();
  // 只有这一次回答需要逐词元显示，光标和末尾的纯文本节点都随渲染器创建。
  const streamer = createStreamer(content);
  const message = { role: 'assistant', content: '', sources: [], prompt: '', timings: null };
  state.messages.push(message);

  state.streaming = true;
  state.controller = new AbortController();
  updateSendState();
  $('input').value = '';
  $('input').style.height = 'auto';
  scrollToBottom(true);
  const context = { streamer, message, body, settled: false, hints: [] };
  if (state.status && !state.status.warm && !state.status.demo) {
    // 第一次提问要加载嵌入、重排和生成模型，先说明可能要等多久，避免看起来像卡住。
    addHint(context, '首次提问需要加载嵌入、重排和生成模型，可能要等一到三分钟才开始输出…');
  }

  try {
    await streamRequest(
      '/api/chat',
      {
        question,
        hybrid: state.settings.hybrid,
        rerank: state.settings.rerank,
        conversation_id: conversationId,
      },
      (name, data) => handleChatEvent(name, data, context),
      state.controller.signal,
    );
    if (!context.settled) {
      // 流在没有 done 的情况下结束，说明连接被中断，保留已收到的内容并提示。
      message.content = streamer.stop();
      message.error = '连接已中断，以上是已经收到的内容。';
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      // 模型仍会把这一轮生成完，这里只是停止接收。
      message.content = streamer.stop();
      message.stopped = true;
    } else {
      message.content = streamer.stop();
      message.error = error.message;
    }
  } finally {
    state.streaming = false;
    state.controller = null;
    if (message.error) renderError(body, message.error);
    renderAssistantExtras(message, body);
    if (message.stopped) {
      const note = document.createElement('p');
      note.className = 'hint';
      note.textContent = '已停止接收。模型可能仍在生成这一轮答案，稍后再提问会更快拿到结果。';
      body.appendChild(note);
    }
    updateSendState();
    scrollToBottom();
    refreshStatus();
    // 服务端在发出 done 之前就写好了记录，这里刷新已经能看到新的标题和时间。
    refreshConversations();
  }
}

/* ---------- 初始化 ---------- */

/** 切换深浅主题并记住选择。 */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = $('theme-toggle');
  button.replaceChildren();
  const icon = document.createElement('span');
  icon.className = 'icon';
  icon.dataset.icon = theme === 'dark' ? 'sun' : 'moon';
  button.appendChild(icon);
  mountIcons(button);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // 无法记住主题不影响使用。
  }
}

/** 渲染空态里的建议问题。 */
function renderSuggestions() {
  const host = $('suggestions');
  host.replaceChildren();
  for (const question of SUGGESTIONS) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'suggestion';
    button.innerHTML = '<span class="icon" data-icon="sparkle"></span><span></span>';
    button.lastChild.textContent = question;
    button.addEventListener('click', () => sendMessage(question));
    host.appendChild(button);
  }
  mountIcons(host);
}

/** 绑定界面上的所有交互事件。 */
function bindEvents() {
  const input = $('input');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
    updateSendState();
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('composer').requestSubmit();
    }
  });
  $('composer').addEventListener('submit', (event) => {
    event.preventDefault();
    sendMessage(input.value);
  });
  $('stop').addEventListener('click', () => {
    if (state.controller) state.controller.abort();
  });
  $('new-chat').addEventListener('click', () => {
    if (state.streaming) return;
    newConversation();
    $('app').classList.remove('is-sidebar-open');
  });
  $('theme-toggle').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  const settingsPanel = $('settings-panel');
  $('settings-toggle').addEventListener('click', (event) => {
    event.stopPropagation();
    settingsPanel.hidden = !settingsPanel.hidden;
    $('settings-toggle').setAttribute('aria-expanded', String(!settingsPanel.hidden));
  });

  const memoryPanel = $('memory-panel');
  $('memory-toggle').addEventListener('click', (event) => {
    event.stopPropagation();
    memoryPanel.hidden = !memoryPanel.hidden;
    if (!memoryPanel.hidden) $('memory-text').focus();
  });
  $('memory-save').addEventListener('click', () => saveMemory());
  $('memory-text').addEventListener('input', () => {
    updateMemoryCounter();
    $('memory-status').textContent = '有未保存的修改';
    $('memory-status').classList.remove('is-error');
  });

  document.addEventListener('click', (event) => {
    // 点面板和按钮之外的地方收起浮层；两者都挂在顶栏上。
    if (!settingsPanel.hidden && !settingsPanel.contains(event.target)) settingsPanel.hidden = true;
    const insideMemory = memoryPanel.contains(event.target) || $('memory-toggle').contains(event.target);
    if (!memoryPanel.hidden && !insideMemory) memoryPanel.hidden = true;
  });

  /** 让顶栏开关和输入框上方的开关保持同步。 */
  function syncSetting(key, value) {
    state.settings[key] = value;
    $(`switch-${key}`).checked = value;
    $(`toggle-${key}`).classList.toggle('is-on', value);
    try {
      localStorage.setItem(`energy-rag-${key}`, String(value));
    } catch {
      // 记不住开关状态不影响本次使用。
    }
  }
  for (const key of ['hybrid', 'rerank']) {
    $(`switch-${key}`).addEventListener('change', (event) => syncSetting(key, event.target.checked));
    $(`toggle-${key}`).addEventListener('click', () => syncSetting(key, !state.settings[key]));
    const saved = localStorage.getItem(`energy-rag-${key}`);
    syncSetting(key, saved === null ? true : saved === 'true');
  }

  $('build-button').addEventListener('click', () => startBuild());
  $('sidebar-toggle').addEventListener('click', () => $('app').classList.toggle('is-sidebar-open'));
  $('backdrop').addEventListener('click', () => $('app').classList.remove('is-sidebar-open'));
}

/** 触发一次增量构建，并按事件流更新状态卡。 */
async function startBuild() {
  const button = $('build-button');
  button.disabled = true;
  button.textContent = '正在构建，请勿关闭页面…';
  let failure = null;
  try {
    await streamRequest('/api/build', { incremental: true }, (name, data) => {
      if (name === 'error') failure = data.message || '构建失败';
    });
  } catch (error) {
    failure = error.message;
  } finally {
    button.textContent = '增量构建索引';
    await refreshStatus();
    if (failure) window.alert(`构建失败：${failure}`);
  }
}

/** 页面入口：恢复主题，从服务端读取会话与记忆，读取状态并开始轮询。 */
async function init() {
  let theme = 'light';
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) theme = saved;
    else if (window.matchMedia('(prefers-color-scheme: dark)').matches) theme = 'dark';
  } catch {
    // 读不到偏好时用浅色主题。
  }
  applyTheme(theme);
  renderSuggestions();
  mountIcons(document);
  bindEvents();

  await refreshConversations();
  if (state.conversations.length) {
    await openConversation(state.conversations[0].id);
  } else {
    newConversation();
  }
  await loadMemory();
  await refreshStatus();
  // 流式问答期间状态由问答流程自己刷新，这里只在空闲时轮询。
  state.statusTimer = setInterval(() => {
    if (!state.streaming) refreshStatus();
  }, 5000);
}

init();
