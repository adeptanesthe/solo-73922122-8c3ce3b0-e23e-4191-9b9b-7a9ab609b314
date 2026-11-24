"use strict";

const $ = (id) => document.getElementById(id);

let meta = null;        // {n,w,h,cw,x_max,boxes,feasible,orientations,stored_sizes}
let frames = [];        // 每帧“转正后”的离屏 canvas（显示/叠加都在转正坐标系）
let trajectory = null;  // 求解成功后的逐帧 x
let locks = {};         // 帧号 -> 锁定 x
let cur = 0;
let shownX = 0;         // 当前帧画面上显示的裁切窗横坐标（转正坐标）
let failFrame = null;   // 求解失败时最早接不上的帧
let dragging = false;
let grabDX = 0;
let loadToken = 0;      // 会话版本，用于帧图片缓存刷新

// EXIF Orientation 1..8：存储像素坐标 (x,y) -> 转正坐标 (X,Y) 的 canvas 矩阵参数。
// setTransform(a,b,c,d,e,f) 对应 X=a*x+c*y+e, Y=b*x+d*y+f；w/h 为存储宽高。
function exifMatrix(o, w, h) {
  switch (o) {
    case 2: return { a: -1, b: 0, c: 0, d: 1, e: w, f: 0 };        // 水平镜像
    case 3: return { a: -1, b: 0, c: 0, d: -1, e: w, f: h };       // 180°
    case 4: return { a: 1, b: 0, c: 0, d: -1, e: 0, f: h };        // 垂直镜像
    case 5: return { a: 0, b: 1, c: 1, d: 0, e: 0, f: 0 };         // 转置
    case 6: return { a: 0, b: 1, c: -1, d: 0, e: h, f: 0 };        // 顺时针 90°
    case 7: return { a: 0, b: -1, c: -1, d: 0, e: h, f: w };        // 副对角翻转
    case 8: return { a: 0, b: -1, c: 1, d: 0, e: 0, f: w };        // 逆时针 90°
    default: return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
  }
}

const ORIENTATION_LABEL = {
  1: "正常", 2: "水平镜像", 3: "旋转180°", 4: "垂直镜像",
  5: "转置", 6: "顺时针90°", 7: "副对角翻转", 8: "逆时针90°",
};

// ------------------------------------------------------------ 状态提示

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + (kind || "");
}

// ------------------------------------------------------------ 数据加载

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return handle(r);
}

async function handle(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function naturalSort(files) {
  return [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

async function doUpload() {
  const fs = naturalSort($("fileFrames").files);
  const bf = $("fileBoxes").files[0];
  if (!fs.length || !bf) { setStatus("请选择 PNG 帧和必留框 JSON", "err"); return; }
  const fd = new FormData();
  for (const f of fs) fd.append("frames", f, f.name);
  fd.append("boxes", bf, bf.name);
  setStatus("上传中…");
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("上传失败：" + e.message, "err"); }
}

async function loadSample(name) {
  setStatus("载入样片…");
  try {
    const r = await fetch("/api/load_sample?name=" + name, { method: "POST" });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("载入样片失败：" + e.message, "err"); }
}

async function afterLoad(m) {
  meta = m;
  locks = {};
  trajectory = null;
  failFrame = null;
  cur = 0;
  shownX = 0;
  loadToken++;
  frames = await Promise.all([...Array(m.n).keys()].map(loadFrame));
  $("slider").max = m.n - 1;
  $("slider").value = 0;
  $("cvMain").width = m.w; $("cvMain").height = m.h;
  $("cvPreview").width = m.cw; $("cvPreview").height = m.h;
  $("viewer").hidden = false;
  $("btnSolve").disabled = false;
  const mixed = new Set(m.orientations || [1]).size > 1;
  setStatus(`已载入 ${m.n} 帧（转正后 ${m.w}×${m.h}，裁切窗 ${m.cw}×${m.h}，x ∈ [0, ${m.x_max}]`
    + (mixed ? "，含混合 EXIF 朝向" : "") + "），求解中…");
  await solve();
}

// 从 PNG 字节中移除 eXIf 块（仅在浏览器内存里做）。这样任何浏览器都只会解码
// 出“文件存储方向”的原始像素，随后由 exifMatrix 精确转正一次，避免不同浏览器
// 对 PNG EXIF / createImageBitmap imageOrientation 行为不一致导致的二次朝向。
// 服务端 /api/frame/{i}.png 仍原样返回上传字节，这里不改服务端数据。
function stripPngExif(buffer) {
  const u8 = new Uint8Array(buffer);
  const isPng = u8.length >= 8 &&
    u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4e && u8[3] === 0x47;
  if (!isPng) return u8;
  const parts = [u8.subarray(0, 8)];
  let off = 8;
  let removed = false;
  while (off + 8 <= u8.length) {
    const len = new DataView(u8.buffer, u8.byteOffset + off, 4).getUint32(0);
    const type = String.fromCharCode(u8[off + 4], u8[off + 5], u8[off + 6], u8[off + 7]);
    const total = 12 + len; // length(4)+type(4)+data(len)+crc(4)
    if (off + total > u8.length) break;
    if (type !== "eXIf") parts.push(u8.subarray(off, off + total));
    else removed = true;
    off += total;
    if (type === "IEND") break;
  }
  if (!removed) return u8;
  const totalLen = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(totalLen);
  let pos = 0;
  for (const p of parts) { out.set(p, pos); pos += p.length; }
  return out;
}

// 解码剥掉 eXIf 的 PNG，得到“存储方向”原始像素源。优先 createImageBitmap；
// 老浏览器退回到 <img> + objectURL（没有 eXIf 块就不会被自动朝向）。
async function decodeRawPng(blob) {
  if (typeof createImageBitmap === "function") {
    try {
      return { src: await createImageBitmap(blob, { imageOrientation: "none" }), close(b) { b.close && b.close(); } };
    } catch (e) {
      return { src: await createImageBitmap(blob), close(b) { b.close && b.close(); } };
    }
  }
  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error("PNG 解码失败"));
      im.src = url;
    });
    return { src: img, close() { URL.revokeObjectURL(url); } };
  } catch (e) {
    URL.revokeObjectURL(url);
    throw e;
  }
}

// 取回与上传字节一致的原始 PNG，剥掉内存中的 eXIf 后解码出存储像素，再显式按
// meta.orientations 转正到离屏 canvas。叠加层坐标全部使用转正坐标系，保证对齐。
async function loadFrame(i) {
  const resp = await fetch("/api/frame/" + i + ".png?t=" + loadToken);
  if (!resp.ok) throw new Error("帧 " + i + " 加载失败");
  const original = await resp.arrayBuffer();
  const rawPng = stripPngExif(original);
  const blob = new Blob([rawPng], { type: "image/png" });
  const decoded = await decodeRawPng(blob);
  const bmp = decoded.src;

  const o = (meta.orientations && meta.orientations[i]) || 1;
  let sw, sh;
  if (meta.stored_sizes && meta.stored_sizes[i]) {
    [sw, sh] = meta.stored_sizes[i];
  } else if (o >= 5) {
    sw = meta.h; sh = meta.w;
  } else {
    sw = meta.w; sh = meta.h;
  }

  const cv = document.createElement("canvas");
  cv.width = meta.w;
  cv.height = meta.h;
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;

  // 双保险：若解码结果尺寸已是转正尺寸（说明仍被浏览器自动转正，仅 90° 类可从
  // 尺寸识别），直接绘制，避免二次旋转。正常路径下 eXIf 已被剥离，不会进这里。
  const browserOriented = o >= 5
    && bmp.width === meta.w && bmp.height === meta.h
    && (bmp.width !== sw || bmp.height !== sh);

  if (o === 1 || browserOriented) {
    ctx.drawImage(bmp, 0, 0, meta.w, meta.h);
  } else {
    const m = exifMatrix(o, sw, sh);
    ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
    ctx.drawImage(bmp, 0, 0, sw, sh);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }
  decoded.close(bmp);
  return cv;
}

// ------------------------------------------------------------ 求解

async function solve() {
  if (!meta) return;
  try {
    const r = await postJSON("/api/solve", { locks });
    if (r.ok) {
      trajectory = r.trajectory;
      failFrame = null;
      shownX = trajectory[cur];
      setStatus(`求解成功，总能量 ${Math.round(r.energy).toLocaleString()}。可拖动裁切窗并锁定。`, "ok");
    } else {
      trajectory = null;
      failFrame = r.fail_frame;
      cur = failFrame;
      $("slider").value = cur;
      shownX = defaultX(cur);
      setStatus(r.message + "——已撤下竖屏预览，可调整锁定后重新求解。", "err");
    }
  } catch (e) {
    setStatus("求解失败：" + e.message, "err");
  }
  render();
}

// ------------------------------------------------------------ 绘制

function render() {
  if (!meta) return;
  const cv = $("cvMain"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(frames[cur], 0, 0);

  // 必留框（红）
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#e33";
  for (const b of meta.boxes[cur] || []) {
    ctx.strokeRect(b[0] + 0.5, b[1] + 0.5, b[2] - b[0] - 1, b[3] - b[1] - 1);
  }

  const failed = trajectory === null;
  $("previewPane").style.visibility = failed ? "hidden" : "visible";

  // 裁切窗（锁定绿 / 未锁定蓝；失败时也可拖动以便重新锁定）
  const locked = locks[cur] !== undefined;
  ctx.strokeStyle = locked ? "#0a0" : "#28f";
  ctx.lineWidth = 2;
  ctx.strokeRect(shownX + 1, 1, meta.cw - 2, meta.h - 2);
  ctx.fillStyle = locked ? "#0a0" : "#28f";
  ctx.font = "12px sans-serif";
  ctx.fillText((locked ? "🔒 " : "") + "x=" + shownX, Math.min(shownX + 4, meta.w - 60), 14);

  if (!failed) {
    // 竖屏预览
    const pv = $("cvPreview").getContext("2d");
    pv.imageSmoothingEnabled = false;
    pv.drawImage(frames[cur], shownX, 0, meta.cw, meta.h, 0, 0, meta.cw, meta.h);
  } else {
    ctx.fillStyle = "rgba(200,0,0,0.85)";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("第 " + cur + " 帧起无法接续", 8, 18);
  }

  $("frameLabel").textContent = `第 ${cur + 1}/${meta.n} 帧`;
  const oriEl = $("oriLabel");
  if (oriEl) {
    const o = (meta.orientations && meta.orientations[cur]) || 1;
    oriEl.textContent = `EXIF ${o}（${ORIENTATION_LABEL[o] || "?"}）`;
  }
  $("btnLock").textContent = locks[cur] !== undefined ? "更新锁定为当前位置" : "锁定本帧位置";
  renderLocks();
}

function renderLocks() {
  const el = $("locks");
  const keys = Object.keys(locks).map(Number).sort((a, b) => a - b);
  el.innerHTML = keys.length
    ? "已锁定：" + keys.map((i) => `<span class="chip">帧 ${i + 1} → x=${locks[i]}</span>`).join("")
    : "未锁定任何帧。";
}

// ------------------------------------------------------------ 交互

function clampX(x) {
  return Math.max(0, Math.min(meta.x_max, Math.round(x)));
}

function defaultX(i) {
  if (locks[i] !== undefined) return locks[i];
  const f = meta.feasible[i];
  return f ? Math.round((f[0] + f[1]) / 2) : 0;
}

function gotoFrame(i) {
  cur = Math.max(0, Math.min(meta.n - 1, i));
  $("slider").value = cur;
  shownX = trajectory ? trajectory[cur] : defaultX(cur);
  render();
}

$("btnUpload").onclick = doUpload;
$("btnSampleA").onclick = () => loadSample("a");
$("btnSampleB").onclick = () => loadSample("b");
$("btnSampleC").onclick = () => loadSample("c");
$("btnSolve").onclick = solve;
$("btnPrev").onclick = () => gotoFrame(cur - 1);
$("btnNext").onclick = () => gotoFrame(cur + 1);
$("slider").oninput = (e) => gotoFrame(+e.target.value);

$("btnLock").onclick = () => { locks[cur] = shownX; solve(); };
$("btnUnlock").onclick = () => { delete locks[cur]; solve(); };

const cvMain = $("cvMain");

// 把鼠标事件换算成 canvas 像素坐标（兼容边框与 CSS 缩放），保证拖动/锁定对齐
function eventCanvasX(e) {
  const rect = cvMain.getBoundingClientRect();
  const xCss = e.clientX - rect.left - cvMain.clientLeft;
  const scale = cvMain.clientWidth ? cvMain.width / cvMain.clientWidth : 1;
  return xCss * scale;
}

cvMain.addEventListener("mousedown", (e) => {
  if (!meta) return;
  dragging = true;
  grabDX = eventCanvasX(e) - shownX;
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  shownX = clampX(eventCanvasX(e) - grabDX);
  render();
});
window.addEventListener("mouseup", async () => {
  if (!dragging) return;
  dragging = false;
  if (locks[cur] !== undefined && locks[cur] !== shownX) {
    locks[cur] = shownX;   // 拖动已锁定帧 = 改锁定位置，前后未锁定帧跟着重新求解
    await solve();
  }
});
