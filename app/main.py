"""横拍转竖屏 9:16 裁切 —— FastAPI 后端。

流程：上传有序 PNG 帧 + 逐帧必留框 JSON -> 求解最优裁切轨迹 ->
逐帧查看裁切结果，可拖动/锁定任一帧裁切窗后重新求解。
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import orientation as ori_mod
from . import solver

MAX_FRAMES = 80
MAX_SIDE = 512

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", BASE_DIR.parent / "sample"))

app = FastAPI(title="横拍转竖屏裁切")

# 单会话演示：服务端只保存一份当前工程
SESSION: dict = {}


# ---------------------------------------------------------------- 工具

def _open_png(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"无法解码 PNG：{exc}") from exc
    if img.format != "PNG":
        raise HTTPException(400, "只接受 PNG 帧")
    return img


def _read_png_record(data: bytes):
    """读取一帧 PNG，返回 (转正后灰度数组, EXIF方向, 存储宽, 存储高)。

    /api/frame 仍直接回传原始字节，因此这里的转正只作用于求解/校验，
    不会改动 SESSION["png"] 中保存的上传字节。
    """
    img = _open_png(data)
    o = ori_mod.get_orientation(img)
    stored_w, stored_h = img.size
    img = ori_mod.transpose_to_oriented(img)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    return gray, o, int(stored_w), int(stored_h)


def _read_png(data: bytes) -> np.ndarray:
    """向后兼容的简写：只返回按 EXIF 转正后的灰度数组。"""
    return _read_png_record(data)[0]


def _parse_boxes(payload) -> list[list[list[float]]]:
    """接受 {"frames": [...]} 或裸数组；每帧为 [[x0,y0,x1,y1], ...]。"""
    if isinstance(payload, dict):
        payload = payload.get("frames")
    if not isinstance(payload, list):
        raise HTTPException(400, "必留框 JSON 须为数组或含 frames 数组的对象")
    frames = []
    for i, entry in enumerate(payload):
        if entry is None:
            entry = []
        if not isinstance(entry, list):
            raise HTTPException(400, f"第 {i} 帧的必留框不是数组")
        boxes = []
        for b in entry:
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                raise HTTPException(400, f"第 {i} 帧存在非法必留框：{b!r}")
            try:
                boxes.append([float(v) for v in b])
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"第 {i} 帧必留框坐标非数字：{b!r}") from exc
        frames.append(boxes)
    return frames


def _load_session(png_blobs: list[bytes], boxes_payload) -> dict:
    if not 1 <= len(png_blobs) <= MAX_FRAMES:
        raise HTTPException(400, f"帧数须在 1..{MAX_FRAMES} 之间，收到 {len(png_blobs)}")

    # 1) 解码并按每帧 EXIF Orientation 转正；同时保留存储方向尺寸
    records = [_read_png_record(b) for b in png_blobs]
    grays = [r[0] for r in records]
    orientations = [r[1] for r in records]
    stored_sizes = [(r[2], r[3]) for r in records]

    # 2) 必留框仍是“文件存储方向”的半开像素坐标：先按原框校验
    frames_boxes_stored = _parse_boxes(boxes_payload)
    if len(frames_boxes_stored) != len(png_blobs):
        raise HTTPException(
            400,
            f"必留框帧数 {len(frames_boxes_stored)} 与 PNG 帧数 {len(png_blobs)} 不一致")
    for i, boxes in enumerate(frames_boxes_stored):
        sw, sh = stored_sizes[i]
        for b in boxes:
            x0, y0, x1, y1 = b
            if not (0 <= x0 < x1 <= sw and 0 <= y0 < y1 <= sh):
                raise HTTPException(
                    400,
                    f"第 {i} 帧必留框 {b} 超出文件存储方向画面 {sw}x{sh} 或非法")

    # 3) 按各帧 EXIF 方向把框换算到转正坐标系
    frames_boxes = [
        [ori_mod.storage_box_to_oriented(b, stored_sizes[i][0],
                                         stored_sizes[i][1], orientations[i])
         for b in boxes]
        for i, boxes in enumerate(frames_boxes_stored)
    ]

    # 4) 整段素材按“转正后的尺寸”校验一致性与规格
    H, W = grays[0].shape
    for i, g in enumerate(grays):
        if g.shape != (H, W):
            raise HTTPException(
                400,
                f"第 {i} 帧按 EXIF {orientations[i]} 转正后尺寸 "
                f"{g.shape[1]}x{g.shape[0]} 与首帧转正尺寸 {W}x{H} 不一致")
    if W > MAX_SIDE or H > MAX_SIDE:
        raise HTTPException(400, f"转正后帧边长至多为 {MAX_SIDE} 像素，收到 {W}x{H}")
    if H % 16 != 0:
        raise HTTPException(400, f"转正后帧高须为 16 的倍数，收到 {H}")
    cw = solver.crop_width(H)
    if cw > W:
        raise HTTPException(400, f"转正后原图宽 {W} 放不下 9:16 竖屏窗口（需 ≥ {cw}）")

    energies = [solver.window_energies(g) for g in grays]
    feasibles = [solver.feasible_mask(b, W, H) for b in frames_boxes]
    X = W - cw + 1
    session = {
        "n": len(grays), "w": W, "h": H, "cw": cw, "x_max": X - 1,
        "png": png_blobs,
        "boxes": frames_boxes,
        "boxes_stored": frames_boxes_stored,
        "orientations": orientations,
        "stored_sizes": stored_sizes,
        "energies": energies, "feasibles": feasibles,
    }
    # 全部校验/计算成功后才替换全局会话：上传被拒时旧素材、锁定与预览不受影响
    SESSION.clear()
    SESSION.update(session)
    return session


def _meta(session: dict) -> dict:
    feas = []
    for f in session["feasibles"]:
        idx = np.flatnonzero(f)
        feas.append([int(idx[0]), int(idx[-1])] if idx.size else None)
    return {
        "n": session["n"], "w": session["w"], "h": session["h"],
        "cw": session["cw"], "x_max": session["x_max"],
        # 以下 boxes 为换算到转正方向后的框，供网页叠加对齐
        "boxes": session["boxes"],
        "feasible": feas,
        # 每帧 EXIF 方向及文件存储方向尺寸，供前端把原始字节手动转正
        "orientations": session.get("orientations", [1] * session["n"]),
        "stored_sizes": session.get(
            "stored_sizes", [(session["w"], session["h"])] * session["n"]),
    }


# ---------------------------------------------------------------- 接口

@app.post("/api/upload")
async def api_upload(frames: list[UploadFile] = File(...),
                     boxes: UploadFile = File(...)):
    png_blobs = [await f.read() for f in frames]
    try:
        payload = json.loads((await boxes.read()).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"必留框 JSON 解析失败：{exc}") from exc
    session = _load_session(png_blobs, payload)
    return _meta(session)


@app.post("/api/load_sample")
async def api_load_sample(name: str = "a"):
    # 内置 a/b/c/d；也允许载入 sample_<安全名称> 下的其他演示素材
    # （如外部测试夹具），但绝不允许路径分隔符或 ".."。
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name or ""):
        raise HTTPException(400, "非法样片名")
    d = SAMPLE_DIR / f"sample_{name}"
    if not d.is_dir():
        raise HTTPException(404, f"样片不存在：{name}")
    paths = sorted(d.glob("frame_*.png"))
    boxes_path = d / "boxes.json"
    if not paths or not boxes_path.is_file():
        raise HTTPException(404, "样片文件不完整")
    png_blobs = [p.read_bytes() for p in paths]
    payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    session = _load_session(png_blobs, payload)
    return _meta(session)


class SolveRequest(BaseModel):
    locks: dict[int, int] = {}


@app.post("/api/solve")
async def api_solve(req: SolveRequest):
    if not SESSION:
        raise HTTPException(400, "请先上传帧或载入样片")
    locks = {}
    for k, v in (req.locks or {}).items():
        i = int(k)
        if not 0 <= i < SESSION["n"]:
            raise HTTPException(400, f"锁定帧号越界：{i}")
        locks[i] = int(v)
    traj, fail, total = solver.solve(SESSION["energies"], SESSION["feasibles"], locks)
    if traj is None:
        return {"ok": False, "fail_frame": fail,
                "message": f"从首帧起第 {fail} 帧无法接续，完整轨迹接不出来"}
    return {"ok": True, "trajectory": traj, "energy": total}


@app.get("/api/frame/{idx}.png")
async def api_frame(idx: int):
    if not SESSION:
        raise HTTPException(400, "请先上传帧或载入样片")
    if not 0 <= idx < SESSION["n"]:
        raise HTTPException(404, "帧号越界")
    return Response(content=SESSION["png"][idx], media_type="image/png")


@app.get("/api/health")
async def api_health():
    return {"ok": True, "loaded": bool(SESSION)}


# 静态页面
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
