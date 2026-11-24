"""生成演示样片：sample_a（可解）、sample_b（不可解演示）、sample_c（混合 EXIF）。

样片：24 帧 512x288（高为 16 的倍数，裁切窗 162x288）。
画面：渐变底 + 静态噪点纹理 + 棋盘格高纹理区 + 移动的"工件"亮块。
必留框逐帧跟踪亮块。样片 B 在第 12 帧让工件瞬移 40px，使轨迹接不上。
样片 C 把样片 A 的若干帧做 EXIF 方向变换（像素按逆变换存储 + Orientation
标记，框换算到存储方向），转正后与 A 完全一致，用于演示混合朝向取景。

用法：python3 scripts/make_sample.py [输出目录，默认 sample/]
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import orientation as ori_mod  # noqa: E402

W, H = 512, 288
N = 24
OBJ = 44          # 工件边长
X0, V = 20, 7     # 工件起始 x 与每帧速度
OY = 120          # 工件 y

# sample_c 的逐帧 EXIF Orientation：覆盖 1..8，重点演示镜像(2)与 90°(6/8)
MIXED_ORIENTATIONS = [
    1, 2, 1, 6, 1, 8, 1, 3,
    1, 2, 1, 6, 1, 8, 1, 7,
    1, 4, 1, 5, 1, 2, 1, 6,
]

# Orientation 修正变换的逆变换（用于生成“存储像素”）
_INVERSE_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def base_frame(rng):
    yy, xx = np.mgrid[0:H, 0:W]
    g = 60 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
    g += rng.normal(0, 18, (H, W))            # 静态噪点（同一随机场）
    # 棋盘格高纹理条带
    chk = ((xx // 6 + yy // 6) % 2) * 90
    band = (xx >= 300) & (xx < 380)
    g = np.where(band, 40 + chk, g)
    return np.clip(g, 0, 255)


def make(out: Path, teleport_at=None):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    bg = base_frame(rng)
    boxes_all = []
    for i in range(N):
        ox = X0 + V * i
        if teleport_at is not None and i >= teleport_at:
            ox += 160                           # 瞬移 -> 相邻帧窗口可行区间错开 >8px，必接不上
        oy = OY + int(6 * np.sin(i / 3.0))
        frame = bg.copy()
        # 暗色圆（干扰物，缓慢左移）
        cx = 430 - 3 * i
        yy, xx = np.mgrid[0:H, 0:W]
        frame[(xx - cx) ** 2 + (yy - 200) ** 2 < 26 ** 2] = 30
        img = Image.fromarray(frame.astype(np.uint8), "L").convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([ox, oy, ox + OBJ - 1, oy + OBJ - 1], fill=(235, 220, 180))
        d.line([ox, oy + OBJ // 2, ox + OBJ - 1, oy + OBJ // 2], fill=(120, 60, 40), width=3)
        img.save(out / f"frame_{i:03d}.png")
        boxes_all.append([[ox, oy, ox + OBJ, oy + OBJ]])
    (out / "boxes.json").write_text(
        json.dumps({"frames": boxes_all}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out}: {N} 帧 {W}x{H}")


def make_mixed(src: Path, out: Path):
    """把已生成的可解样片 src 转成混合 EXIF 朝向版本 out（sample_c）。

    - 像素按 Orientation 修正变换的逆变换存储，并写入 eXIf Orientation；
    - 必留框换算到“文件存储方向”；
    - 生成后逐帧断言：EXIF 转正结果与源帧像素一致、框往返一致。
    """
    meta = json.loads((src / "boxes.json").read_text(encoding="utf-8"))
    src_boxes = meta["frames"] if isinstance(meta, dict) else meta
    paths = sorted(src.glob("frame_*.png"))
    assert len(paths) == N and len(src_boxes) == N

    out.mkdir(parents=True, exist_ok=True)
    stored_boxes_all = []
    for i, (p, boxes) in enumerate(zip(paths, src_boxes)):
        o = MIXED_ORIENTATIONS[i]
        up = Image.open(p).convert("RGB")
        uw, uh = up.size
        if o == 1:
            stored = up
        else:
            stored = up.transpose(_INVERSE_TRANSPOSE[o])
        sw, sh = stored.size

        # 写带 EXIF Orientation 的 PNG
        exif = Image.Exif()
        exif[ori_mod.ORIENTATION_TAG] = o
        buf = io.BytesIO()
        stored.save(buf, format="PNG", exif=exif.tobytes())
        (out / f"frame_{i:03d}.png").write_bytes(buf.getvalue())

        sboxes = []
        for b in boxes:
            sb = b if o == 1 else ori_mod.oriented_box_to_storage(b, sw, sh, o)
            sb = [int(round(v)) for v in sb]
            sboxes.append(sb)
            rt = ori_mod.storage_box_to_oriented(sb, sw, sh, o)
            assert all(abs(a - c) <= 1e-6 for a, c in zip(rt, b)), (i, b, sb, rt)
        stored_boxes_all.append(sboxes)

        # 像素级断言：读回 + exif_transpose 必须等于源帧
        chk = Image.open(io.BytesIO(buf.getvalue()))
        assert ori_mod.get_orientation(chk) == o
        back = ori_mod.transpose_to_oriented(chk).convert("RGB")
        assert back.size == (uw, uh)
        assert np.array_equal(np.asarray(back), np.asarray(up)), \
            f"frame {i} orientation {o} 往返像素不一致"

    (out / "boxes.json").write_text(
        json.dumps({"frames": stored_boxes_all}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"{out}: {N} 帧，混合 EXIF 朝向 {sorted(set(MIXED_ORIENTATIONS))}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample")
    make(root / "sample_a")
    make(root / "sample_b", teleport_at=12)
    make_mixed(root / "sample_a", root / "sample_c")


if __name__ == "__main__":
    main()
