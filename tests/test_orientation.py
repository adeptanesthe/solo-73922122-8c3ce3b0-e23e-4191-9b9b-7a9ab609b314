"""EXIF 方向支持测试：坐标换算、与 PIL 像素变换一致、转正等价性、拒绝时保会话。

运行：python3 tests/test_orientation.py
（也可直接被 pytest 收集；只依赖 pillow/numpy 与 app 本身，不需要 httpx。）
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import orientation as om  # noqa: E402
from app import solver  # noqa: E402
from app.main import SESSION, SAMPLE_DIR, _load_session  # noqa: E402

try:  # pytest 下每个用例后清空全局单会话，避免污染同进程的其他测试模块
    import pytest

    @pytest.fixture(autouse=True)
    def _clear_session_after_test():
        yield
        SESSION.clear()
except ImportError:  # 直接用 python3 运行时不需要 fixture
    pass


# ------------------------------------------------------------ 夹具工具

def png_bytes(arr_or_img, exif_orientation=None) -> bytes:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    if exif_orientation:
        exif = Image.Exif()
        exif[om.ORIENTATION_TAG] = exif_orientation
        img.save(buf, format="PNG", exif=exif.tobytes())
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def make_exif_variant(upright_blobs, upright_boxes, orientation_at):
    """把转正后的 PNG/框按 orientation_at(i) 包装成“存储像素 + EXIF 标记”。"""
    blobs, boxes_out = [], []
    inverse = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,
    }
    for i, (data, boxlist) in enumerate(zip(upright_blobs, upright_boxes)):
        o = orientation_at(i)
        up = Image.open(io.BytesIO(data))
        stored = up if o == 1 else up.transpose(inverse[o])
        sw, sh = stored.size
        blobs.append(png_bytes(stored, o))
        sboxes = []
        for b in boxlist:
            sb = list(b) if o == 1 else om.oriented_box_to_storage(b, sw, sh, o)
            sboxes.append([int(round(v)) for v in sb])
        boxes_out.append(sboxes)
    return blobs, boxes_out


def solve_session(locks=None):
    traj, fail, energy = solver.solve(
        SESSION["energies"], SESSION["feasibles"], locks or {})
    return traj, fail, energy


# ------------------------------------------------------------ 纯函数测试

def test_point_maps_match_pil_transpose():
    """每个方向的点映射必须与 PIL exif_transpose 的真实像素搬运一致。"""
    rng = np.random.RandomState(42)
    w, h = 13, 7
    a = rng.randint(0, 256, (h, w), dtype=np.uint8)
    im = Image.fromarray(a, "L")
    for o in range(1, 9):
        exif = Image.Exif()
        exif[om.ORIENTATION_TAG] = o
        buf = io.BytesIO()
        im.save(buf, format="PNG", exif=exif.tobytes())
        chk = Image.open(io.BytesIO(buf.getvalue()))
        assert om.get_orientation(chk) == o
        out = np.asarray(om.transpose_to_oriented(chk))
        ow, oh = out.shape[1], out.shape[0]
        assert (ow, oh) == om.oriented_size(w, h, o)
        # 随机抽像素点：映射像素中心 (x+.5,y+.5)，存储 (x,y) 的像素必须出现在
        # 转正结果的对应位置（点变换本身描述的是像素边界坐标）
        for y in range(h):
            for x in range(w):
                Xc, Yc = om.storage_to_oriented_point(
                    x + 0.5, y + 0.5, w, h, o)
                X, Y = int(np.floor(Xc)), int(np.floor(Yc))
                assert out[Y, X] == a[y, x], (o, x, y, X, Y)
        # 逆映射往返
        for x in (0, 3, w):
            for y in (0, 2, h):
                X, Y = om.storage_to_oriented_point(float(x), float(y), w, h, o)
                x2, y2 = om.oriented_to_storage_point(X, Y, w, h, o)
                assert abs(x2 - x) < 1e-9 and abs(y2 - y) < 1e-9


def test_box_transform_roundtrip():
    w, h = 200, 96
    box = [120, 20, 170, 60]
    for o in range(1, 9):
        # 先求该方向的存储宽高（5..8 时相对转正框尺寸交换）
        sw, sh = (h, w) if o >= 5 else (w, h)
        sb = om.oriented_box_to_storage(box, sw, sh, o)
        rb = om.storage_box_to_oriented(sb, sw, sh, o)
        for a, b in zip(rb, box):
            assert abs(a - b) < 1e-9, (o, sb, rb, box)


# ------------------------------------------------------------ 行为测试

def test_two_identical_frames_with_mirror_tag():
    """复现题述场景：两帧转正后相同，仅一帧带 EXIF=2 镜像标记。

    忽略 EXIF 时，存储方向的框一个在右、一个被镜像到左，可行区间错开 >8，
    会误报断轨；正确换算后两帧坐标一致，必须可解且 x 相同。
    """
    W, H = 200, 64
    rng = np.random.RandomState(1)
    g = rng.randint(0, 256, (H, W), dtype=np.uint8)
    up_png = png_bytes(g)
    box = [[160, 10, 190, 54]]

    mirrored = np.fliplr(g)
    mirror_png = png_bytes(mirrored, exif_orientation=2)
    mbox = om.oriented_box_to_storage(box[0], W, H, 2)  # [10,10,40,54] 左右

    # 旧逻辑（忽略 EXIF）下的可行性确实相隔很远 —— 证明这是真回归点
    f_raw0 = solver.feasible_mask(box, W, H)
    f_raw1 = solver.feasible_mask([mbox], W, H)
    i0 = np.flatnonzero(f_raw0)
    i1 = np.flatnonzero(f_raw1)
    assert abs(i1.mean() - i0.mean()) > solver.MAX_SHIFT

    _load_session([up_png, mirror_png], {"frames": [box, [mbox]]})
    assert SESSION["w"] == W and SESSION["h"] == H
    assert SESSION["orientations"] == [1, 2]
    traj, fail, energy = solve_session()
    assert traj is not None and fail is None, fail
    assert traj[0] == traj[1]
    # 与“先把图和框转正再上传”完全一致
    _load_session([up_png, up_png], {"frames": [box, box]})
    traj2, _, energy2 = solve_session()
    assert traj == traj2 and energy == energy2


def test_ninety_degree_frame_not_rejected_by_stored_size():
    """EXIF=6 的帧存储尺寸是转正尺寸的交换；不应再因宽度不足/尺寸不一致被拒。"""
    W, H = 128, 64
    g = np.random.RandomState(2).randint(0, 256, (H, W), dtype=np.uint8)
    up = Image.fromarray(g, "L")
    stored = up.transpose(Image.Transpose.ROTATE_90)  # 逆变换，得到 64x128
    assert stored.size == (64, 128)
    data = png_bytes(stored, exif_orientation=6)
    sb = om.oriented_box_to_storage([44, 10, 76, 50], 64, 128, 6)
    _load_session([data], {"frames": [[sb]]})
    assert (SESSION["w"], SESSION["h"]) == (W, H)
    assert SESSION["cw"] == solver.crop_width(H)
    traj, fail, _ = solve_session()
    assert traj is not None and fail is None


def test_mixed_orientation_equivalence_with_samples():
    """用内置样片 A/B 构造全朝向混合夹具，校验轨迹/能量/失败帧完全一致。"""
    d = SAMPLE_DIR / "sample_a"
    if not d.is_dir():
        print("（跳过样片等价测试：未找到 sample_a）")
        return
    paths = sorted(d.glob("frame_*.png"))
    blobs = [p.read_bytes() for p in paths]
    boxes = json.loads((d / "boxes.json").read_text(encoding="utf-8"))["frames"]

    _load_session(blobs, {"frames": boxes})
    base_traj, _, base_energy = solve_session()
    base_locked, _, locked_energy = solve_session(
        {5: base_traj[5], 12: min(SESSION["x_max"], base_traj[12] + 12),
         20: base_traj[20]})

    pattern = lambda i: (i % 8) + 1  # noqa: E731
    mblobs, mboxes = make_exif_variant(blobs, boxes, pattern)
    _load_session(mblobs, {"frames": mboxes})
    assert SESSION["w"] == 512 and SESSION["h"] == 288
    t, fail, energy = solve_session()
    assert fail is None and t == base_traj and energy == base_energy

    lt, lfail, lenergy = solve_session(
        {5: base_traj[5], 12: min(SESSION["x_max"], base_traj[12] + 12),
         20: base_traj[20]})
    assert lt == base_locked and lfail is None and lenergy == locked_energy

    # 不可解样片 B：混合朝向后首个接不上的帧仍为 12
    db = SAMPLE_DIR / "sample_b"
    if db.is_dir():
        pb = sorted(db.glob("frame_*.png"))
        bb = [p.read_bytes() for p in pb]
        bxb = json.loads((db / "boxes.json").read_text(encoding="utf-8"))["frames"]
        _load_session(bb, {"frames": bxb})
        _, fail_b, _ = solve_session()
        mbb, mbxb = make_exif_variant(bb, bxb, pattern)
        _load_session(mbb, {"frames": mbxb})
        _, fail_m, _ = solve_session()
        assert fail_b == 12 and fail_m == fail_b


def test_rejected_load_keeps_session():
    """加载被拒（框数不符 / 原框越界）时，全局 SESSION 必须仍是旧素材。"""
    W, H = 96, 64
    g = np.zeros((H, W), dtype=np.uint8)
    good = png_bytes(g)
    _load_session([good, good], {"frames": [[[0, 0, 10, 10]], []]})
    old_png = SESSION["png"]
    old_traj, _, _ = solve_session()

    # 框帧数不符
    try:
        _load_session([good, good], {"frames": [[]]})
        raise AssertionError("应当拒绝框帧数不符")
    except Exception as e:
        assert getattr(e, "status_code", None) == 400
    assert SESSION["png"] is old_png
    t, _, _ = solve_session()
    assert t == old_traj

    # 原框超出“存储方向”尺寸
    tagged = png_bytes(Image.fromarray(g).transpose(Image.Transpose.ROTATE_90), 6)
    try:
        _load_session([tagged], {"frames": [[[0, 0, 999, 9]]]})
        raise AssertionError("应当拒绝越界原框")
    except Exception as e:
        assert getattr(e, "status_code", None) == 400
    assert SESSION["png"] is old_png  # 仍是旧的两帧素材


def test_frame_bytes_are_kept_verbatim():
    g = np.random.RandomState(3).randint(0, 256, (64, 96), dtype=np.uint8)
    raw = png_bytes(g, exif_orientation=8)
    _load_session([raw], {"frames": [[]]})
    assert SESSION["png"][0] == raw


if __name__ == "__main__":
    test_point_maps_match_pil_transpose()
    print("点映射 vs PIL 像素搬运：一致")
    test_box_transform_roundtrip()
    print("框坐标往返：一致")
    test_two_identical_frames_with_mirror_tag()
    print("镜像标记双帧误报断轨：已修复且与转正上传等价")
    test_ninety_degree_frame_not_rejected_by_stored_size()
    print("90° 方向帧尺寸：按转正尺寸通过")
    test_mixed_orientation_equivalence_with_samples()
    print("样片 A/B 全朝向混合：轨迹/能量/失败帧完全等价")
    test_rejected_load_keeps_session()
    print("拒绝加载：旧 SESSION 完整保留")
    test_frame_bytes_are_kept_verbatim()
    print("原始 PNG 字节：原样保留")
    print("全部通过")
