"""EXIF 方向（Orientation tag 274）支持。

约定：
- “存储方向 / 文件方向”：PNG 字节里实际像素的坐标系，必留框 JSON 用的就是它，
  为半开像素区间 [x0, x1) × [y0, y1)。
- “转正方向 / 显示方向”：按 EXIF Orientation 把像素转正后的坐标系，
  求解器、meta 中的 w/h/boxes、网页画布与锁定 x 都用它。

EXIF Orientation 1..8 与 PIL ``ImageOps.exif_transpose`` 的变换一一对应：

==  ==============================  ==============  ==============
值  含义                            PIL transpose   存储尺寸 -> 转正尺寸
==  ==============================  ==============  ==============
1   正常                            -               (w, h)
2   水平镜像                        FLIP_LEFT_RIGHT (w, h)
3   旋转 180°                       ROTATE_180      (w, h)
4   垂直镜像                        FLIP_TOP_BOTTOM (w, h)
5   主对角线翻转（转置）            TRANSPOSE       (h, w)
6   顺时针 90°                      ROTATE_270      (h, w)
7   副对角线翻转                    TRANSVERSE      (h, w)
8   逆时针 90°（顺时针 270°）       ROTATE_90       (h, w)
==  ==============================  ==============  ==============

所有点变换都满足：存储坐标 (x, y) -> 转正坐标 (X, Y)，
其中 w/h 始终指**存储**像素宽高。
"""

from __future__ import annotations

from PIL import Image, ImageOps

ORIENTATION_TAG = 0x0112  # 274

# 90° 旋转 / 对角翻转会交换宽高
SWAP_ORIENTATIONS = (5, 6, 7, 8)

VALID_ORIENTATIONS = frozenset(range(1, 9))


def normalize_orientation(value) -> int:
    """把任意 EXIF orientation 值归一到 1..8；缺失或非法时按 1（不处理）。"""
    try:
        o = int(value)
    except (TypeError, ValueError):
        return 1
    return o if o in VALID_ORIENTATIONS else 1


def get_orientation(img: Image.Image) -> int:
    """读取一张已打开图片的 EXIF Orientation（1..8）。"""
    try:
        exif = img.getexif()
        value = exif.get(ORIENTATION_TAG, 1) if exif else 1
    except Exception:  # noqa: BLE001 — 损坏/无法解析的 EXIF 不应影响取景
        value = 1
    return normalize_orientation(value)


def oriented_size(stored_w: int, stored_h: int, orientation: int) -> tuple[int, int]:
    """存储宽高 -> 转正宽高。"""
    orientation = normalize_orientation(orientation)
    if orientation in SWAP_ORIENTATIONS:
        return stored_h, stored_w
    return stored_w, stored_h


def transpose_to_oriented(img: Image.Image) -> Image.Image:
    """返回按 EXIF orientation 转正后的图片（像素排列被真正变换）。"""
    orientation = get_orientation(img)
    if orientation == 1:
        return img
    return ImageOps.exif_transpose(img)


def storage_to_oriented_point(x: float, y: float,
                             w: int, h: int, o: int) -> tuple[float, float]:
    """存储像素边界坐标 -> 转正坐标。w/h 为存储宽高。"""
    o = normalize_orientation(o)
    if o == 1:
        return x, y
    if o == 2:
        return w - x, y
    if o == 3:
        return w - x, h - y
    if o == 4:
        return x, h - y
    if o == 5:       # 主对角线翻转：(x, y) -> (y, x)
        return y, x
    if o == 6:       # 顺时针 90°：(x, y) -> (h - y, x)
        return h - y, x
    if o == 7:       # 副对角线翻转：(x, y) -> (h - y, w - x)
        return h - y, w - x
    # o == 8：逆时针 90°：(x, y) -> (y, w - x)
    return y, w - x


def oriented_to_storage_point(X: float, Y: float,
                             w: int, h: int, o: int) -> tuple[float, float]:
    """转正坐标 -> 存储像素边界坐标（上面变换的逆）。w/h 仍为存储宽高。"""
    o = normalize_orientation(o)
    if o == 1:
        return X, Y
    if o == 2:
        return w - X, Y
    if o == 3:
        return w - X, h - Y
    if o == 4:
        return X, h - Y
    if o == 5:
        return Y, X
    if o == 6:       # X = h - y, Y = x  =>  x = Y, y = h - X
        return Y, h - X
    if o == 7:       # X = h - y, Y = w - x  =>  x = w - Y, y = h - X
        return w - Y, h - X
    # o == 8：X = y, Y = w - x  =>  x = w - Y, y = X
    return w - Y, X


def _box_from_points(p0, p1) -> list[float]:
    (ax, ay), (bx, by) = p0, p1
    x0, x1 = min(ax, bx), max(ax, bx)
    y0, y1 = min(ay, by), max(ay, by)
    return [x0, y0, x1, y1]


def storage_box_to_oriented(box, w: int, h: int, o: int) -> list[float]:
    """存储方向半开框 [x0,y0,x1,y1] -> 转正方向轴对齐框。"""
    x0, y0, x1, y1 = box
    return _box_from_points(
        storage_to_oriented_point(x0, y0, w, h, o),
        storage_to_oriented_point(x1, y1, w, h, o),
    )


def oriented_box_to_storage(box, w: int, h: int, o: int) -> list[float]:
    """转正方向半开框 -> 存储方向框（构造混合朝向样片/测试夹具时使用）。"""
    X0, Y0, X1, Y1 = box
    return _box_from_points(
        oriented_to_storage_point(X0, Y0, w, h, o),
        oriented_to_storage_point(X1, Y1, w, h, o),
    )
