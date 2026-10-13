"""Китык — длинный плюшевый кот — крутится как УИИА-кот, 1 в 1 в такт оригиналу. 13 c, 720x1280.

Usage: python3 kityk.py <out.mp4>
Фото Китыка — video/kityk_photos/ (front1, front2 — анфас, back — спина).
Звук и тайминг — оригинальный TikTok «uia original cat»: когда оригинальный кот крутится,
крутится и Китык (лицо/спина по углу поворота), когда стоит — Китык «танцует» лапками
(смена двух фото анфас) и камера наезжает. Оригинальный кот — в окошке в углу.
Общие помощники (скачивание, вырезка, текст) — из make_video.py.
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import make_video as mv

W, H, FPS, SR = mv.W, mv.H, mv.FPS, mv.SR
DUR = 13.0
PHOTOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kityk_photos")
CAPTIONS = [                       # (с какой секунды, текст)
    (0.0, "Это Китык."),
    (2.8, "Китык услышал УИИА"),
    (6.25, "Китык не может\nостановиться"),
    (9.45, "Китык — самый\nдлинный кот"),
]
STRETCH = (10.9, 12.6)             # тут Китык вытягивается (longcat)


def kityk_cutout(name):
    """Вырезка с заливкой дыр (тёмные глаза), баланс белого по пузику и осветление."""
    path = mv.asset(f"kityk_{name}.png")
    if not os.path.exists(path):
        rgb = np.asarray(Image.open(os.path.join(PHOTOS, f"{name}.jpg")).convert("RGB"))
        rgba = mv.cutout(rgb)
        alpha = mv.solid_alpha(rgba[..., 3] < 128)
        f = rgb.astype(float)
        inside = alpha > 0.5
        lum = f.mean(2)
        belly = inside & (lum > np.percentile(lum[inside], 80))
        gain = np.array([238, 236, 232]) / f[belly].mean(0)
        f = 255 * np.clip(f * gain / 255, 0, 1) ** 0.85
        out = np.dstack([f, alpha * 255]).clip(0, 255).astype(np.uint8)
        mv.to_sprite(out)[0].save(path)
    return Image.open(path).convert("RGBA")


def spin_states(cat_frames):
    """По кадрам оригинального кота: крутится (буханка, вытянут вбок) или стоит (выше, чем шире)."""
    raw = []
    for img, _ in cat_frames:
        raw.append(img.height / max(1, img.width) < 1.05)
    k = 7                                                # сглаживание большинством в окне
    return [sum(raw[max(0, i - k):i + k + 1]) * 2 > len(raw[max(0, i - k):i + k + 1]) for i in range(len(raw))]


def rays_background(t, pulse, hue):
    """Вращающиеся лучи, цвет меняется по фразам, пульсирует в такт."""
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(float)
    ang = np.arctan2(yy - H / 8, xx - W / 8) + t * 0.6
    stripes = (np.sin(ang * 12) > 0).astype(float)
    c1 = np.array(_hsv(hue, 0.65, 0.55 + 0.25 * pulse))
    c2 = np.array(_hsv(hue + 0.08, 0.8, 0.30 + 0.15 * pulse))
    img = stripes[..., None] * c1 + (1 - stripes[..., None]) * c2
    r = np.hypot(yy - H / 8, xx - W / 8) / (H / 8)
    img *= (1.15 - 0.55 * r.clip(0, 1.4))[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(1.5))


def _hsv(h, s, v):
    import colorsys
    return [255 * c for c in colorsys.hsv_to_rgb(h % 1, s, min(1, v))]


def caption_at(t):
    return [c for c in CAPTIONS if c[0] <= t][-1]


def render(fi, K, layers, pulse):
    t = fi / FPS
    bp = pulse[min(len(pulse) - 1, fi)]
    spinning = K["spin"][min(fi, len(K["spin"]) - 1)]
    seg = sum(1 for i in range(1, fi + 1) if K["spin"][i] != K["spin"][i - 1])   # номер отрезка
    frame = rays_background(t, bp, 0.78 + 0.13 * seg).convert("RGBA")

    # пока стоит — камера наезжает, как в оригинале; в longcat Китык вытягивается
    start = max([i for i in range(fi + 1) if i == 0 or K["spin"][i] != K["spin"][i - 1]])
    since = (fi - start) / FPS
    zoom = 1.0 if spinning else 1.0 + 0.12 * min(1, since / 1.2)
    stretch = 1.0
    if STRETCH[0] <= t < STRETCH[1]:
        stretch = 1.0 + 0.55 * min(1, (t - STRETCH[0]) / 1.2)
    if spinning:
        theta = since * 4.0 * 2 * math.pi
        c = math.cos(theta)
        spr = K["front1"] if c >= 0 else K["back"]
        wscale = max(abs(c), 0.06)
    else:
        spr = K["front1"] if int(t * 4) % 2 == 0 else K["front2"]    # «танцует» лапками в такт
        wscale = 1.0
    h = 980 * zoom * (1 + 0.04 * bp)
    s = h / spr.height
    # longcat: чуть выше (голова остаётся в кадре) и заметно тоньше — «макаронина»
    img = spr.resize((max(4, round(spr.width * s * wscale / stretch)),
                      round(min(spr.height * s * min(stretch, 1.12), 1060))), Image.BILINEAR)   # макушка ниже подписи
    # тень и сам Китык: низ уходит за нижний край кадра (на фото ноги обрезаны)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([W / 2 - img.width * 0.7, H - 70, W / 2 + img.width * 0.7, H + 40],
                                   fill=(0, 0, 0, 110))
    frame.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
    frame.alpha_composite(img, (round(W / 2 - img.width / 2), round(H + 40 - img.height)))

    # оригинальный кот в окошке
    cat, _ = K["cat"][min(fi, len(K["cat"]) - 1)]
    box = Image.new("RGBA", (190, 250), (0, 0, 0, 150))
    cs = min(0.3, 170 / cat.width, 200 / cat.height)
    cimg = cat.resize((max(1, round(cat.width * cs)), max(1, round(cat.height * cs))), Image.BILINEAR)
    box.alpha_composite(cimg, (round(95 - cimg.width / 2), round(140 - cimg.height / 2)))
    ImageDraw.Draw(box).rectangle([0, 0, 189, 249], outline=(255, 255, 255, 200), width=3)
    frame.alpha_composite(box, (24, H - 300))
    frame.alpha_composite(layers["orig"], (24 + 95 - layers["orig"].width // 2, H - 296))

    t0, _ = caption_at(t)
    k = min(1, (t - t0) / 0.3)
    mv.paste_c(frame, layers["caps"][t0], W / 2, 140, 1 + 0.5 * (1 - k) * math.cos(k * 8))
    if spinning:
        mv.paste_c(frame, layers["uiia"], W / 2, 290, 0.9 + 0.15 * bp)
    if t < 0.25:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - t / 0.25)))))
    if t > DUR - 0.35:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (t - DUR + 0.35) / 0.35))))
    return frame.convert("RGB")


def main():
    out = sys.argv[1]
    cat_path = mv.fetch_tiktok(mv.CAT_TT)
    cat = [mv.to_sprite(f) for f in
           mv.cached_sequence("cat", mv.read_frames(cat_path, 0, DUR, FPS), mv.cat_cutout)]
    K = dict(front1=kityk_cutout("front1"), front2=kityk_cutout("front2"), back=kityk_cutout("back"),
             cat=cat, spin=spin_states(cat))

    n = int(DUR * SR)
    music = mv.decode_audio(cat_path, 0, DUR)[:n]
    music = music / (np.abs(music).max() + 1e-9) * 0.9
    fade = np.ones(len(music))
    fade[-int(0.35 * SR):] = np.linspace(1, 0, int(0.35 * SR))
    wav = out.rsplit(".", 1)[0] + "_audio.wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((music * fade * 32767).astype(np.int16).tobytes())
    frames = int(DUR * FPS)
    onset = np.maximum(0, np.diff(mv.envelope(music, frames), prepend=0))
    onset /= np.percentile(onset[onset > 0], 95) + 1e-9
    pulse, p = np.zeros(frames), 0.0
    for i in range(frames):
        p = max(min(1.0, onset[i]), p * 0.72)
        pulse[i] = p

    layers = dict(
        caps={t0: mv.text_img(txt, 54) for t0, txt in CAPTIONS},
        uiia=mv.text_img("УИИА  УИИА", 80, fill=(255, 255, 255), stroke=8, stroke_fill=(220, 30, 90)),
        orig=mv.text_img("оригинал", 24, fill=(255, 255, 255), stroke=3),
    )
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for fi in range(frames):
        ff.stdin.write(render(fi, K, layers, pulse).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


if __name__ == "__main__":
    main()
