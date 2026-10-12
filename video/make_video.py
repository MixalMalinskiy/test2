"""УИИА-кот крутится вокруг девушки, рядом рыба и Mr Jabec — одна сцена, 14 c, 720x1280.

Usage: python3 make_video.py <meme_grid.png> <out.mp4>
Требует: ffmpeg, pip install numpy pillow opencv-python-headless "rembg[cpu]" gTTS.

Всё берётся из оригиналов, 1 в 1:
  * кот и звук — TikTok «uia original cat» (green screen), фон вырезается rembg;
  * мелкая жаба Mr Jabec, её реплика «Пять.» (родной голос) и фон-лес — TikTok 1loxxy;
  * девушка и рыба — вырезаны из присланной картинки (кадр «фронталка»).
Видео качаются через tikwm.com; вырезки кэшируются в video/assets/.
"""
import json
import math
import os
import subprocess
import sys
import urllib.request
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 720, 1280, 30
DUR = 14.0
SR = 44100
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
OUT_LINE = (25, 20, 20)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CAT_TT = "7484505655684893958"      # vm.tiktok.com/ZGdCUD1LH — uia original cat
TOAD_TT = "7616371658462399760"     # vm.tiktok.com/ZGdCUHfw6 — жаба (1loxxy)
TOAD_T0, TOAD_T1 = 5.0, 6.7         # крупный план жабы с горшком золота, она говорит «Пять.»
TOAD_IDLE = 10                      # первые кадры — жаба молчит
BG_T = 1.3                          # кадр с пустой лесной тропинкой
CAT_TEXT_BOXES = [(190, 725, 386, 772),   # «Thank you.» поверх кота (в кадре 576x1024)
                  (60, 236, 215, 305)]    # водяной знак Vizard.ai
GRID_FRONTALKA = (0, 1094, 541, 1448)
GIRL_BOX = (0, 38, 350, 300)        # внутри панели «фронталка»
FISH_BOX = (320, 30, 541, 300)


def asset(name):
    os.makedirs(ASSETS, exist_ok=True)
    return os.path.join(ASSETS, name)


def fetch_tiktok(video_id, name):
    path = asset(name)
    if os.path.exists(path):
        return path
    api = f"https://www.tikwm.com/api/?hd=1&url=https://www.tiktok.com/@x/video/{video_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)["data"]
    url = data.get("hdplay") or data["play"]
    if not url.startswith("http"):
        url = "https://www.tikwm.com" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def read_frames(path, t0, t1, fps):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], check=True, capture_output=True, text=True).stdout
    w, h = map(int, probe.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(t1 - t0), "-i", path, "-vf", f"fps={fps}",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def decode_audio(path, t0=0.0, dur=None, af=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0)] + (["-t", str(dur)] if dur else []) + ["-i", path]
    cmd += (["-af", af] if af else []) + ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32).astype(float)


# --------------------------------------------------------------------------- вырезка (rembg, с кэшем)
_SESSION = None


def cutout(rgb, main_only=True):
    """RGB -> RGBA без фона (rembg)."""
    global _SESSION
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("isnet-general-use")
    rgba = np.asarray(remove(Image.fromarray(rgb), session=_SESSION)).copy()
    return keep_main(rgba) if main_only else rgba


def keep_main(rgba):
    """Оставляет крупные куски альфы, выкидывает мелкие обрывки (водяные знаки)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((rgba[..., 3] > 40).astype(np.uint8))
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.15 * big]
        rgba[..., 3] *= np.isin(lab, keep).astype(np.uint8)
    return rgba


def cat_cutout(rgb):
    """Зелёный фон — быстрый хромакей; кадры со светлыми вспышками — через rembg."""
    rgb = remove_cat_caption(rgb)
    f = rgb.astype(float)
    green = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    border = np.concatenate([green[:8].ravel(), green[-8:].ravel(), green[:, :8].ravel(), green[:, -8:].ravel()])
    if (border > 45).mean() < 0.95:
        return cutout(rgb)
    solid = keep_main(np.dstack([rgb, (green < 70).astype(np.uint8) * 255]))[..., 3] > 0
    # заливаем дыры внутри силуэта (размытый в движении кот отливает зелёным)
    flood = solid.astype(np.uint8)
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 2)
    inside = flood != 2
    alpha = cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3))), (5, 5), 0)
    g = np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2]) + 8)       # без зелёного ореола
    return np.dstack([f[..., 0], g, f[..., 2], alpha * 255]).clip(0, 255).astype(np.uint8)


def to_sprite(rgba):
    """Обрезка по альфе; возвращает (картинка, центр исходного кадра в координатах картинки)."""
    ys, xs = np.nonzero(rgba[..., 3] > 10)
    if len(xs) == 0:
        return Image.new("RGBA", (2, 2)), (1, 1)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    h, w = rgba.shape[:2]
    return Image.fromarray(np.ascontiguousarray(rgba[y0:y1, x0:x1]), "RGBA"), (w / 2 - x0, h / 2 - y0)


def cached_sequence(name, frames, cut=cutout):
    folder = asset(name)
    os.makedirs(folder, exist_ok=True)
    out = []
    for i, f in enumerate(frames):
        p = os.path.join(folder, f"{i:04d}.png")
        if not os.path.exists(p):
            Image.fromarray(cut(f), "RGBA").save(p)
            if i % 30 == 0:
                print(f"  {name}: {i}/{len(frames)}", flush=True)
        out.append(np.asarray(Image.open(p).convert("RGBA")))
    return out


def remove_cat_caption(rgb):
    img = np.ascontiguousarray(rgb)
    mask = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in CAT_TEXT_BOXES:
        mask[y0:y1, x0:x1] = (img[y0:y1, x0:x1].min(2) > 190).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8))         # вместе с тёмной обводкой букв
    return cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)


def make_background(toad_path):
    p = asset("bg.png")
    if not os.path.exists(p):
        rgb = read_frames(toad_path, BG_T, BG_T + 0.05, 30)[0].copy()
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        # подпись и водяной знак 1loxxy
        cap = rgb[int(0.15 * h):int(0.30 * h)]
        mask[int(0.15 * h):int(0.30 * h)] = (cap.min(2) > 200).astype(np.uint8) * 255
        mask[int(0.78 * h):int(0.86 * h), int(0.33 * w):int(0.67 * w)] = 255
        # сама жаба на тропинке
        frog = cutout(rgb)[..., 3] > 30
        frog[: int(0.35 * h)] = False
        mask[frog] = 255
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        Image.fromarray(cv2.inpaint(rgb, mask, 15, cv2.INPAINT_TELEA)).save(p)
    return Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)


def load_assets(grid_path):
    cat_path = fetch_tiktok(CAT_TT, "cat_tiktok.mp4")
    toad_path = fetch_tiktok(TOAD_TT, "toad_tiktok.mp4")
    print("cat frames...", flush=True)
    cat = [to_sprite(f) for f in cached_sequence("cat", read_frames(cat_path, 0, DUR, FPS), cat_cutout)]
    print("toad frames...", flush=True)
    toad = [to_sprite(f)[0] for f in cached_sequence("toad", read_frames(toad_path, TOAD_T0, TOAD_T1, FPS))]
    pan = Image.open(grid_path).convert("RGB").crop(GRID_FRONTALKA)
    girl = cached_sequence("girl", [np.asarray(pan.crop(GIRL_BOX))])[0].copy()
    fade = int(girl.shape[0] * 0.25)               # мягкий низ, без резкого среза
    girl[-fade:, :, 3] = (girl[-fade:, :, 3] * np.linspace(1, 0, fade)[:, None]).astype(np.uint8)
    fish = cached_sequence("fish", [np.asarray(pan.crop(FISH_BOX))])[0]
    return dict(cat_path=cat_path, toad_path=toad_path, cat=cat, toad=toad, girl=to_sprite(girl)[0], fish=to_sprite(fish)[0],
                bg=make_background(toad_path))


# --------------------------------------------------------------------------- графика
def text_img(txt, size, fill=(255, 225, 40), stroke=6, stroke_fill=(10, 10, 10)):
    f = ImageFont.truetype(FONT, size)
    bb = [int(round(v)) for v in ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox(
        (0, 0), txt, font=f, stroke_width=stroke, align="center", spacing=6)]
    im = Image.new("RGBA", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).multiline_text((4 - bb[0], 4 - bb[1]), txt, font=f, fill=fill, stroke_width=stroke,
                                      stroke_fill=stroke_fill, align="center", spacing=6)
    return im


def bubble_img(txt, tail_left=True, size=34):
    t = text_img(txt, size, fill=(15, 15, 15), stroke=0)
    pad = 18
    bw, bh = t.width + pad * 2, t.height + pad * 2
    im = Image.new("RGBA", (bw + 8, bh + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    tx = 50 if tail_left else bw - 50
    tip = tx + (-8 if tail_left else 8)
    d.polygon([(tx - 14, bh - 4), (tx + 14, bh - 4), (tip, bh + 34)], fill="white", outline=OUT_LINE, width=4)
    d.rounded_rectangle([2, 2, bw, bh], radius=22, fill="white", outline=OUT_LINE, width=4)
    d.polygon([(tx - 11, bh - 6), (tx + 11, bh - 6), (tip, bh + 28)], fill="white")
    im.alpha_composite(t, (pad + 2, pad + 2))
    return im


def paste_c(frame, img, cx, cy, scale=1.0, angle=0.0):
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    frame.alpha_composite(img, (round(cx - img.width / 2), round(cy - img.height / 2)))


CAPTION = "Она случайно открыла\nфронталку в 4 утра"
# (t0, t1, кто, текст пузыря, озвучка, высота голоса); озвучка None — родной голос жабы из TikTok
LINES = [
    (2.3, 4.6, "fish", "жаба, оцени её\nиз десяти", "Жаба, оцени её из десяти", 1.45),
    (4.8, 6.5, "toad", "пять.", None, 1.0),
    (6.8, 9.0, "fish", "даже я так\nне надуваюсь", "Даже я так не надуваюсь", 1.45),
    (9.2, 11.6, "toad", "ква. скидываемся ей\nна новую камеру", "Ква. Скидываемся ей на новую камеру", 0.72),
]
TOAD_SYNC = 4.8           # с этого момента кадры жабы идут 1 в 1 с её голосом
GIRL_C = (360, 500)       # центр лица девушки на экране
FISH_C = (150, 1000)
TOAD_C = (545, 985)


def render(fi, A, pulse, layers):
    t = fi / FPS
    bp = pulse[min(len(pulse) - 1, fi)]
    z = 1.0 + 0.05 * t / DUR + 0.015 * bp
    bg = A["bg"].resize((round(W * z), round(H * z)), Image.BILINEAR)
    x0, y0 = (bg.width - W) // 2, (bg.height - H) // 2
    frame = bg.crop((x0, y0, x0 + W, y0 + H)).convert("RGBA")

    # кот: оригинальные кадры в такт оригинальному звуку, орбита вокруг лица
    cat_img, (ox, oy) = A["cat"][min(fi, len(A["cat"]) - 1)]
    orb = t * 0.55 * 2 * math.pi
    depth = math.sin(orb)                                  # >0 — кот перед девушкой
    cs = 0.42 + 0.08 * depth
    cx = GIRL_C[0] + 270 * math.cos(orb)
    cy = GIRL_C[1] + 30 + 110 * depth
    # центр исходного кадра кота ставим в (cx, cy) — так сохраняются его наезды камеры
    cat_pos = (cx + (cat_img.width / 2 - ox) * cs, cy + (cat_img.height / 2 - oy) * cs)

    if depth < 0:
        paste_c(frame, cat_img, *cat_pos, cs)
    wob = 4 * math.sin(t * 2 * math.pi * 0.9)
    paste_c(frame, A["girl"], GIRL_C[0], GIRL_C[1] + 20, 1.55 * (1 + 0.05 * bp), wob)
    if depth >= 0:
        paste_c(frame, cat_img, *cat_pos, cs)

    # рыба и жаба
    paste_c(frame, A["fish"], FISH_C[0], FISH_C[1] + 12 * math.sin(t * 3.1) - 20 * bp, 1.15 * (1 + 0.06 * bp))
    frames = A["toad"]
    talking = any(b0 <= t < b1 and who == "toad" and voice for b0, b1, who, _, voice, _ in LINES)
    if TOAD_SYNC <= t < TOAD_SYNC + len(frames) / FPS:    # «Пять.» — кадры подряд, в синхрон с голосом
        toad = frames[int((t - TOAD_SYNC) * FPS)]
    else:                                                  # туда-обратно, без скачка
        n = len(frames) if talking else TOAD_IDLE
        k = int(t * FPS) % (2 * n - 2)
        toad = frames[k if k < n else 2 * n - 2 - k]
    paste_c(frame, toad, TOAD_C[0], TOAD_C[1] - 10 * bp, 0.55)
    paste_c(frame, layers["name"], TOAD_C[0], 1240)

    # подпись
    k = min(1, t / 0.3)
    paste_c(frame, layers["cap"], W / 2, 120, (1 + 0.5 * (1 - k) * math.cos(k * 8)) * (1 + 0.03 * bp))

    for (b0, b1, who, *_), img in zip(LINES, layers["bubbles"]):
        if b0 <= t < b1:
            s = 0.6 + 0.4 * min(1, (t - b0) / 0.15)
            if who == "fish":
                paste_c(frame, img, 30 + img.width / 2, 840 - img.height / 2, s)
            else:
                paste_c(frame, img, 700 - img.width / 2, 780 - img.height / 2, s)

    if t > 11.7:                                           # финал
        paste_c(frame, layers["uiia"], W / 2, 760, (0.9 + 0.2 * bp) * min(1, (t - 11.7) / 0.2))
    if t > DUR - 0.35:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (t - DUR + 0.35) / 0.35))))
    return frame.convert("RGB")


# --------------------------------------------------------------------------- звук
def tts_line(text, pitch, window, name):
    """gTTS + сдвиг высоты (рыба — писклявая, жаба — басом)."""
    from gtts import gTTS
    path = asset(name)
    if not os.path.exists(path):
        gTTS(text, lang="ru").save(path)
    length = len(decode_audio(path)) / SR
    tempo = min(2.0, max(1.1, length / window))
    chain = [tempo / pitch]
    while chain[-1] > 2.0:
        chain = chain[:-1] + [2.0, chain[-1] / 2.0]
    af = f"asetrate={round(24000 * pitch)},aresample={SR}," + ",".join(f"atempo={c:.4f}" for c in chain)
    return decode_audio(path, af=af)


def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def make_audio(A, path):
    n = int(DUR * SR)
    music = np.zeros(n)
    orig = decode_audio(A["cat_path"], 0, DUR)              # оригинальный звук УИИА
    music[: min(n, len(orig))] = orig[:n]
    voice = np.zeros(n)
    duck = np.ones(n)
    for i, (b0, b1, _, _, text, pitch) in enumerate(LINES):
        if text is None:                                   # родной голос жабы
            line, at = decode_audio(A["toad_path"], TOAD_T0, TOAD_T1 - TOAD_T0), TOAD_SYNC
        else:
            line, at = tts_line(text, pitch, b1 - b0 - 0.4, f"tts_line{i}.mp3"), b0 + 0.1
        add(voice, line / (np.abs(line).max() + 1e-9), at)
        duck[int(at * SR):int(at * SR) + len(line)] = 0.4
    duck = np.convolve(duck, np.ones(2000) / 2000, mode="same")
    music *= 0.85 / (np.abs(music).max() + 1e-9)
    mix = music * duck + 0.8 * voice
    fade = np.ones(n)
    fade[-int(0.35 * SR):] = np.linspace(1, 0, int(0.35 * SR))
    mix = np.tanh(1.1 * mix * fade) * 0.95
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # пульс для анимации — по атакам оригинального звука
    hop = SR // FPS
    frames = int(DUR * FPS)
    rms = np.array([np.sqrt(np.mean(music[k * hop:(k + 1) * hop] ** 2)) if k * hop < n else 0.0
                    for k in range(frames)])
    onset = np.maximum(0, np.diff(rms, prepend=0))
    onset /= np.percentile(onset[onset > 0], 95) + 1e-9
    pulse, p = np.zeros(frames), 0.0
    for k in range(frames):
        p = max(min(1.0, onset[k]), p * 0.72)
        pulse[k] = p
    return pulse


def main():
    grid, out = sys.argv[1], sys.argv[2]
    A = load_assets(grid)
    wav = out.rsplit(".", 1)[0] + "_audio.wav"
    pulse = make_audio(A, wav)
    layers = dict(
        cap=text_img(CAPTION, 50),
        name=text_img("Mr Jabec", 32, fill=(140, 230, 90), stroke=5),
        uiia=text_img("УИИА  УИИА", 92, fill=(255, 255, 255), stroke=8, stroke_fill=(220, 30, 90)),
        bubbles=[bubble_img(b[3], b[2] == "fish") for b in LINES],
    )
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for fi in range(int(DUR * FPS)):
        ff.stdin.write(render(fi, A, pulse, layers).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


if __name__ == "__main__":
    main()
