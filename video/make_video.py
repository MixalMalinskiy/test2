"""УИИА-кот + рыба + Mr Jabec: генератор короткого мем-видео (14 c, 720x1280).

Usage: python3 make_video.py <meme_grid.png> <out.mp4>
Требует: ffmpeg, pip install numpy pillow gTTS.
Настоящий УИИА-кот (green screen, Tenor) и ремикс (MyInstants) скачиваются в
video/assets/ при первом запуске; реплики рыбы и жабы озвучиваются через gTTS.
Рыба и жаба рисуются кодом.
"""
import math
import os
import random
import subprocess
import sys
import urllib.request
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 720, 1280, 30
DUR = 14.0
SR = 44100
DROP = 1.5          # кот падает, начинается музыка
BOOM = 7.0          # фронталка
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
OUT_LINE = (25, 20, 20)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CAT_URL = "https://media.tenor.com/QB21w9GkiOoAAAP1/spincat.mp4"
MUSIC_URL = "https://www.myinstants.com/media/sounds/oiia-cat-remix.mp3"
CAT_FRAME = 0                             # кадр клипа, где кот стоит анфас
MUSIC_LOOP_START = 6.13                   # с этой секунды в ремиксе начинается громкая часть

random.seed(7)


def fetch(url, name):
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def decode_audio(path, af=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, np.float32).astype(float)


def font(size):
    return ImageFont.truetype(FONT, size)


def aa(img, size):
    """Рисуем в 2x и уменьшаем — сглаживание."""
    return img.resize(size, Image.LANCZOS)


# --------------------------------------------------------------------------- sprites
def load_cat():
    """Настоящий УИИА-кот: вырезаем зелёный фон; возвращает (анфас, зеркальный и чуть темнее — «спина»)."""
    path = fetch(CAT_URL, "oiia_cat_greenscreen.mp4")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 320, 320, 3).astype(float)
    r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
    green = g - np.maximum(r, b)
    alpha = np.clip((90 - green) / 50, 0, 1)                   # мягкий край
    g = np.minimum(g, np.maximum(r, b) + 10)                  # убираем зелёный ореол
    rgba = np.stack([r, g, b, alpha * 255], -1).clip(0, 255).astype(np.uint8)

    ys, xs = np.nonzero(alpha[CAT_FRAME] > 0.5)
    front = Image.fromarray(rgba[CAT_FRAME], "RGBA").crop((xs.min() - 2, ys.min() - 2, xs.max() + 3, ys.max() + 3))
    back = front.transpose(Image.FLIP_LEFT_RIGHT)
    back = Image.merge("RGBA", [*(c.point(lambda v: int(v * 0.7)) for c in back.split()[:3]), back.getchannel("A")])
    return front, back


def draw_fish(mouth_open=False):
    S = 2
    im = Image.new("RGBA", (260 * S, 240 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    o, lw = OUT_LINE, 5 * S
    org, dorg, lorg = (250, 140, 40), (215, 95, 25), (255, 200, 110)
    # колючки
    for k in range(22):
        a = k / 22 * 2 * math.pi
        cx, cy, r = 130, 125, 92
        pts = [(cx + r * math.cos(a - 0.12), cy + r * math.sin(a - 0.12)),
               (cx + (r + 18) * math.cos(a), cy + (r + 18) * math.sin(a)),
               (cx + r * math.cos(a + 0.12), cy + r * math.sin(a + 0.12))]
        d.polygon([(x * S, y * S) for x, y in pts], fill=dorg, outline=o, width=3 * S)
    # хвост и плавники
    d.polygon([(x * S, y * S) for x, y in [(42, 125), (2, 85), (14, 125), (2, 165)]], fill=lorg, outline=o, width=lw)
    d.polygon([(x * S, y * S) for x, y in [(100, 40), (135, 2), (160, 40)]], fill=lorg, outline=o, width=lw)
    d.ellipse([38 * S, 33 * S, 222 * S, 217 * S], fill=org, outline=o, width=lw)
    d.ellipse([70 * S, 50 * S, 150 * S, 100 * S], fill=(255, 180, 90))
    for _ in range(26):
        x, y = random.uniform(60, 200), random.uniform(55, 200)
        if (x - 130) ** 2 + (y - 125) ** 2 < 80 ** 2:
            d.ellipse([(x - 4) * S, (y - 4) * S, (x + 4) * S, (y + 4) * S], fill=dorg)
    # глаза
    for ex in (95, 165):
        d.ellipse([(ex - 26) * S, 88 * S, (ex + 26) * S, 140 * S], fill=(255, 250, 220), outline=o, width=lw)
        d.ellipse([(ex - 13) * S, 101 * S, (ex + 13) * S, 129 * S], fill=(20, 20, 20))
        d.ellipse([(ex - 9) * S, 104 * S, (ex - 1) * S, 112 * S], fill="white")
    d.ellipse([200 * S, 125 * S, 245 * S, 150 * S], fill=lorg, outline=o, width=4 * S)  # боковой плавник
    if mouth_open:
        d.ellipse([105 * S, 150 * S, 155 * S, 195 * S], fill=(110, 20, 30), outline=o, width=lw)
        d.ellipse([115 * S, 172 * S, 145 * S, 194 * S], fill=(255, 120, 140))
    else:
        d.arc([100 * S, 135 * S, 160 * S, 180 * S], 15, 165, fill=o, width=lw)
    return aa(im, (260, 240))


def draw_toad(mouth_open=False):
    S = 2
    im = Image.new("RGBA", (280 * S, 250 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    o, lw = OUT_LINE, 5 * S
    g, lg, dg = (110, 175, 60), (190, 220, 120), (70, 120, 40)

    def e(box, fill, w=lw):
        d.ellipse([v * S for v in box], fill=fill, outline=o, width=w)

    e((20, 150, 80, 248), g)                    # лапы
    e((200, 150, 260, 248), g)
    e((25, 70, 255, 245), g)                    # тело
    e((75, 140, 205, 240), lg)                  # пузо
    for x, y in ((60, 110), (220, 120), (50, 170), (230, 175)):
        d.ellipse([(x - 7) * S, (y - 7) * S, (x + 7) * S, (y + 7) * S], fill=dg)
    for ex in (78, 202):                        # глаза на макушке
        e((ex - 42, 18, ex + 42, 102), g)
        e((ex - 30, 30, ex + 30, 92), (255, 255, 230))
        d.ellipse([(ex - 9) * S, 45 * S, (ex + 15) * S, 85 * S], fill=(20, 20, 20))
        d.ellipse([(ex - 3) * S, 50 * S, (ex + 7) * S, 60 * S], fill="white")
    d.ellipse([45 * S, 125 * S, 75 * S, 145 * S], fill=(240, 130, 140))   # щёки
    d.ellipse([205 * S, 125 * S, 235 * S, 145 * S], fill=(240, 130, 140))
    if mouth_open:
        d.chord([70 * S, 95 * S, 210 * S, 185 * S], 0, 180, fill=(120, 25, 35), outline=o, width=lw)
        d.ellipse([115 * S, 150 * S, 165 * S, 182 * S], fill=(240, 110, 130))
    else:
        d.arc([65 * S, 80 * S, 215 * S, 160 * S], 15, 165, fill=o, width=lw)
    # геймерская гарнитура
    d.arc([20 * S, -10 * S, 260 * S, 150 * S], 200, 340, fill=(30, 30, 30), width=12 * S)
    e((5, 75, 45, 135), (35, 35, 40))
    e((235, 75, 275, 135), (35, 35, 40))
    d.ellipse([12 * S, 92 * S, 38 * S, 118 * S], fill=(220, 40, 40))
    d.ellipse([242 * S, 92 * S, 268 * S, 118 * S], fill=(220, 40, 40))
    d.line([(255 * S, 125 * S), (230 * S, 160 * S), (200 * S, 165 * S)], fill=(30, 30, 30), width=7 * S, joint="curve")
    e((188, 157, 206, 175), (30, 30, 30), 2 * S)
    return aa(im, (280, 250))


CAT_FRONT, CAT_BACK = load_cat()
FISH = draw_fish(False)
FISH_O = draw_fish(True)
TOAD = draw_toad(False)
TOAD_O = draw_toad(True)


def text_img(txt, size, fill=(255, 225, 40), stroke=6, stroke_fill=(10, 10, 10)):
    f = font(size)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bb = [int(round(v)) for v in tmp.multiline_textbbox((0, 0), txt, font=f, stroke_width=stroke, align="center", spacing=6)]
    im = Image.new("RGBA", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).multiline_text((4 - bb[0], 4 - bb[1]), txt, font=f, fill=fill, stroke_width=stroke,
                                      stroke_fill=stroke_fill, align="center", spacing=6)
    return im


def bubble_img(txt, size=30, tail="down-left"):
    t = text_img(txt, size, fill=(15, 15, 15), stroke=0)
    pad = 18
    bw, bh = t.width + pad * 2, t.height + pad * 2
    im = Image.new("RGBA", (bw + 8, bh + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    tx = 40 if tail == "down-left" else bw - 40
    d.polygon([(tx - 14, bh - 4), (tx + 14, bh - 4), (tx + (-6 if tail == "down-left" else 6), bh + 34)],
              fill="white", outline=OUT_LINE, width=4)
    d.rounded_rectangle([2, 2, bw, bh], radius=22, fill="white", outline=OUT_LINE, width=4)
    d.polygon([(tx - 11, bh - 6), (tx + 11, bh - 6), (tx + (-5 if tail == "down-left" else 5), bh + 28)], fill="white")
    im.alpha_composite(t, (pad + 2, pad + 2))
    return im


# --------------------------------------------------------------------------- scenes
GRID = Image.open(sys.argv[1]).convert("RGB")
PANELS = {"p1": (0, 0, 541, 254), "p5": (0, 528, 541, 815), "p9": (0, 1094, 541, 1448), "p10": (546, 1094, 1086, 1448)}
# центр лица девушки в координатах панели (для орбиты кота)
FACE = {"p1": (380, 120), "p5": (150, 150), "p9": (170, 190), "p10": (215, 140)}
CROP_CX = {"p1": 360, "p5": 230, "p9": 270, "p10": 270}   # центр кропа по X
PANEL_Y, PANEL_H = 250, 560

SCENES = [
    dict(t0=0.0, t1=3.5, panel="p1", cap="POV: 03:48,\nона «уже спит»"),
    dict(t0=3.5, t1=7.0, panel="p5", cap="Когда в 4 утра\nпришли друзья"),
    dict(t0=7.0, t1=10.5, panel="p9", cap="Случайно открыла\nфронталку"),
    dict(t0=10.5, t1=DUR, panel="p10", cap="Мы вчетвером:\nдержим оборону дивана"),
]
for s in SCENES:
    x0, y0, x1, y1 = PANELS[s["panel"]]
    cw = round((y1 - y0) * W / PANEL_H)
    cx0 = min(max(0, CROP_CX[s["panel"]] - cw // 2), (x1 - x0) - cw)
    pan = GRID.crop((x0 + cx0, y0, x0 + cx0 + cw, y1))
    s["img"] = pan.resize((W, PANEL_H), Image.LANCZOS)
    bg = pan.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(22))
    s["bg"] = Image.blend(bg, Image.new("RGB", (W, H), (10, 5, 20)), 0.55).convert("RGBA")
    s["cap_img"] = text_img(s["cap"], 50)
    sc = W / pan.width
    s["face"] = ((FACE[s["panel"]][0] - cx0) * sc, FACE[s["panel"]][1] * sc)

# (t0, t1, кто, текст)
BUBBLES = [
    (0.3, 1.5, "fish", "zzz... она\nточно спит"),
    (3.7, 5.4, "toad", "ГО ДОТУ?\nя на миде"),
    (5.3, 7.0, "fish", "4 утра,\nжаба!!"),
    (7.6, 9.0, "toad", "это ФИД.\n-25 MMR"),
    (9.0, 10.5, "fish", "я всё равно\nкрасивее"),
    (10.8, 12.4, "toad", "катку после\nсериала?"),
    (12.4, 14.0, "fish", "ГГ ВП"),
]
for b in BUBBLES:
    b_img = bubble_img(b[3], 36, "down-left" if b[2] == "fish" else "down-right")
    BUBBLES[BUBBLES.index(b)] = b + (b_img,)

UIIA = text_img("УИИА  УИИА", 84, fill=(255, 255, 255), stroke=8, stroke_fill=(220, 30, 90))
NAME_FISH = text_img("рыба", 30, fill=(255, 180, 70), stroke=4)
NAME_TOAD = text_img("Mr Jabec", 30, fill=(140, 230, 90), stroke=4)


PULSE = np.zeros(int(DUR * FPS) + 1)   # огибающая музыки по кадрам, заполняется в make_audio


def beat_pulse(t):
    return float(PULSE[min(len(PULSE) - 1, int(t * FPS))])


def paste_c(frame, img, cx, cy, scale=1.0):
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    frame.alpha_composite(img, (round(cx - img.width / 2), round(cy - img.height / 2)))


def spin_cat(frame, cx, cy, theta, scale):
    """Кручение вокруг вертикальной оси, как в меме."""
    c = math.cos(theta)
    spr = CAT_FRONT if c >= 0 else CAT_BACK
    w = max(4, round(spr.width * scale * max(abs(c), 0.06)))
    img = spr.resize((w, round(spr.height * scale)), Image.BILINEAR)
    frame.alpha_composite(img, (round(cx - w / 2), round(cy - img.height / 2)))


def render(fi):
    t = fi / FPS
    s = next(x for x in SCENES if x["t0"] <= t < x["t1"])
    lt = t - s["t0"]
    frame = s["bg"].copy()
    bp = beat_pulse(t)

    # тряска на фронталке
    sx = sy = 0
    if t >= BOOM:
        amp = 22 * math.exp(-(t - BOOM) / 0.5) + 4 * bp
        sx, sy = random.uniform(-amp, amp), random.uniform(-amp, amp)

    # панель с лёгким зумом в лицо (на фронталке — резкий)
    zoom = 1.0 + 0.06 * lt / (s["t1"] - s["t0"]) + 0.02 * bp
    if s["panel"] == "p9":
        zoom += 0.35 * min(1, lt / 0.12) * math.exp(-lt / 1.2)
    pan = s["img"]
    pw, ph = round(pan.width * zoom), round(pan.height * zoom)
    fx, fy = s["face"]
    px = round(fx - fx * zoom + sx)
    py = round(PANEL_Y + fy - fy * zoom + sy)
    clip = Image.new("RGBA", (W, pan.height), (0, 0, 0, 0))
    clip.paste(pan.resize((pw, ph), Image.BILINEAR), (px, py - PANEL_Y))
    frame.alpha_composite(clip, (0, PANEL_Y))
    ImageDraw.Draw(frame).rectangle([0, PANEL_Y - 3, W, PANEL_Y + pan.height + 3], outline=(255, 225, 40), width=5)

    # подпись сцены (вылет с пружинкой)
    k = min(1, lt / 0.25)
    cs = 1 + 0.5 * (1 - k) * math.cos(k * 8)
    paste_c(frame, s["cap_img"], W / 2 + sx * 0.3, 125, cs * (1 + 0.04 * bp))

    # кот УИИА
    fx, fy = s["face"][0], PANEL_Y + s["face"][1]
    if t < DROP:                                  # кот подглядывает из-за края
        k = min(1, t / 0.6)
        paste_c(frame, CAT_FRONT, W - 70, PANEL_Y + PANEL_H + 100 - 150 * k * k, 0.75)
    else:
        theta = (t - DROP) * (6.0 if s["panel"] == "p9" else 3.0) * 2 * math.pi
        if t < DROP + 0.35:                       # прыжок в кадр
            k = (t - DROP) / 0.35
            cx, cy, sc = W - 70 + (fx + 170 - W + 70) * k, PANEL_Y + PANEL_H - 50 - 300 * math.sin(k * math.pi), 0.9
        else:
            orb = (t - DROP - 0.35) * (2.6 if s["panel"] == "p9" else 1.2) * 2 * math.pi
            rx, ry = (250, 150)
            cx = min(W - 70, max(70, fx + rx * math.cos(orb)))
            cy = fy + 40 + ry * math.sin(orb)
            sc = 1.1 + 0.2 * math.sin(orb)
        spin_cat(frame, cx + sx, cy + sy, theta, sc * (1 + 0.12 * bp))
        if t < BOOM or t > BOOM + 0.6:
            a = 0.5 + 0.5 * bp
            u = UIIA.copy()
            u.putalpha(u.getchannel("A").point(lambda v: int(v * a)))
            paste_c(frame, u, W / 2 + random.uniform(-3, 3) * bp, PANEL_Y + s["img"].height + 55, 0.9 + 0.15 * bp)

    # персонажи снизу
    base_y = 1120
    fish_talk = any(b0 <= t < b1 and who == "fish" for b0, b1, who, *_ in BUBBLES)
    toad_talk = any(b0 <= t < b1 and who == "toad" for b0, b1, who, *_ in BUBBLES)
    fy_ = base_y - 30 * bp + 8 * math.sin(t * 5)
    paste_c(frame, FISH_O if fish_talk and int(t * 9) % 2 else FISH, 175, fy_, 1.15 + 0.08 * bp)
    paste_c(frame, NAME_FISH, 175, 1258)
    if t >= 3.55:
        slide = min(1, (t - 3.55) / 0.3)
        tx = 545 + (1 - slide) ** 2 * 400
        ty = base_y + 10 - 45 * bp * (1 if t > DROP else 0)
        paste_c(frame, TOAD_O if toad_talk and int(t * 8) % 2 else TOAD, tx, ty, 1.15 + 0.06 * bp)
        paste_c(frame, NAME_TOAD, tx, 1258)

    for b0, b1, who, _, img in BUBBLES:
        if b0 <= t < b1:
            k = min(1, (t - b0) / 0.15)
            if who == "fish":
                paste_c(frame, img, 40 + img.width / 2, 975 - img.height / 2, 0.6 + 0.4 * k)
            else:
                paste_c(frame, img, 690 - img.width / 2, 960 - img.height / 2, 0.6 + 0.4 * k)

    # вспышка на смене сцен
    for x in SCENES[1:]:
        if 0 <= t - x["t0"] < 0.18:
            fl = Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - (t - x["t0"]) / 0.18))))
            frame.alpha_composite(fl)
    if t > DUR - 0.35:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (t - DUR + 0.35) / 0.35))))
    return frame.convert("RGB")


# --------------------------------------------------------------------------- audio
def env_ar(n, a, r):
    t = np.arange(n) / SR
    return np.minimum(1, t / a) * np.exp(-t / r)


def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def hat():
    n = int(0.06 * SR)
    x = np.diff(np.random.randn(n + 1))
    return 0.25 * x * env_ar(n, 0.001, 0.015)


def boom():
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    f = 38 + 60 * np.exp(-t / 0.08)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    x = np.tanh(3 * x) * env_ar(n, 0.003, 0.55)
    return 1.2 * x + 0.3 * np.random.randn(n) * env_ar(n, 0.001, 0.02)


def boing():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 260 + 140 * np.sin(2 * np.pi * 14 * t) * np.exp(-t / 0.25) + 200 * t
    return 0.45 * np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ar(n, 0.003, 0.25)


def croak():
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    saw = 2 * ((np.cumsum(np.full(n, 95.0)) / SR) % 1) - 1
    return 0.35 * saw * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 22 * t))) * env_ar(n, 0.01, 0.2)


# озвучка: (индекс реплики в BUBBLES, текст, сдвиг высоты голоса)
SPEECH = [
    (0, "Тсс... она точно спит", 1.45),
    (1, "Го доту? Я на миде!", 0.72),
    (2, "Четыре утра, жаба!", 1.45),
    (3, "Это фид. Минус двадцать пять эм эм эр.", 0.72),
    (4, "Я всё равно красивее.", 1.45),
    (5, "Катку после сериала?", 0.72),
    (6, "Гэ гэ, вэ пэ!", 1.45),
]


def tts_line(text, pitch, window, name):
    """gTTS + сдвиг высоты (рыба — писклявая, жаба — басом), ускоряем, чтобы влезло в окно реплики."""
    from gtts import gTTS
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path):
        os.makedirs(ASSETS, exist_ok=True)
        gTTS(text, lang="ru").save(path)
    length = len(decode_audio(path)) / SR
    tempo = min(2.2, max(1.15, length / window))
    chain = [tempo / pitch]
    while chain[-1] > 2.0:
        chain = chain[:-1] + [2.0, chain[-1] / 2.0]
    af = f"asetrate={round(24000 * pitch)},aresample={SR}," + ",".join(f"atempo={c:.4f}" for c in chain)
    return decode_audio(path, af)


def make_audio(path):
    np.random.seed(3)
    n = int(DUR * SR)
    sfx = np.zeros(n)
    voice = np.zeros(n)
    duck = np.ones(n)

    # музыка: зацикленный ремикс УИИА с дропа, тишина на «бум» фронталки
    remix = decode_audio(fetch(MUSIC_URL, "oiia_cat_remix.mp3"))
    loop = remix[int(MUSIC_LOOP_START * SR):]
    music = np.zeros(n)
    t = DROP
    while t < DUR:
        add(music, loop, t)
        t += len(loop) / SR
    i0, i1 = int(BOOM * SR), int((BOOM + 0.6) * SR)
    music[i0:i1] = 0
    ramp = int(0.05 * SR)
    music[i1:i1 + ramp] *= np.linspace(0, 1, ramp)

    # реплики рыбы и жабы
    for bi, text, pitch in SPEECH:
        b0, b1 = BUBBLES[bi][0], BUBBLES[bi][1]
        line = tts_line(text, pitch, b1 - b0 - 0.15, f"tts_{bi}.mp3")
        add(voice, line / (np.abs(line).max() + 1e-9), b0 + 0.05)
        duck[int(b0 * SR):int((b0 + 0.05) * SR) + len(line)] = 0.4
    duck = np.convolve(duck, np.ones(2000) / 2000, mode="same")

    # тихое тиканье часов до дропа
    for k in range(3):
        add(sfx, 0.15 * hat(), 0.2 + k * 0.45)
    add(sfx, boing(), DROP)
    add(sfx, croak(), 3.55)
    add(sfx, boom(), BOOM)
    for x in SCENES[1:]:
        add(sfx, 0.4 * np.random.randn(int(0.12 * SR)) * env_ar(int(0.12 * SR), 0.001, 0.04), x["t0"])

    music *= 0.8 / (np.abs(music).max() + 1e-9)
    mix = music * duck + 0.75 * voice + 0.6 * sfx
    fade = np.ones(n)
    fade[-int(0.3 * SR):] = np.linspace(1, 0, int(0.3 * SR))
    mix = np.tanh(1.2 * mix * fade) * 0.9
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    # пульс для анимации — по атакам музыки
    hop = SR // FPS
    rms = np.array([np.sqrt(np.mean(music[k * hop:(k + 1) * hop] ** 2)) if k * hop < n else 0.0
                    for k in range(len(PULSE))])
    onset = np.maximum(0, np.diff(rms, prepend=0))
    onset /= np.percentile(onset[onset > 0], 95) + 1e-9
    p = 0.0
    for k in range(len(PULSE)):
        p = max(min(1.0, onset[k]), p * 0.72)
        PULSE[k] = p


def main():
    out = sys.argv[2]
    wav = out.rsplit(".", 1)[0] + "_audio.wav"
    make_audio(wav)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for fi in range(int(DUR * FPS)):
        ff.stdin.write(render(fi).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
