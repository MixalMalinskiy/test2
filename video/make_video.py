"""УИИА-кот + рыба + Mr Jabec: генератор короткого мем-видео (14 c, 720x1280).

Usage: python3 make_video.py <meme_grid.png> <out.mp4>
Все персонажи рисуются кодом, звук синтезируется (numpy), сборка через ffmpeg.
"""
import math
import random
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 720, 1280, 30
DUR = 14.0
SR = 44100
BPM = 128
BEAT = 60 / BPM
DROP = 1.5          # кот падает, начинается бит
BOOM = 7.0          # фронталка
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
OUT_LINE = (25, 20, 20)

random.seed(7)


def font(size):
    return ImageFont.truetype(FONT, size)


def aa(img, size):
    """Рисуем в 2x и уменьшаем — сглаживание."""
    return img.resize(size, Image.LANCZOS)


# --------------------------------------------------------------------------- sprites
def draw_cat(front=True, mouth_open=False):
    S = 2
    im = Image.new("RGBA", (220 * S, 300 * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    o, lw = OUT_LINE, 5 * S
    grey, dark, white, pink = (178, 178, 186), (120, 120, 130), (250, 250, 250), (255, 150, 170)

    def e(box, fill):
        d.ellipse([v * S for v in box], fill=fill, outline=o, width=lw)

    def p(pts, fill, outline=o):
        d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline, width=lw if outline else 0)

    # хвост
    tx = 190 if front else 30
    d.line([(tx * S, 270 * S), ((tx + (15 if front else -15)) * S, 210 * S), ((tx - (5 if front else -5)) * S, 170 * S)],
           fill=o, width=22 * S, joint="curve")
    d.line([(tx * S, 270 * S), ((tx + (15 if front else -15)) * S, 210 * S), ((tx - (5 if front else -5)) * S, 170 * S)],
           fill=grey, width=13 * S, joint="curve")
    # тело
    e((55, 170, 165, 292), grey)
    if front:
        e((80, 200, 140, 285), white)
    # лапки
    e((60, 270, 100, 296), white if front else grey)
    e((120, 270, 160, 296), white if front else grey)
    # уши
    p([(35, 80), (45, 8), (95, 50)], grey)
    p([(185, 80), (175, 8), (125, 50)], grey)
    if front:
        p([(48, 66), (52, 25), (82, 50)], pink, None)
        p([(172, 66), (168, 25), (138, 50)], pink, None)
    # голова
    e((22, 35, 198, 190), grey)
    # полоски на лбу
    for dx in (-22, 0, 22):
        d.line([((110 + dx) * S, 42 * S), ((110 + dx * 0.8) * S, 68 * S)], fill=dark, width=6 * S)
    if front:
        e((70, 120, 150, 182), white)                         # мордочка
        e((55, 80, 95, 128), (30, 30, 35))                    # глаза
        e((125, 80, 165, 128), (30, 30, 35))
        d.ellipse([62 * S, 88 * S, 76 * S, 102 * S], fill=white)
        d.ellipse([132 * S, 88 * S, 146 * S, 102 * S], fill=white)
        p([(102, 122), (118, 122), (110, 132)], pink, None)  # нос
        if mouth_open:
            e((92, 136, 128, 182), (60, 15, 25))
            d.ellipse([100 * S, 160 * S, 120 * S, 180 * S], fill=pink)
        else:
            d.arc([92 * S, 126 * S, 112 * S, 146 * S], 20, 160, fill=o, width=4 * S)
            d.arc([108 * S, 126 * S, 128 * S, 146 * S], 20, 160, fill=o, width=4 * S)
        for y, dy in ((138, -8), (148, 0), (158, 8)):         # усы
            d.line([(70 * S, y * S), (20 * S, (y + dy) * S)], fill=o, width=3 * S)
            d.line([(150 * S, y * S), (200 * S, (y + dy) * S)], fill=o, width=3 * S)
    return aa(im, (220, 300))


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


CAT_F = draw_cat(True, False)
CAT_FO = draw_cat(True, True)
CAT_B = draw_cat(False)
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


def beat_pulse(t):
    if t < DROP or (BOOM <= t < BOOM + 0.6):
        return 0.0
    return math.exp(-((t - DROP) % BEAT) / 0.09)


def paste_c(frame, img, cx, cy, scale=1.0):
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    frame.alpha_composite(img, (round(cx - img.width / 2), round(cy - img.height / 2)))


def spin_cat(frame, cx, cy, theta, scale, mouth):
    c = math.cos(theta)
    spr = (CAT_FO if mouth else CAT_F) if c >= 0 else CAT_B
    w = max(4, round(spr.width * scale * max(abs(c), 0.07)))
    h = round(spr.height * scale)
    img = spr.resize((w, h), Image.BILINEAR)
    frame.alpha_composite(img, (round(cx - w / 2), round(cy - h / 2)))


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
    speed = 9.0 if s["panel"] == "p9" else 5.0
    if t >= DROP:
        theta = (t - DROP) * speed * 2 * math.pi
        mouth = ((t - DROP) // (BEAT / 2)) % 4 in (0, 3)
        if t < DROP + 0.35:                       # падение сверху
            k = (t - DROP) / 0.35
            cx, cy, sc = fx + 170, -150 + (fy + 40 + 150) * k * k, 0.8
        else:
            orb = (t - DROP - 0.35) * (2.6 if s["panel"] == "p9" else 1.2) * 2 * math.pi
            rx, ry = (250, 150)
            cx = min(W - 70, max(70, fx + rx * math.cos(orb)))
            cy = fy + 40 + ry * math.sin(orb)
            sc = 0.9 + 0.2 * math.sin(orb)
        spin_cat(frame, cx + sx, cy + sy, theta, sc * (1 + 0.12 * bp), mouth)
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


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ar(n, 0.002, 0.14)


def hat():
    n = int(0.06 * SR)
    x = np.diff(np.random.randn(n + 1))
    return 0.25 * x * env_ar(n, 0.001, 0.015)


def clap():
    n = int(0.2 * SR)
    x = np.diff(np.random.randn(n + 1)) * 0.5
    e = env_ar(n, 0.001, 0.05)
    return 0.5 * x * e


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


VOWELS = {"u": (380, 900, 2500), "i": (330, 2700, 3400), "a": (900, 1400, 2900)}


def uiia_voice(total):
    """Кошачье 'у-и-и-а' формантным синтезом поверх бита."""
    keys = []          # (time, f0, vowel, amp)
    t = DROP
    while t < DUR - 0.3:
        fast = BOOM + 0.6 <= t < 10.5
        if BOOM <= t < BOOM + 0.6:
            t = BOOM + 0.6
            continue
        step = BEAT / (4 if fast else 2)
        up = 1.26 if fast else 1.0
        for v, f0 in (("u", 560), ("i", 840), ("i", 840), ("a", 660)):
            keys.append((t, f0 * up, v, 1.0))
            keys.append((t + step * 0.8, f0 * up * (0.88 if v == "a" else 1.0), v, 0.9))
            keys.append((t + step * 0.97, f0 * up, v, 0.15 if v == "i" else 0.6))
            t += step
    # финальное протяжное «А-а-а»
    keys.append((DUR - 0.3, 700, "a", 1.0))
    keys.append((DUR - 0.05, 420, "a", 0.0))
    kt = np.array([k[0] for k in keys])
    n = total
    ts = np.arange(n) / SR
    f0 = np.interp(ts, kt, [k[1] for k in keys])
    amp = np.interp(ts, kt, [k[3] for k in keys])
    amp[ts < DROP] = 0
    amp[(ts >= BOOM) & (ts < BOOM + 0.6)] = 0
    F = [np.interp(ts, kt, [VOWELS[k[2]][j] for k in keys]) for j in range(3)]
    f0 = f0 * (1 + 0.012 * np.sin(2 * np.pi * 6 * ts))
    ph = np.cumsum(f0) / SR
    src = 2 * (ph % 1) - 1
    src = src - 0.6 * np.concatenate([[0], src[:-1]])   # лёгкий high-shelf
    out = np.zeros(n)
    idx = np.nonzero(amp > 0)[0]
    lo, hi = idx.min(), idx.max() + 1
    for j, bw in enumerate((90, 140, 200)):
        r = math.exp(-math.pi * bw / SR)
        a1 = 2 * r * np.cos(2 * np.pi * F[j][lo:hi] / SR)
        a2 = r * r
        x = src[lo:hi].tolist()
        a1l = a1.tolist()
        y = [0.0] * len(x)
        y1 = y2 = 0.0
        g = 1 - r
        for i in range(len(x)):
            yi = g * x[i] + a1l[i] * y1 - a2 * y2
            y[i] = yi
            y2, y1 = y1, yi
        out[lo:hi] += np.array(y) * (1.0, 0.6, 0.3)[j]
    out *= np.convolve(amp, np.ones(200) / 200, mode="same")
    return out / (np.abs(out).max() + 1e-9)


def make_audio(path):
    np.random.seed(3)
    n = int(DUR * SR)
    drums = np.zeros(n)
    sfx = np.zeros(n)
    t = DROP
    b = 0
    while t < DUR - 0.2:
        if not (BOOM <= t < BOOM + 0.6):
            add(drums, kick(), t)
            add(drums, hat(), t + BEAT / 2)
            if b % 2 == 1:
                add(drums, clap(), t)
        t += BEAT
        b += 1
    # тихое тиканье часов до дропа
    for k in range(3):
        add(sfx, 0.15 * hat(), 0.2 + k * 0.45)
    add(sfx, boing(), DROP)
    add(sfx, croak(), 3.6)
    add(sfx, croak(), 7.65)
    add(sfx, boom(), BOOM)
    for x in SCENES[1:]:
        add(sfx, 0.4 * np.random.randn(int(0.12 * SR)) * env_ar(int(0.12 * SR), 0.001, 0.04), x["t0"])
    voice = uiia_voice(n)
    mix = 0.55 * drums + 0.5 * voice + 0.6 * sfx
    fade = np.ones(n)
    fade[-int(0.3 * SR):] = np.linspace(1, 0, int(0.3 * SR))
    mix = np.tanh(1.3 * mix * fade) * 0.9
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


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
