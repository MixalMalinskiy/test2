"""Коля (Mr Jabec) впервые видит её фронталку: УИИА-кот, рыба и жаба — ~18 c, 720x1280.

Usage: python3 make_video.py <meme_grid.png> <out.mp4>
Требует: ffmpeg, pip install numpy pillow opencv-python-headless "rembg[cpu]" coqui-tts,
torch/torchaudio 2.8 (CPU). Новые реплики озвучиваются клоном родных голосов (XTTS v2,
COQUI_TOS_AGREED=1).

Всё берётся из оригиналов, 1 в 1:
  * кот и звук — TikTok «uia original cat» (green screen), фон вырезается rembg;
  * мелкая жаба Mr Jabec и фон-лес — TikTok 1loxxy;
  * диалог собран из фраз разных роликов 1loxxy родными голосами (см. LINES);
    голос Ларго отдан рыбе;
  * концовка — жаба поёт «Only a genius could love a woman» (1loxxy «Только Коля мог полюбить…»);
  * новые реплики (say=...) — клон голосов Коли и Ларго из этих же роликов;
  * её кадры на экране телефона (спит, фронталка, маска, холодильник, диван) и рыба —
    из присланной картинки, меняются по сюжету.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 720, 1280, 30
DUR = 18.0                # пересчитывается в build_schedule() по длине реплик
SR = 44100
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
SYMBOL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"   # для ❤
OUT_LINE = (25, 20, 20)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CAT_TT = "7484505655684893958"      # vm.tiktok.com/ZGdCUD1LH — uia original cat
TOAD_TT = "7616371658462399760"     # vm.tiktok.com/ZGdCUHfw6 — жаба (1loxxy)
LOVE_TT = "7686588987435158802"     # 1loxxy: «Только Коля мог полюбить такую женщину, как она»
LOVE_T0, LOVE_T1 = 12.6, 16.0       # крупный план жабы на белом фоне под «only a genius could love a woman»
TOAD_T0, TOAD_T1 = 5.0, 6.7         # крупный план жабы с горшком золота, она говорит «Пять.»
TOAD_IDLE = 10                      # первые кадры — жаба молчит
BG_T = 1.3                          # кадр с пустой лесной тропинкой
CAT_TEXT_BOXES = [(190, 725, 386, 772),   # «Thank you.» поверх кота (в кадре 576x1024)
                  (60, 236, 215, 305)]    # водяной знак Vizard.ai
GRID_FRONTALKA = (0, 1094, 541, 1448)
FISH_BOX = (320, 30, 541, 300)      # рыба внутри панели «фронталка»
# её кадры из присланной картинки: панель в сетке, кусок без подписей внутри неё, X её лица в куске
SHOTS = {
    "sleep": ((0, 0, 541, 254), (200, 40, 541, 185), 200),
    "fridge": ((0, 257, 541, 524), (110, 45, 541, 185), 215),
    "mask": ((0, 528, 541, 815), (15, 42, 285, 222), 160),
    "frontalka": ((0, 1094, 541, 1448), (25, 45, 305, 300), 140),
    "couch": ((546, 1094, 1086, 1448), (40, 40, 470, 295), 190),
}
PREVIEW = (300, 270)        # окно превью камеры на экране телефона
FISH_TT = "7688388792935501063"     # 1loxxy: «Коля, ты что, не можешь говорить?» — голос Ларго
SONG_TT = "7690632407040494856"     # 1loxxy: Коля поёт «Твої очі — ліхтарики…»
# образцы голосов для клона: (ролик, начало, конец)
VOICE_REFS = {
    "toad": [(TOAD_TT, 0.0, 2.25), (TOAD_TT, 5.15, 5.75)],
    "fish": [(TOAD_TT, 2.3, 4.65), (TOAD_TT, 6.25, 10.2), (FISH_TT, 0.0, 2.45), (FISH_TT, 3.2, 4.55)],
}
VOICE_SPEED = {"toad": 1.5, "fish": 1.2}


def asset(name):
    os.makedirs(ASSETS, exist_ok=True)
    return os.path.join(ASSETS, name)


CACHED_NAMES = {CAT_TT: "cat_tiktok.mp4", TOAD_TT: "toad_tiktok.mp4", LOVE_TT: "love_tiktok.mp4"}


def fetch_tiktok(video_id, name=None):
    path = asset(name or CACHED_NAMES.get(video_id, f"tt_{video_id}.mp4"))
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
        rgba = cutout(rgb)
        if not edges_opaque(rgba):
            return rgba
        # бледный однотонный фон rembg иногда берёт целиком — режем по цвету фона с краёв
        edge = np.concatenate([f[:8].reshape(-1, 3), f[:, :8].reshape(-1, 3), f[:, -8:].reshape(-1, 3)])
        dist = np.abs(f - np.median(edge, 0)).max(2)
        return np.dstack([rgb, (solid_alpha(dist < 28) * 255).astype(np.uint8)])
    # порог от зелёности краёв: на вспышках фон бледно-зелёный; дыры заливаются — размытый кот отливает зелёным
    alpha = solid_alpha(green >= min(70, 0.6 * np.median(border)))
    g = np.minimum(f[..., 1], np.maximum(f[..., 0], f[..., 2]) + 8)       # без зелёного ореола
    return np.dstack([f[..., 0], g, f[..., 2], alpha * 255]).clip(0, 255).astype(np.uint8)


def edges_opaque(rgba):
    """Верх и бока кадра непрозрачные — значит, вырезка захватила фон."""
    a = rgba[..., 3] > 128
    return np.concatenate([a[:6].ravel(), a[:, :6].ravel(), a[:, -6:].ravel()]).mean() > 0.5


def solid_alpha(bg_like):
    """Маска «не фон» -> крупные куски, дыры внутри залиты, края мягкие."""
    solid = keep_main(np.dstack([np.zeros(bg_like.shape + (3,), np.uint8), (~bg_like).astype(np.uint8) * 255]))
    n, lab = cv2.connectedComponents((solid[..., 3] == 0).astype(np.uint8))
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    inside = ~np.isin(lab, edge[edge > 0])                 # фон — всё, что касается края кадра
    return cv2.GaussianBlur(cv2.erode(inside.astype(np.float32), np.ones((3, 3))), (5, 5), 0)


def love_cutout(rgb):
    """Жаба на белом фоне: всё почти белое — прозрачное."""
    alpha = solid_alpha(rgb.min(2) > 236)
    return np.dstack([rgb, (alpha * 255).astype(np.uint8)])


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
    cat_path = fetch_tiktok(CAT_TT)
    toad_path = fetch_tiktok(TOAD_TT)
    print("cat frames...", flush=True)
    cat = []
    for f in cached_sequence("cat", read_frames(cat_path, 0, DUR, FPS), cat_cutout):
        # если вырезка всё же взяла фон целиком — кадр заменяем предыдущим
        cat.append(cat[-1] if edges_opaque(f) and cat else to_sprite(f))
    print("toad frames...", flush=True)
    toad = [to_sprite(f)[0] for f in cached_sequence("toad", read_frames(toad_path, TOAD_T0, TOAD_T1, FPS))]
    love_path = fetch_tiktok(LOVE_TT)
    love = [Image.fromarray(f, "RGBA") for f in
            cached_sequence("love", read_frames(love_path, LOVE_T0, LOVE_T1, FPS), love_cutout)]
    grid = Image.open(grid_path).convert("RGB")
    fish = cached_sequence("fish", [np.asarray(grid.crop(GRID_FRONTALKA).crop(FISH_BOX))])[0]
    phones = {k: draw_phone(grid.crop(panel).crop(box), fx) for k, (panel, box, fx) in SHOTS.items()}
    flash = Image.new("RGBA", phones[INTRO_SHOT].size, (255, 255, 255, 0))
    flash.putalpha(phones[INTRO_SHOT].getchannel("A"))
    bg = make_background(toad_path).filter(ImageFilter.GaussianBlur(2.5))
    # чуть темнее и виньетка — персонажи выделяются, следы стирания не видны
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.45 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2)
    bg = Image.fromarray((np.asarray(bg) * (0.85 * vig.clip(0.4, 1))[..., None]).astype(np.uint8))
    return dict(cat_path=cat_path, toad_path=toad_path, love_path=love_path, love=love, cat=cat, toad=toad,
                fish=to_sprite(fish)[0], phones=phones, phone_flash=flash, bg=bg)


# --------------------------------------------------------------------------- голоса и расписание
_XTTS = None


def clone_line(text, who):
    """Новая фраза голосом персонажа: XTTS v2 по образцам его родного голоса (с кэшем)."""
    global _XTTS
    import hashlib
    speed = VOICE_SPEED[who]
    path = asset(f"clone_{who}_{hashlib.md5(f'{text}|{speed}'.encode()).hexdigest()[:10]}.wav")
    if not os.path.exists(path):
        refs = []
        for i, (vid, a, b) in enumerate(VOICE_REFS[who]):
            ref = asset(f"ref_{who}_{i}.wav")
            if not os.path.exists(ref):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", fetch_tiktok(vid),
                                "-ac", "1", "-ar", "24000", ref], check=True)
            refs.append(ref)
        if _XTTS is None:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            from TTS.api import TTS
            _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        print(f"  voice {who}: {text}", flush=True)
        _XTTS.tts_to_file(text=text, speaker_wav=refs, language="ru", speed=speed, file_path=path)
    a = decode_audio(path)
    loud = np.nonzero(np.abs(a) > 0.03 * np.abs(a).max())[0]   # срезаем тишину по краям
    return a[max(0, loud[0] - int(0.05 * SR)):loud[-1] + int(0.08 * SR)]


def build_schedule():
    """Раскладывает реплики подряд по их реальной длине; задаёт LOVE_AT, TOAD_SYNC и DUR."""
    global LOVE_AT, TOAD_SYNC, DUR
    t = 0.0
    for ln in LINES:
        if "src" in ln:
            vid, a, b = ln["src"]
            audio = decode_audio(fetch_tiktok(vid), a, b - a)
        else:
            audio = clone_line(ln["say"], ln["who"])
        ramp = int(0.03 * SR)                               # без щелчков на стыках
        audio[:ramp] *= np.linspace(0, 1, ramp)
        audio[-ramp:] *= np.linspace(1, 0, ramp)
        ln["audio"] = audio / (np.abs(audio).max() + 1e-9)
        ln["at"], ln["dur"] = t + ln["gap"], len(audio) / SR
        t = ln["at"] + ln["dur"]
        if ln.get("src", (None,))[0] == TOAD_TT:            # «Пять.» — кадры жабы 1 в 1 с голосом
            TOAD_SYNC = ln["at"] - (ln["src"][1] - TOAD_T0)
    LOVE_AT = t + 0.2
    DUR = round(LOVE_AT + (LOVE_T1 - LOVE_T0) + 0.3, 2)


# --------------------------------------------------------------------------- графика
def text_img(txt, size, fill=(255, 225, 40), stroke=6, stroke_fill=(10, 10, 10)):
    f = ImageFont.truetype(SYMBOL_FONT if txt == "\u2764" else FONT, size)
    bb = [int(round(v)) for v in ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox(
        (0, 0), txt, font=f, stroke_width=stroke, align="center", spacing=6)]
    im = Image.new("RGBA", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).multiline_text((4 - bb[0], 4 - bb[1]), txt, font=f, fill=fill, stroke_width=stroke,
                                      stroke_fill=stroke_fill, align="center", spacing=6)
    return im


def bubble_img(txt, tail_left=True, size=30):
    t = text_img(txt, size, fill=(15, 15, 15), stroke=0)
    pad = 16
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


def draw_phone(selfie, focus_x):
    """Телефон с открытой фронталкой: статус-бар 04:00, превью её кадра, кнопка спуска."""
    sw, sh, bez = 300, 640, 14
    im = Image.new("RGBA", (sw + 2 * bez + 40, sh + 2 * bez + 40), (0, 0, 0, 0))
    glow = Image.new("L", im.size, 0)
    ImageDraw.Draw(glow).rounded_rectangle([20, 20, im.width - 20, im.height - 20], radius=60, fill=150)
    im.putalpha(glow.filter(ImageFilter.GaussianBlur(14)))
    im = Image.composite(Image.new("RGBA", im.size, (255, 255, 230, 255)), im, im.getchannel("A"))
    d = ImageDraw.Draw(im)
    x0, y0 = 20, 20
    d.rounded_rectangle([x0, y0, x0 + sw + 2 * bez, y0 + sh + 2 * bez], radius=52, fill=(28, 28, 32),
                        outline=(120, 120, 130), width=3)
    sx, sy = x0 + bez, y0 + bez
    d.rounded_rectangle([sx, sy, sx + sw, sy + sh], radius=40, fill=(0, 0, 0))
    f_small, f_mode = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 19)
    d.text((sx + 28, sy + 14), "04:00", font=f_small, fill="white")
    d.rounded_rectangle([sx + sw - 62, sy + 18, sx + sw - 28, sy + 34], radius=4, outline="white", width=2)
    d.rectangle([sx + sw - 59, sy + 21, sx + sw - 52, sy + 31], fill=(255, 70, 60))       # 5% заряда
    d.ellipse([sx + sw / 2 - 9, sy + 12, sx + sw / 2 + 9, sy + 30], fill=(10, 10, 14))      # фронтальная камера
    pw, ph = PREVIEW                                          # кадр заполняет превью, лицо по центру
    k = max(pw / selfie.width, ph / selfie.height)
    big = selfie.resize((round(selfie.width * k), round(selfie.height * k)), Image.LANCZOS)
    x = min(max(0, round(focus_x * k - pw / 2)), big.width - pw)
    y = (big.height - ph) // 2
    im.paste(big.crop((x, y, x + pw, y + ph)), (sx, sy + 270 - ph // 2))
    for i, (txt, col) in enumerate((("ВИДЕО", "white"), ("ФОТО", (255, 210, 40)), ("ПОРТРЕТ", "white"))):
        w = d.textlength(txt, font=f_mode)
        d.text((sx + sw / 2 + (i - 1) * 95 - w / 2, sy + 450), txt, font=f_mode, fill=col)
    cx, cy = sx + sw / 2, sy + 545
    d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline="white", width=5)
    d.ellipse([cx - 31, cy - 31, cx + 31, cy + 31], fill="white")
    d.rounded_rectangle([sx + 34, cy - 24, sx + 82, cy + 24], radius=10, fill=(90, 90, 100))  # галерея
    d.ellipse([sx + sw - 82, cy - 24, sx + sw - 34, cy + 24], fill=(60, 60, 66))           # смена камеры
    d.arc([sx + sw - 72, cy - 14, sx + sw - 44, cy + 14], 200, 520, fill="white", width=3)
    return im


def paste_c(frame, img, cx, cy, scale=1.0, angle=0.0):
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    frame.alpha_composite(img, (round(cx - img.width / 2), round(cy - img.height / 2)))


CAPTION = "Коля впервые увидел\nеё фронталку"
LOVE_CAPTION = "Только гений мог полюбить\nтакую, как она"
INTRO_SHOT, LOVE_SHOT = "sleep", "couch"
# Сюжет. src — родная фраза (ролик, начало, конец; разметка whisper), say — новая фраза клоном
# того же голоса. gap — пауза перед репликой; shot — какой её кадр на телефоне (или (кадр, сдвиг)).
LINES = [
    dict(who="fish", src=(FISH_TT, 0.00, 2.45), gap=1.0, shot="frontalka",
         bubble="Коля, ты что,\nне можешь говорить?"),
    dict(who="toad", src=(TOAD_TT, 5.15, 5.75), gap=0.12, bubble="пять."),
    dict(who="toad", src=(SONG_TT, 4.20, 8.65), gap=0.1, shot=("mask", 2.4),
         bubble="твої очі — ліхтарики,\nпогляд — п’ять зірок, шок"),
    dict(who="fish", src=(FISH_TT, 3.20, 4.55), gap=0.1, bubble="Слава Богу!"),
    dict(who="fish", say="Коля, она ночью съела весь торт.", gap=0.25, shot="fridge",
         bubble="Коля, она ночью\nсъела весь торт"),
    dict(who="toad", say="Ква! Значит, это любовь!", gap=0.15, bubble="ква! значит, это любовь"),
]
LOVE_AT = TOAD_SYNC = 0.0   # считаются в build_schedule()
PHONE_C = (360, 520)
FISH_C = (160, 1010)
TOAD_C = (545, 990)


def speaking(t, who):
    return any(ln["who"] == who and ln["at"] <= t < ln["at"] + ln["dur"] for ln in LINES)


def current_shot(t):
    """(кадр на телефоне, когда он включился)."""
    shot, since = INTRO_SHOT, 0.0
    for ln in LINES:
        if "shot" in ln:
            key, off = ln["shot"] if isinstance(ln["shot"], tuple) else (ln["shot"], 0.0)
            if t >= ln["at"] + off:
                shot, since = key, ln["at"] + off
    return (LOVE_SHOT, LOVE_AT) if t >= LOVE_AT else (shot, since)


def render(fi, A, pulse, voice_env, layers):
    t = fi / FPS
    bp = pulse[min(len(pulse) - 1, fi)]
    z = 1.0 + 0.05 * t / DUR
    bg = A["bg"].resize((round(W * z), round(H * z)), Image.BILINEAR)
    x0, y0 = (bg.width - W) // 2, (bg.height - H) // 2
    frame = bg.crop((x0, y0, x0 + W, y0 + H)).convert("RGBA")

    # кот: оригинальные кадры в такт оригинальному звуку, орбита вокруг телефона
    cat_img, (ox, oy) = A["cat"][min(fi, len(A["cat"]) - 1)]
    orb = t * 0.5 * 2 * math.pi
    depth = math.sin(orb)                                  # >0 — кот перед телефоном
    cs = 0.45 + 0.08 * depth
    cx = PHONE_C[0] + 290 * math.cos(orb)
    cy = PHONE_C[1] + 60 + 90 * depth
    # центр исходного кадра кота ставим в (cx, cy) — так сохраняются его наезды камеры
    cat_pos = (cx + (cat_img.width / 2 - ox) * cs, cy + (cat_img.height / 2 - oy) * cs)
    if depth < 0:
        paste_c(frame, cat_img, *cat_pos, cs)
    shot, since = current_shot(t)
    phone = A["phones"][shot]
    if t - since < 0.25 and since > 0:                      # вспышка камеры при смене кадра
        phone = Image.blend(phone, A["phone_flash"], 0.85 * (1 - (t - since) / 0.25))
    paste_c(frame, phone, PHONE_C[0], PHONE_C[1] + 6 * math.sin(t * 2.2), 0.82 * (1 + 0.03 * bp),
            3 * math.sin(t * 1.7))
    if depth >= 0:
        paste_c(frame, cat_img, *cat_pos, cs)

    # рыба надувается, когда говорит
    ve = voice_env[min(len(voice_env) - 1, fi)] if speaking(t, "fish") else 0.0
    paste_c(frame, A["fish"], FISH_C[0], FISH_C[1] + 10 * math.sin(t * 3.1), 1.2 * (1 + 0.22 * ve + 0.04 * bp))
    frames = A["toad"]
    if TOAD_SYNC <= t < TOAD_SYNC + len(frames) / FPS and not speaking(t, "fish"):  # «Пять.» 1 в 1
        toad = frames[int((t - TOAD_SYNC) * FPS)]
    else:                                                  # туда-обратно, без скачка
        n = len(frames) if speaking(t, "toad") else TOAD_IDLE
        k = int(t * FPS) % (2 * n - 2)
        toad = frames[k if k < n else 2 * n - 2 - k]
    paste_c(frame, toad, TOAD_C[0], TOAD_C[1] - 8 * bp, 0.55)
    paste_c(frame, layers["name"], TOAD_C[0], 1240)

    love = t >= LOVE_AT
    lt = t - LOVE_AT if love else t
    k = min(1, lt / 0.3)
    paste_c(frame, layers["love_cap" if love else "cap"], W / 2, 110, 1 + 0.5 * (1 - k) * math.cos(k * 8))

    if love:                                               # сердечки от телефона и жаба-певец
        for i in range(9):
            ph = (lt * 0.45 + i / 9) % 1
            hx = PHONE_C[0] + 260 * math.sin(i * 2.4) + 20 * math.sin(lt * 3 + i)
            paste_c(frame, layers["heart"], hx, PHONE_C[1] + 260 - 520 * ph, 0.5 + 0.6 * math.sin(math.pi * ph))
        rise = (1 - min(1, lt / 0.35)) ** 2 * 700
        frame.alpha_composite(A["love"][min(len(A["love"]) - 1, int(lt * FPS))].resize((W, H), Image.BILINEAR),
                              (0, round(rise)))

    for ln, img in zip(LINES, layers["bubbles"]):
        who, b0, b1 = ln["who"], ln["at"], min(LOVE_AT, ln["at"] + ln["dur"] + 0.35)
        if b0 <= t < b1:
            s = 0.6 + 0.4 * min(1, (t - b0) / 0.15)
            if who == "fish":
                paste_c(frame, img, 24 + img.width / 2, 900 - img.height / 2, s)
            else:
                paste_c(frame, img, 704 - img.width / 2, 870 - img.height / 2, s)

    if t > DUR - 0.35:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (t - DUR + 0.35) / 0.35))))
    return frame.convert("RGB")


# --------------------------------------------------------------------------- звук
def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


def envelope(sig, frames):
    hop = SR // FPS
    return np.array([np.sqrt(np.mean(sig[k * hop:(k + 1) * hop] ** 2)) if k * hop < len(sig) else 0.0
                     for k in range(frames)])


def make_audio(A, path):
    n = int(DUR * SR)
    frames = int(DUR * FPS)
    music = np.zeros(n)
    orig = decode_audio(A["cat_path"], 0, DUR)              # оригинальный звук УИИА
    music[: min(n, len(orig))] = orig[:n]
    voice = np.zeros(n)
    duck = np.ones(n)
    for ln in LINES:                                       # голоса, фраза за фразой
        add(voice, ln["audio"], ln["at"])
        duck[int((ln["at"] - 0.1) * SR):int((ln["at"] + ln["dur"] + 0.1) * SR)] = 0.25
    song = decode_audio(A["love_path"], LOVE_T0, LOVE_T1 - LOVE_T0)
    add(voice, 0.9 * song / (np.abs(song).max() + 1e-9), LOVE_AT)
    duck[int(LOVE_AT * SR):] = 0.12
    duck = np.convolve(duck, np.ones(4000) / 4000, mode="same")
    music *= 0.85 / (np.abs(music).max() + 1e-9)
    mix = music * duck + 0.9 * voice
    fade = np.ones(n)
    fade[-int(0.35 * SR):] = np.linspace(1, 0, int(0.35 * SR))
    mix = np.tanh(1.1 * mix * fade) * 0.95
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())

    # пульс анимации — по атакам музыки; огибающая голоса — чтобы рыба «надувалась»
    onset = np.maximum(0, np.diff(envelope(music * duck, frames), prepend=0))
    onset /= np.percentile(onset[onset > 0], 95) + 1e-9
    pulse, p = np.zeros(frames), 0.0
    for k in range(frames):
        p = max(min(1.0, onset[k]), p * 0.72)
        pulse[k] = p
    ve = envelope(voice, frames)
    return pulse, np.convolve(ve / (ve.max() + 1e-9), np.ones(3) / 3, mode="same")


def main():
    grid, out = sys.argv[1], sys.argv[2]
    build_schedule()
    A = load_assets(grid)
    wav = out.rsplit(".", 1)[0] + "_audio.wav"
    pulse, voice_env = make_audio(A, wav)
    layers = dict(
        cap=text_img(CAPTION, 46),
        name=text_img("Mr Jabec", 32, fill=(140, 230, 90), stroke=5),
        love_cap=text_img(LOVE_CAPTION, 46, fill=(255, 150, 190)),
        heart=text_img("\u2764", 70, fill=(255, 60, 110), stroke=3, stroke_fill=(120, 0, 30)),
        bubbles=[bubble_img(ln["bubble"], ln["who"] == "fish") for ln in LINES],
    )
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for fi in range(int(DUR * FPS)):
        ff.stdin.write(render(fi, A, pulse, voice_env, layers).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


if __name__ == "__main__":
    main()
