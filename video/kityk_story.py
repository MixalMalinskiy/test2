"""«Китык знакомится» — длинный плюшевый кот рассказывает о себе голосом Коли (Mr Jabec), ~25 c.

Usage: python3 kityk_story.py <out.mp4>
Голос Коли — банк его родной речи (~30 c, 12 фраз), вытащенный из YouTube-роликов 1loxxy
(«Жаби Коля Лучшее за Март 2026», «Коля читает новости») через vidIQ и отобранный по голосу
(ECAPA): video/assets/kolya_bank/*.wav, kolya_bank.wav и kolya_f5ref_news.wav.
Каждая фраза: 3 дубля F5-TTS по образцу «новостей» + перекраска лучшего Seed-VC
(github.com/Plachtaa/seed-vc, путь SEED_VC_DIR) по всему банку; побеждает вариант,
ближе всего к среднему голосу Коли и разборчивее всего (whisper).
Фото Китыка — kityk_photos/, УИИА-кот и его звук — из make_video.py.
"""
import hashlib
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import kityk
import make_video as mv

W, H, FPS, SR = mv.W, mv.H, mv.FPS, mv.SR
SEED_VC_DIR = os.environ.get("SEED_VC_DIR", "/root/seed-vc")
KOLYA_BANK_DIR = mv.asset("kolya_bank")        # 12 родных фраз Коли
KOLYA_BANK = mv.asset("kolya_bank.wav")         # они же подряд, ~30 c — образец для Seed-VC
KOLYA_F5_REF = (mv.asset("kolya_f5ref_news.wav"),
                "С+амая гл+авная н+овость. Ам+ерика посс+орилась с С+ирией. "
                "И не т+олько с С+ирией. Он+а посс+орилась с Л+ивией, Ир+аном.")
F5_TAKES = 3
CAT_SPIN_SRC = (2.8, 4.45)                 # кусок оригинала, где кот крутится (и звук)
KITYK_SPIN_SRC = (6.25, 9.2)               # звук УИИА, под который крутится Китык

# Сюжет: act — что делает Китык; say — фраза голосом Коли («+» перед ударной гласной)
EVENTS = [
    dict(act="intro", say="Прив+ет. Я Кит+ык.", bubble="Привет. Я Китык.", gap=0.6),
    dict(act="pan", say="Я +очень дл+инный кот. Ц+елый метр.", bubble="Я очень длинный кот.\nЦелый метр.", tail=1.2),
    dict(act="turn", say="А с+зади я пох+ож на бат+он.", bubble="А сзади я\nпохож на батон.", gap=0.6, tail=0.6),
    dict(act="cat", say="+Ой. А +это кто так кр+утится?", bubble="Ой. А это кто\nтак крутится?", gap=1.4),
    dict(act="ready", say="Л+адно. Я т+оже так ум+ею.", bubble="Ладно.\nЯ тоже так умею.", gap=0.3),
    dict(act="spin", dur=KITYK_SPIN_SRC[1] - KITYK_SPIN_SRC[0], gap=0.1),
    dict(act="dizzy", say="+Ой. Гол+ова кр+ужится. Я сп+ать.", bubble="Ой... голова кружится.\nЯ спать.", gap=0.3),
    dict(act="sleep", dur=2.6, gap=0.2),
]
LABELS = {"pan": "длина: 1 метр", "turn": "вид сзади", "sleep": "Китык. Метр счастья."}


# --------------------------------------------------------------------------- голос
def kolya_center():
    """Средний голос Коли по его родным фразам (вектор ECAPA)."""
    vecs = [mv.speaker_vec(os.path.join(KOLYA_BANK_DIR, f)) for f in sorted(os.listdir(KOLYA_BANK_DIR))]
    c = np.mean(vecs, 0)
    return c / np.linalg.norm(c)


def kolya_voice(text):
    """Лучший из: F5_TAKES дублей F5 и перекраски лучшего дубля Seed-VC — по похожести на Колю + разборчивости."""
    import difflib
    path = mv.asset(f"kityk_voice2_{hashlib.md5(text.encode()).hexdigest()[:10]}.wav")
    if not os.path.exists(path):
        m = mv.voice_models()
        center = kolya_center()
        norm = lambda x: "".join(c for c in x.lower().replace("ё", "е") if c.isalnum())  # noqa: E731

        def score(wav):
            heard = " ".join(seg.text for seg in m["whisper"].transcribe(mv.pcm16k(wav), language="ru")[0])
            sim = float(mv.speaker_vec(wav) @ center)
            clarity = difflib.SequenceMatcher(None, norm(text), norm(heard)).ratio()
            print(f"    похожесть на Колю {sim:.2f}, разборчивость {clarity:.2f}: «{heard.strip()}»", flush=True)
            return sim + 0.5 * clarity, sim

        cands = []
        for take in range(F5_TAKES):
            tmp = path.replace(".wav", f"_f5_{take}.wav")
            print(f"  F5 #{take}: {text}", flush=True)
            m["f5"].infer(ref_file=KOLYA_F5_REF[0], ref_text=KOLYA_F5_REF[1], gen_text=text, file_wave=tmp,
                          nfe_step=32, seed=take, show_info=lambda *a, **k: None)
            cands.append((*score(tmp), tmp))
        best_f5 = max(cands)[2]
        out_dir = path.replace(".wav", "_vc")
        os.makedirs(out_dir, exist_ok=True)
        print(f"  Seed-VC: {text}", flush=True)
        subprocess.run([sys.executable, "inference.py", "--source", best_f5, "--target", KOLYA_BANK, "--output", out_dir,
                        "--diffusion-steps", "50", "--inference-cfg-rate", "0.7", "--fp16", "False"],
                       cwd=SEED_VC_DIR, check=True, capture_output=True)
        vc = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")][0]
        cands.append((*score(vc), vc))
        best = max(cands)
        print(f"  выбран: похожесть {best[1]:.2f} ({os.path.basename(best[2])})", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", best[2], "-ac", "1", "-ar", str(SR), path], check=True)
    a = mv.decode_audio(path)
    loud = np.nonzero(np.abs(a) > 0.03 * np.abs(a).max())[0]
    a = a[max(0, loud[0] - int(0.05 * SR)):loud[-1] + int(0.08 * SR)]
    return a / (np.abs(a).max() + 1e-9)


def build_timeline():
    t = 0.0
    for ev in EVENTS:
        if "say" in ev:
            ev["audio"] = kolya_voice(ev["say"])
            ev["speech"] = len(ev["audio"]) / SR
        ev["start"] = t + ev.get("gap", 0.0)
        ev["talk_at"] = ev["start"]
        ev["end"] = ev["start"] + ev.get("dur", ev.get("speech", 0.0)) + ev.get("tail", 0.3)
        t = ev["end"]
    return t + 0.3


def event_at(t):
    cur = EVENTS[0]
    for ev in EVENTS:
        if t >= ev["start"] - ev.get("gap", 0.0):
            cur = ev
    return cur


# --------------------------------------------------------------------------- картинка
def blanket_background():
    """Его же одеяло с фото: размытое и чуть темнее — Китык поверх выглядит «на своём месте»."""
    im = Image.open(os.path.join(kityk.PHOTOS, "front1.jpg")).convert("RGB")
    cw = round(im.height * W / H)
    im = im.crop(((im.width - cw) // 2, 0, (im.width - cw) // 2 + cw, im.height)).resize((W, H), Image.LANCZOS)
    im = im.filter(ImageFilter.GaussianBlur(22))
    return Image.blend(im, Image.new("RGB", (W, H), (10, 12, 20)), 0.25).convert("RGBA")


def ruler(height):
    """Рулетка сбоку: жёлтая лента с делениями (подпись «100 см» рисуется отдельно)."""
    img = Image.new("RGBA", (70, height + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 20, 36, 20 + height], fill=(250, 210, 40), outline=(40, 30, 0), width=3)
    for i in range(11):
        y = 20 + height * i / 10
        d.line([(10, y), (10 + (24 if i % 5 == 0 else 13), y)], fill=(40, 30, 0), width=3)
    return img


def render(fi, K, L, voice_env, bg):
    t = fi / FPS
    ev = event_at(t)
    act, lt = ev["act"], t - ev["start"]
    frame = bg.copy()
    env = voice_env[min(fi, len(voice_env) - 1)]

    # освещение по сценам: при кручении — цветные вспышки, во сне — темнеет
    if act == "spin":
        hue = (t * 1.7) % 1
        frame.alpha_composite(Image.new("RGBA", (W, H), (*[int(c) for c in kityk._hsv(hue, 0.8, 1)], 70)))
    sleep_dark = 0.0
    if act == "sleep":
        sleep_dark = min(1, lt / 1.0)
        frame.alpha_composite(Image.new("RGBA", (W, H), (5, 5, 25, int(150 * sleep_dark))))

    # поза Китыка
    spr, wscale, angle, cam, cam_y = K["front1"], 1.0, 0.0, 1.0, 0.0
    if act == "pan":                                   # камера едет по длинному коту сверху вниз
        k = min(1, max(0, lt / max(0.1, ev["end"] - ev["start"] - 0.4)))
        cam = 1.0 + 0.7 * math.sin(math.pi * k)
        cam_y = -330 + 660 * k
    elif act == "turn":                                # разворот спиной
        th = min(1, lt / 0.5) * math.pi
        spr, wscale = (K["front1"], abs(math.cos(th))) if th < math.pi / 2 else (K["back"], abs(math.cos(th)))
    elif act in ("cat", "ready"):
        if act == "cat" and lt < 0.5:                  # разворот обратно лицом
            th = math.pi * (1 - lt / 0.5)
            spr, wscale = (K["back"], abs(math.cos(th))) if th > math.pi / 2 else (K["front1"], abs(math.cos(th)))
        if act == "cat":
            angle = -6 * min(1, max(0, lt - 0.5) / 0.4)    # с интересом наклоняется к коту
        else:
            spr = K["front2"] if int(lt * 6) % 2 else K["front1"]   # кивает-разминается
    elif act == "spin":
        th = lt * 3.5 * 2 * math.pi
        spr, wscale = (K["front1"], abs(math.cos(th))) if math.cos(th) >= 0 else (K["back"], abs(math.cos(th)))
    elif act == "dizzy":
        angle = 14 * math.sin(lt * 5) * math.exp(-lt / 2.5)
    elif act == "sleep":
        angle = -8 * min(1, lt / 0.8)

    talk = 0.06 * env if "say" in ev and ev["talk_at"] <= t < ev["talk_at"] + ev.get("speech", 0) else 0.0
    breathe = 0.012 * math.sin(t * 2.4)
    h = 980 * cam * (1 + talk + breathe)
    s = h / spr.height
    img = spr.resize((max(4, round(spr.width * s * max(wscale, 0.06) * (1 - talk / 3))), round(h)), Image.BILINEAR)
    if angle:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
    cx, cy = W / 2, H / 2 + 60 - cam_y * cam
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse([cx - img.width * 0.55 + 30, cy - img.height / 2 + 40,
                                    cx + img.width * 0.55 + 30, cy + img.height / 2 + 40], fill=(0, 0, 0, 90))
    frame.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(25)))
    mv.paste_c(frame, img, cx, cy)

    if act == "pan":                                   # рулетка вдоль Китыка
        r = L["ruler"].resize((L["ruler"].width, max(10, round(img.height * 0.98))), Image.BILINEAR)
        frame.alpha_composite(r, (round(cx + img.width / 2 + 10), round(cy - r.height / 2)))
        mv.paste_c(frame, L["cm"], cx + img.width / 2 + 120, cy)

    # УИИА-кот: заходит справа и крутится (его кадры и звук — 1 в 1 из оригинала)
    if act in ("cat", "ready"):
        appear = min(1, (t - EVENTS[3]["start"] + EVENTS[3]["gap"]) / 0.4)
        k = int(CAT_SPIN_SRC[0] * FPS) + int((t * FPS)) % int((CAT_SPIN_SRC[1] - CAT_SPIN_SRC[0]) * FPS)
        cat, _ = K["cat"][k]
        mv.paste_c(frame, cat, W - 120 + (1 - appear) * 300, 1000, 0.5)

    if act == "dizzy":                                 # звёздочки вокруг головы
        for i in range(5):
            a = t * 4 + i * 2 * math.pi / 5
            mv.paste_c(frame, L["star"], cx + 130 * math.cos(a), cy - img.height / 2 + 60 + 30 * math.sin(a))
    if act == "sleep":
        for i in range(3):
            ph = (lt * 0.5 + i / 3) % 1
            mv.paste_c(frame, L["z"], cx + 120 + 60 * ph, cy - img.height / 2 + 80 - 160 * ph, 0.5 + 0.7 * ph)
    if act == "spin":
        mv.paste_c(frame, L["uiia"], W / 2, 150, 0.95 + 0.1 * math.sin(t * 20))

    # подписи: имя в начале, ярлыки сцен внизу, пузырь с репликой
    if act == "intro" and lt > 0.2:
        mv.paste_c(frame, L["name"], W / 2, 1150, min(1, (lt - 0.2) / 0.25))
    if act in LABELS:
        mv.paste_c(frame, L["labels"][act], W / 2, 1180 if act != "sleep" else 1150, min(1, lt / 0.25))
    if "say" in ev and ev["talk_at"] <= t < ev["talk_at"] + ev["speech"] + 0.4:
        b = L["bubbles"][ev["act"]]
        mv.paste_c(frame, b, 30 + b.width / 2, 120 + b.height / 2, 0.6 + 0.4 * min(1, (t - ev["talk_at"]) / 0.15))

    if t < 0.3:
        frame.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(255 * (1 - t / 0.3)))))
    return frame.convert("RGB")


def bubble_right_tail(txt):
    """Пузырь с хвостиком вниз-вправо — к голове Китыка."""
    return mv.bubble_img(txt, tail_left=False, size=36)


def main():
    out = sys.argv[1]
    dur = build_timeline()
    cat_path = mv.fetch_tiktok(mv.CAT_TT)
    cat = [mv.to_sprite(f) for f in mv.cached_sequence("cat", mv.read_frames(cat_path, 0, 14.0, FPS), mv.cat_cutout)]
    K = dict(front1=kityk.kityk_cutout("front1"), front2=kityk.kityk_cutout("front2"),
             back=kityk.kityk_cutout("back"), cat=cat)

    n = int(dur * SR)
    voice, music = np.zeros(n), np.zeros(n)
    for ev in EVENTS:
        if "audio" in ev:
            mv.add(voice, ev["audio"], ev["talk_at"])
    cat_ev, spin_ev = EVENTS[3], EVENTS[5]
    uiia = mv.decode_audio(cat_path, CAT_SPIN_SRC[0], CAT_SPIN_SRC[1] - CAT_SPIN_SRC[0])
    mv.add(music, 0.55 * uiia / np.abs(uiia).max(), cat_ev["start"] - cat_ev["gap"])
    spin = mv.decode_audio(cat_path, KITYK_SPIN_SRC[0], KITYK_SPIN_SRC[1] - KITYK_SPIN_SRC[0])
    mv.add(music, 0.9 * spin / np.abs(spin).max(), spin_ev["start"])
    mix = np.tanh(1.1 * (0.95 * voice + music))
    fade = np.ones(n)
    fade[-int(0.4 * SR):] = np.linspace(1, 0, int(0.4 * SR))
    wav = out.rsplit(".", 1)[0] + "_audio.wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * fade * 32767 * 0.95).astype(np.int16).tobytes())
    frames = int(dur * FPS)
    ve = mv.envelope(voice, frames)
    voice_env = np.convolve(ve / (ve.max() + 1e-9), np.ones(3) / 3, mode="same")

    rul = ruler(600)
    L = dict(
        name=mv.text_img("КИТЫК", 110, fill=(255, 255, 255), stroke=9, stroke_fill=(230, 60, 120)),
        labels={k: mv.text_img(v, 48) for k, v in LABELS.items()},
        bubbles={ev["act"]: bubble_right_tail(ev["bubble"]) for ev in EVENTS if "bubble" in ev},
        uiia=mv.text_img("УИИА  УИИА", 80, fill=(255, 255, 255), stroke=8, stroke_fill=(220, 30, 90)),
        star=mv.text_img("★", 46, fill=(255, 230, 60), stroke=3),
        z=mv.text_img("Z", 70, fill=(200, 220, 255), stroke=4, stroke_fill=(20, 30, 80)),
        ruler=rul, cm=mv.text_img("100 см", 34, fill=(255, 255, 255), stroke=5),
    )
    bg = blanket_background()
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for fi in range(frames):
        ff.stdin.write(render(fi, K, L, voice_env, bg).tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)


if __name__ == "__main__":
    main()
