#!/usr/bin/env python3
"""
split_commercials.py - cut a compilation of TV commercials into one clip per ad.

How it decides where one ad ends and the next begins:
  1. TransNetV2 (a neural shot-boundary detector) finds every cut in the video.
  2. Each cut is scored with signals that separate "new ad" from "new shot in the same ad":
       - blank / gray / noise frames between ads (works on gray VHS "black")
       - change in the sound (announcer voice, music, room tone) before vs. after the cut
       - change in the overall look of the picture before vs. after the cut
       - a dip in audio level at the cut
  3. A dynamic-programming pass picks the set of cuts that best fits real ad
     lengths (10/15/20/30/60/90/120 s) while keeping the strongest boundaries.
  4. Each ad is exported as an H.264/AAC MP4 with normalized loudness.
     Clips it isn't sure about (odd length, weak boundary, first/last partial clip)
     go to a _review subfolder instead, so NeXroll only picks up the confident ones.

Usage:
  python split_commercials.py "D:\\Tapes\\commercials_50s.mp4" -o "D:\\NeXroll\\Prerolls\\50s Ads"
  python split_commercials.py "D:\\Tapes" -o "D:\\NeXroll\\Prerolls\\Vintage"
  python split_commercials.py tape.mp4 -o out --dry-run          (just the cut list + contact sheet)

Requirements: ffmpeg + ffprobe on PATH, Python 3.10+,
  pip install torch transnetv2-pytorch librosa opencv-python-headless numpy
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

VERSION = "2026-11-16n"
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv", ".webm", ".vob"}

# Typical ad lengths and how common they are (relative). Off-length ads are allowed, just less likely.
STD_LENGTHS = {10: .25, 15: .6, 20: .6, 30: 1.0, 45: .3, 60: 1.0, 90: .1, 120: .1}

DEFAULT_W = dict(blank=3.0, audio=1.0, visual=0.5, shot=0.5, dip=0.3, bias=2.0,
                 duration=1.5, floor=0.2, seg_cost=1.0,
                 x_cchg=0.8, x_tchg=0.8, x_gchg=0.8, x_achg=0.0, x_vchg=0.0, zthr=1.5,
                 wipe=5.0, sep_penalty=1.5, separators=False)

BLANK_STD = 6.0      # spatial detail below this = blank/gray/noise frame
AUDIO_SR = 16000
HOP = 160            # 10 ms audio hop


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- ffmpeg helpers
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,avg_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    rate = st.get("avg_frame_rate") or st["r_frame_rate"]
    if rate in ("0/0", None):
        rate = st["r_frame_rate"]
    num, den = rate.split("/")
    return float(num) / float(den), float(j["format"]["duration"])


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())


# ----------------------------------------------------------------------------- analysis
def _lead(mask):
    n = 0
    for v in mask:
        if not v:
            break
        n += 1
    return n


def frame_extra(f, cv2):
    """Picture size (black borders), color, and texture/grain of one 64x36 frame."""
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dc = (g.mean(0) < 22) & (g.std(0) < 10)
    dr = (g.mean(1) < 22) & (g.std(1) < 10)
    L, R, T, B = _lead(dc), _lead(dc[::-1]), _lead(dr), _lead(dr[::-1])
    inner = g[T:36 - B, L:64 - R] if (L + R < 48 and T + B < 27) else g
    if L + R >= 48 or T + B >= 27 or inner.mean() < 45:   # dark frame: can't tell borders from scenery
        L = R = T = B = np.nan
    core = f[4:-4, 6:-6]
    yuv = cv2.cvtColor(core, cv2.COLOR_RGB2YUV).astype(np.float32)
    sat = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32).mean() / 255
    gc = g[4:-4, 6:-6]
    hf = float(np.abs(gc - cv2.GaussianBlur(gc, (3, 3), 0)).mean())
    return [L, R, T, B, yuv[..., 1].mean() - 128, yuv[..., 2].mean() - 128, sat, hf]


def stripe_stats(g_now, g_before):
    """How 'striped' the change between two frames is (venetian-blind style wipes):
    returns (stripe strength, number of stripes across the frame, how regular they are)."""
    d = np.abs(g_now - g_before)
    cp, rp = d.mean(0), d.mean(1)
    strength = float(np.abs(np.diff(cp)).mean() / (np.abs(np.diff(rp)).mean() + 0.5))
    P = np.abs(np.fft.rfft(cp - cp.mean())) ** 2
    P[0] = 0
    k = int(np.argmax(P))
    conc = float(P[k] / (P[1:].sum() + 1e-6))
    return strength, k, conc


def _spot_groups(mask, fps, min_count):
    idx = np.where(mask)[0]
    W = int(1.5 * fps)
    spots, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] <= W:
            j += 1
        grp = idx[i:j + 1]
        if max(((grp >= g) & (grp < g + 2 * W)).sum() for g in grp) >= min_count:
            spots.append((grp[0] / fps, (grp[-1] + 1) / fps))
        i = j + 1
    return spots


def separator_spots(stripe_info, fps, min_count=12, min_spots=4):
    """Find a transition effect the compiler of the tape put between ads (e.g. a venetian-blind
    wipe). Only used when the same striped pattern shows up at least `min_spots` times.
    Pass 1 finds the clear examples; pass 2 learns their stripe count and searches the whole
    tape again, more loosely, for the same pattern so faint wipes aren't missed."""
    s, k, conc = stripe_info
    if len(s) == 0:
        return []
    r = s / (np.percentile(s, 99) + 1e-6)
    strict = (k >= 6) & (conc >= 0.2) & (r > 0.3)
    spots = _spot_groups(strict, fps, min_count)
    if len(spots) < min_spots:
        return []
    inside = np.zeros(len(s), bool)
    for a, b in spots:
        inside[int(a * fps):int(b * fps)] = True
    ks = k[strict & inside]
    lo, hi = np.percentile(ks, 10) - 1, np.percentile(ks, 90) + 1
    loose = (k >= max(5, lo)) & (k <= hi) & (conc >= 0.12) & (r > 0.15)
    found = list(spots)
    for a, b in _spot_groups(loose, fps, max(5, min_count // 2)):
        if not any(a < y + 0.5 and b > x - 0.5 for x, y in found):
            found.append((a, b))
    return sorted(found)


def add_separators(cands, spots):
    for a, b in spots:
        near = [x for x in cands if a - 1.0 <= x["t"] <= b + 1.0]
        for x in near:
            cands.remove(x)
        a, b = a - 0.5, b + 0.5      # the stripes fade in/out around the detected middle of the wipe
        cands.append(dict(t=(a + b) / 2, start=a, end=b, wipe=True,
                          shot=max([x["shot"] for x in near], default=0.0),
                          blank=max([x["blank"] for x in near], default=0.0)))
    cands.sort(key=lambda x: x["t"])
    return cands


def analyze_video(path, fps, dur):
    """Stream frames once: TransNetV2 shot probabilities + blank/look descriptors per frame."""
    import torch, cv2
    from transnetv2_pytorch import TransNetV2
    model = TransNetV2(device="auto")
    model.eval()
    dev = model.device
    log(f"  shot detector running on {dev}")

    W, H = 64, 36
    fsize = W * H * 3
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf", f"scale={W}:{H}",
                             "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:"], stdout=subprocess.PIPE)
    CH, CTX = 1500, 50
    left, pend = [], []
    probs, stds, thumbs, extras = [], [], [], []
    stripe, sfreq, sconc = [], [], []
    hist = []
    expected = max(1, int(dur * fps))
    t0 = time.time()

    def run(frames):
        x = torch.from_numpy(np.stack(frames)).to(dev)
        with torch.no_grad():
            single, _ = model.predict_frames(x, quiet=True)
        return single.cpu().numpy()

    n = 0
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)[4:-4, 6:-6]       # drop VHS edges
        k = g.reshape(g.shape[0] // 2, 2, g.shape[1] // 2, 2).mean(axis=(1, 3))      # blur tape grain
        stds.append(float(k.std()))
        thumbs.append(k.ravel().astype(np.float16))
        extras.append(frame_extra(f, cv2))
        gc = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)[4:-4, 5:-5]
        hist.append(gc)
        if len(hist) > 3:
            hist.pop(0)
        if len(hist) == 3:
            st, kf, cc = stripe_stats(hist[2], hist[0])
        else:
            st, kf, cc = 0.0, 0, 0.0
        stripe.append(st); sfreq.append(kf); sconc.append(cc)
        pend.append(cv2.resize(f, (48, 27), interpolation=cv2.INTER_AREA))
        n += 1
        if len(pend) >= CH + CTX:
            p = run(left + pend[:CH + CTX])
            probs.append(p[len(left):len(left) + CH])
            left, pend = pend[CH - CTX:CH], pend[CH:]
            el = time.time() - t0
            log(f"  video {100 * n / expected:5.1f}%  ({n / max(el, 1e-6):.0f} frames/s)")
    proc.wait()
    if pend:
        p = run(left + pend)
        probs.append(p[len(left):])
    single = np.concatenate(probs) if probs else np.zeros(0, np.float32)
    m = min(len(single), len(stds))
    stripe_info = (np.array(stripe[:m], np.float32), np.array(sfreq[:m]), np.array(sconc[:m], np.float32))
    return (single[:m].astype(np.float32), np.array(stds[:m], np.float32), np.array(thumbs[:m]),
            np.array(extras[:m], np.float32), stripe_info)


def analyze_audio(path, dur):
    import librosa
    if not has_audio(path):
        n = int(dur * AUDIO_SR / HOP)
        return np.full(n, 1e-5, np.float32), np.zeros((n, 20), np.float32)
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                             "-f", "s16le", "pipe:"], stdout=subprocess.PIPE)
    chunk = AUDIO_SR * 60  # 60 s, multiple of HOP
    tail = np.zeros(0, np.float32)
    rms_all, mfcc_all = [], []
    while True:
        b = proc.stdout.read(chunk * 2)
        if not b:
            break
        y = np.concatenate([tail, np.frombuffer(b, np.int16).astype(np.float32) / 32768.0])
        usable = (len(y) - 512) // HOP * HOP if len(y) > 512 else 0
        if usable <= 0:
            tail = y
            continue
        seg = y[:usable + 512]
        rms_all.append(librosa.feature.rms(y=seg, frame_length=400, hop_length=HOP, center=False)[0][:usable // HOP])
        mfcc_all.append(librosa.feature.mfcc(y=seg, sr=AUDIO_SR, n_mfcc=20, n_fft=512, hop_length=HOP,
                                             center=False).T[:usable // HOP])
        tail = y[usable:]
    proc.wait()
    if not rms_all:
        n = int(dur * AUDIO_SR / HOP)
        return np.full(n, 1e-5, np.float32), np.zeros((n, 20), np.float32)
    return np.concatenate(rms_all).astype(np.float32), np.concatenate(mfcc_all).astype(np.float32)


def find_candidates(fps, single, stds):
    n = len(stds)
    cands = {}
    i = 0
    while i < n:                                     # runs of blank frames
        if stds[i] < BLANK_STD:
            j = i
            while j < n and stds[j] < BLANK_STD * 1.6:
                j += 1
            if j - i >= 2:
                cands[(i + j) / 2 / fps] = dict(blank=(j - i) / fps, start=i / fps, end=j / fps)
            i = j
        else:
            i += 1
    for k in range(1, n - 1):                        # shot cuts
        if single[k] > 0.3 and single[k] >= single[k - 1] and single[k] >= single[k + 1]:
            t = (k + 0.5) / fps
            near = [c for c in cands if abs(t - c) < 0.6]
            if near:
                for c in near:
                    cands[c]["shot"] = max(cands[c].get("shot", 0.0), float(single[k]))
                continue
            cands[t] = dict(blank=0.0, shot=float(single[k]), start=t, end=t)
    out = []
    for t in sorted(cands):
        c = cands[t]; c["t"] = t; c.setdefault("shot", 0.0); out.append(c)
    merged = []
    for c in out:
        if merged and c["t"] - merged[-1]["t"] < 0.6:
            m = merged[-1]
            if c["blank"] > m["blank"] or (c["blank"] == m["blank"] and c["shot"] > m["shot"]):
                c["shot"] = max(c["shot"], m["shot"]); merged[-1] = c
            else:
                m["shot"] = max(c["shot"], m["shot"])
        else:
            merged.append(c)
    return merged


def _win_med(ex, a, b):
    seg = ex[max(0, a):max(a + 1, b)]
    out = np.full(ex.shape[1], np.nan, np.float32)
    for k in range(ex.shape[1]):
        col = seg[:, k] if len(seg) else np.zeros(0)
        col = col[~np.isnan(col)]
        if len(col):
            out[k] = np.median(col)
    return out


def _border_steady(ex, a, b, min_valid=0.7, max_spread=1.0):
    seg = ex[max(0, a):max(a + 1, b), :4]
    if len(seg) == 0:
        return False
    ok = ~np.isnan(seg).any(axis=1)
    if ok.mean() < min_valid:
        return False
    v = seg[ok]
    spread = np.median(np.abs(v - np.median(v, axis=0)), axis=0).max()
    return bool(spread <= max_spread)


LOCAL_EPS = {"cchg": 0.5, "tchg": 0.1, "gchg": 1.0, "achg": 0.3, "vchg": 0.1}


def score_candidates(cands, fps, thumbs, rms, mfcc, extras=None, win=3.0, span=45.0):
    cands = _score_base(cands, fps, thumbs, rms, mfcc, win)
    if extras is None:
        for c in cands:
            c.update(cchg=0.0, tchg=0.0, gchg=0.0)
    else:
        for c in cands:
            lo, hi = c["start"], c["end"]
            L = _win_med(extras, int((lo - win) * fps), int((lo - 0.1) * fps))
            R = _win_med(extras, int((hi + 0.1) * fps), int((hi + win) * fps))
            g = np.abs(L[:4] - R[:4])
            steady = (_border_steady(extras, int((lo - win) * fps), int((lo - 0.1) * fps)) and
                      _border_steady(extras, int((hi + 0.1) * fps), int((hi + win) * fps)))
            # picture size change - only trusted when the borders are steady on both sides
            # (dark or spotlit scenes make the border measurement jump around)
            c["gchg"] = float(np.nansum(g)) if steady and not np.all(np.isnan(g)) else 0.0
            c["cchg"] = float(np.hypot(L[4] - R[4], L[5] - R[5]) + 20 * abs(L[6] - R[6])) \
                if not np.isnan(L[4] + R[4]) else 0.0                                     # color change
            c["tchg"] = float(abs(L[7] - R[7]) / (0.5 * (L[7] + R[7]) + 1e-6)) \
                if not np.isnan(L[7] + R[7]) else 0.0                                     # grain/sharpness change
    # how unusual each change is compared with the other cuts nearby (tapes differ a lot)
    ts = np.array([c["t"] for c in cands])
    for c in cands:
        near = np.where(np.abs(ts - c["t"]) <= span)[0]
        for k, e in LOCAL_EPS.items():
            v = np.array([cands[i][k] for i in near])
            med = np.median(v)
            c["z_" + k] = float((c[k] - med) / (np.median(np.abs(v - med)) * 1.4826 + e))
    return cands


def _score_base(cands, fps, thumbs, rms, mfcc, win=3.0):
    hop = HOP / AUDIO_SR
    db = 20 * np.log10(rms + 1e-5)
    mz = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-6)
    th = thumbs.astype(np.float32)
    thz = (th - th.mean(0)) / (th.std(0) + 1e-6)
    for c in cands:
        lo, hi = c["start"], c["end"]
        a, b = int((lo - 0.35) / hop), int((hi + 0.35) / hop)
        A, B = int((lo - 6) / hop), int((hi + 6) / hop)
        local = db[max(0, a):max(a + 1, b)]
        ctx = db[max(0, A):max(A + 1, B)]
        c["dip"] = float(np.median(ctx) - local.min()) if len(local) and len(ctx) else 0.0
        L = mz[max(0, int((lo - win) / hop)):max(1, int(lo / hop))]
        R = mz[int(hi / hop):int((hi + win) / hop)]
        c["achg"] = float(np.linalg.norm(L.mean(0) - R.mean(0)) + 0.5 * np.linalg.norm(L.std(0) - R.std(0))) \
            if len(L) > 10 and len(R) > 10 else 0.0
        Lv = thz[max(0, int((lo - win) * fps)):max(1, int((lo - 0.1) * fps))]
        Rv = thz[int((hi + 0.1) * fps):int((hi + win) * fps)]
        c["vchg"] = float(np.linalg.norm(Lv.mean(0) - Rv.mean(0)) / np.sqrt(Lv.shape[1])) \
            if len(Lv) > 3 and len(Rv) > 3 else 0.0
    return cands


def boundary_score(c, w):
    s = 0.0
    if c["blank"] >= 0.1:
        # A dark or gray moment only counts as an ad break if something actually changes across it
        # (sound or look). Night scenes and fades inside one ad keep the same music and look.
        gate = np.clip(0.5 * (c["achg"] - 1.8) / 1.2 + 0.5 * (c["vchg"] - 0.5) / 0.4, 0.0, 1.0)
        if c["blank"] >= 1.5:      # long gaps / tape static are never part of an ad
            gate = 1.0
        s += gate * (w["blank"] + (0.6 if c["blank"] >= 0.4 else 0.0))
    s += w["audio"] * np.clip(c["achg"] - 2.2, -2.0, 3.0)
    s += w["visual"] * np.clip((c["vchg"] - 0.9) / 0.4, -2.0, 2.0)
    s += w["shot"] * c["shot"]
    s += w["dip"] * np.clip((c["dip"] - 18) / 6, -1.5, 1.5)
    for k in ("cchg", "tchg", "gchg", "achg", "vchg"):
        s += w["x_" + k] * np.clip(c.get("z_" + k, 0.0) - w["zthr"], 0.0, 3.0)
    if c.get("wipe"):
        s += w["wipe"]              # the tape's own between-ads transition
    elif w.get("separators"):
        s -= w["sep_penalty"]       # on such tapes, plain cuts need more evidence
    return float(s - w["bias"])


def length_fit(d):
    # Ads in compilations often run SHORT of their nominal length (trimmed film/tape transfers),
    # but rarely run long, so the tolerance is wider on the short side.
    best = 0.0
    for L, wt in STD_LENGTHS.items():
        sd = max(2.5, 0.10 * L) if d < L else max(1.5, 0.04 * L)
        best = max(best, wt * np.exp(-((d - L) ** 2) / (2 * sd * sd)))
    return best


def duration_score(d, w):
    return w["duration"] * np.log(w["floor"] + length_fit(d)) - w["seg_cost"]


def split_merged_pairs(chosen, cands, total, w, min_len, thr2=-0.5, thr3=-1.75, fit_min=0.5):
    """Safety net: a long clip with an odd length that splits cleanly into two or three
    standard-length ads at reasonable cuts is almost always ads stuck together
    (e.g. 84s = 55s + 29s, 143s = 55s + 29s + 59s)."""
    changed = True
    while changed:
        changed = False
        bounds = [dict(end=0.0)] + chosen + [dict(start=total)]
        for k in range(len(bounds) - 1):
            a, b = bounds[k]["end"], bounds[k + 1]["start"]
            edge = k == 0 or k == len(bounds) - 2
            # first/last clips may be cut off by the recording, so only a clearly strong break splits them
            if edge and b - a < 40:
                continue
            if not edge and (b - a < 75 or length_fit(b - a) >= 0.35):
                continue
            inside = [c for c in cands if c["start"] - a >= min_len and b - c["end"] >= min_len]
            sc_ = {id(c): boundary_score(c, w) for c in inside}
            best = None
            ranked = sorted(sc_.values(), reverse=True)
            top = max(inside, key=lambda c: sc_[id(c)]) if inside else None
            # runner-up = best cut that is NOT part of the same transition (title cards, credits
            # and dissolves often add a second cut a second or two after the real break)
            others = [sc_[id(c)] for c in inside if top is not None and abs(c["t"] - top["t"]) > 3.0]
            runner = max(others) if others else -9.0
            margin_ok = lambda v: v >= 1.0 or (v >= 0.0 and v - runner >= 1.2)
            for i, c1 in enumerate(inside):
                f1 = length_fit(c1["start"] - a)
                f2 = length_fit(b - c1["end"])
                # strong = clearly the best break inside this clip, not just "above a fixed number",
                # because scores shift a little with what's around the clip on the tape
                strong = (sc_[id(c1)] == ranked[0] and margin_ok(sc_[id(c1)])
                          and c1["start"] - a >= 12 and b - c1["end"] >= 12)
                if strong and (edge or not (f1 >= fit_min and f2 >= fit_min)):
                    # a clearly strong break (new sound + new look) wins even if the pieces have odd lengths
                    val = sc_[id(c1)] - 2.0
                    if best is None or val > best[0]:
                        best = (val, [c1])
                if edge:
                    continue
                if f1 >= fit_min and f2 >= fit_min and sc_[id(c1)] > thr2:
                    val = sc_[id(c1)] + np.log(f1) + np.log(f2)
                    if best is None or val > best[0]:
                        best = (val, [c1])
                if f1 < fit_min:
                    continue
                for c2 in inside[i + 1:]:
                    if c2["start"] - c1["end"] < min_len:
                        continue
                    fm, fe = length_fit(c2["start"] - c1["end"]), length_fit(b - c2["end"])
                    if fm >= fit_min and fe >= fit_min and sc_[id(c1)] > thr3 and sc_[id(c2)] > thr3:
                        val = sc_[id(c1)] + sc_[id(c2)] + np.log(f1) + np.log(fm) + np.log(fe) - 1.0
                        if best is None or val > best[0]:
                            best = (val, [c1, c2])
            if best:
                chosen = sorted(chosen + best[1], key=lambda c: c["t"])
                changed = True
                break
    return chosen


def choose_cuts(cands, total, w, min_len, max_len):
    pts = [dict(t=0.0, start=0.0, end=0.0)] + cands + [dict(t=total, start=total, end=total)]
    n = len(pts)
    bs = [0.0] + [boundary_score(c, w) for c in cands] + [0.0]
    best = np.full(n, -1e18); prev = np.full(n, -1, int); best[0] = 0.0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            d = pts[j]["start"] - pts[i]["end"]
            whole = (i == 0 and j == n - 1)
            if d > max_len and not whole:
                if 0 < i:
                    break
                continue
            if d < min_len and not whole:
                continue
            if i == 0 or j == n - 1:
                # first/last clip may be cut off by the recording, so a short odd length is fine,
                # but a partial ad is never longer than a full one
                ds = w["duration"] * np.log(w["floor"] + (max(length_fit(d), 0.6) if d <= 125 else length_fit(d))) \
                    - w["seg_cost"]
            else:
                ds = duration_score(d, w)
            v = best[i] + ds + bs[j]
            if v > best[j]:
                best[j] = v; prev[j] = i
    path, j = [], n - 1
    while j > 0:
        path.append(j); j = prev[j]
    chosen = [pts[k] for k in path[::-1][:-1]]
    chosen = split_merged_pairs(chosen, cands, total, w, min_len)
    for c in chosen:
        c["score"] = boundary_score(c, w)
    return chosen


def junk_frames(stds, thumbs, fps):
    """True for black, gray, fuzzy or static frames (no real picture)."""
    t = thumbs.astype(np.float32)
    t = t - t.mean(1, keepdims=True)
    n = np.linalg.norm(t, axis=1) + 1e-6
    corr = np.ones(len(t), np.float32)
    corr[1:] = (t[1:] * t[:-1]).sum(1) / (n[1:] * n[:-1])
    blank = stds < BLANK_STD * 1.5
    # static/snow: each frame unrelated to the last one; only trust it over a run of ~0.3 s
    noisy = (corr < 0.3) & (stds < 30)
    run = max(2, int(0.3 * fps))
    sustained = np.zeros_like(noisy)
    i = 0
    while i < len(noisy):
        if noisy[i]:
            j = i
            while j < len(noisy) and noisy[j]:
                j += 1
            if j - i >= run:
                sustained[i:j] = True
            i = j
        else:
            i += 1
    return blank | sustained


def cut_into(t, side, db, junk, fps, a, b, dur, window=0.3, drop_db=6.0):
    """True if the ad is still going at a wipe edge: its sound at the cut is within `drop_db`
    of the clip's normal level and there is real picture inside the wipe next to it."""
    hop = HOP / AUDIO_SR
    body = db[int(a / hop):int(b / hop)]
    if len(body) == 0:
        return False
    edge = db[int(t / hop):int((t + window) / hop)] if side == "start" else db[int((t - window) / hop):int(t / hop)]
    if len(edge) == 0:
        return False
    loud = edge.max() - np.median(body) > -drop_db
    lo, hi = (t - 2.0, t) if side == "start" else (t, t + 2.0)
    lo, hi = max(0.0, lo), min(dur, hi)
    pic = junk[int(lo * fps):int(hi * fps)]
    picture = len(pic) > 0 and (1 - pic.mean()) > 0.5
    return bool(loud and picture)


def trim_junk(a, b, junk, fps, keep=1.0):
    """Shorten [a, b] so no more than `keep` seconds of black/static remain at either end."""
    ia, ib = int(a * fps), min(len(junk), int(b * fps))
    j = ib
    while j > ia and junk[j - 1]:
        j -= 1
    if (ib - j) / fps > keep:
        b = j / fps + keep
    i = ia
    while i < ib and junk[i]:
        i += 1
    if (i - ia) / fps > keep:
        a = i / fps - keep
    return a, b


# ----------------------------------------------------------------------------- export
def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60]


def export_clip(path, a, b, dst, crf):
    d = b - a
    af = f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:d=0.05,afade=t=out:st={max(0, d - 0.08):.3f}:d=0.08"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(path), "-t", f"{d:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", "setsar=1,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-af", af, "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
           "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def contact_sheet(path, segs, dst):
    import cv2
    rows = []
    for s in segs:
        tiles = []
        for t in (s["start"] + 0.3, (s["start"] + s["end"]) / 2, s["end"] - 0.3):
            p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", str(path), "-frames:v", "1",
                                "-vf", "scale=200:120", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
                               capture_output=True)
            img = np.frombuffer(p.stdout, np.uint8)
            tiles.append(img.reshape(120, 200, 3).copy() if img.size == 120 * 200 * 3 else np.zeros((120, 200, 3), np.uint8))
        label = np.full((120, 300, 3), 30, np.uint8)
        col = (80, 200, 255) if s["review"] else (120, 230, 120)
        cv2.putText(label, f"#{s['idx']:02d}  {s['dur']:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(label, f"{fmt(s['start'])} - {fmt(s['end'])}", (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        if s.get("note"):
            which = " + ".join(x for x, f in (("start", s.get("wipe_start")), ("end", s.get("wipe_end"))) if f)
            cv2.putText(label, f"CUT SHORT at {which} (wipe)", (8, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 1)
        cv2.putText(label, ("REVIEW: " + s["why"]) if s["review"] else "ok", (8, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        rows.append(np.hstack([label] + tiles))
    if rows:
        cv2.imwrite(str(dst), np.vstack(rows))


def fmt(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


# ----------------------------------------------------------------------------- main
def parse_times(text):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        secs = 0.0
        for p in part.split(":"):
            secs = secs * 60 + float(p)
        out.append(secs)
    return sorted(out)


def manual_split(path, out, args, dur):
    times = [t for t in parse_times(args.split_at) if 0.5 < t < dur - 0.5]
    edges = [0.0] + times + [dur]
    out.mkdir(parents=True, exist_ok=True)
    stem = safe(path.stem)
    segs = []
    with open(out / f"{stem}_segments.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["#", "start", "end", "seconds", "status", "reason", "transition note", "file"])
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            dst = out / f"{stem}_{chr(97 + i)}_{round(b - a)}s.mp4"
            if not args.dry_run:
                export_clip(path, a, b, dst, args.crf)
            log(f"  part {chr(97 + i)}: {fmt(a)} - {fmt(b)}  {b - a:5.1f}s -> {dst.name}")
            segs.append(dict(idx=i + 1, start=a, end=b, dur=b - a, review=False, why="", note=""))
            wr.writerow([i + 1, f"{a:.2f}", f"{b:.2f}", f"{b - a:.1f}", "ok", "split by hand", "",
                         "" if args.dry_run else dst.name])
    contact_sheet(path, segs, out / f"{stem}_segments.png")
    log(f"  done: split into {len(segs)} parts.")


def review_help(segs, out, stem, args):
    """Print (and save) what to check in each _review clip and the exact commands to re-split it."""
    rev = [s for s in segs if s["review"] and "blank" not in s["why"] and not args.skip_review]
    if not rev or getattr(args, "quiet_review", False):
        return
    me = Path(sys.argv[0]).name
    lines = ["CLIPS TO REVIEW",
             "Watch each clip below. If it holds two or more ads (e.g. two different products),",
             "run ONE of the commands under it. They only re-split that clip file, not the whole tape.", ""]
    for s in rev:
        clip = s["path"]
        lines.append(f"#{s['idx']:02d}  {clip.name}  ({s['dur']:.1f}s) - {s['why']}")
        lines.append(f'   auto:    python {me} "{clip}" -o "{clip.with_suffix("")}_split" --refine')
        lines.append(f'   harder:  python {me} "{clip}" -o "{clip.with_suffix("")}_split" --refine --sensitivity 1')
        if s.get("split_hints"):
            hints = ",".join(f"{t:.1f}" for t in s["split_hints"])
            first = f"{s['split_hints'][0]:.1f}"
            lines.append(f"   likely break points inside this clip (seconds): {hints}")
            lines.append(f'   by hand: python {me} "{clip}" -o "{clip.with_suffix("")}_split" --split-at {first}')
            lines.append("            (use any of the times above, or your own, comma-separated)")
        lines.append("")
    log("")
    for ln in lines:
        log("  " + ln)
    with open(out / f"{stem}_review_commands.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  (saved to {out / (stem + '_review_commands.txt')})")


def refine_weights(sensitivity=0.0):
    """Settings for re-splitting ONE clip that may hold several ads: trust a clear break more and
    standard ad lengths less (tuned on two-ad clips and single ads that must stay whole)."""
    w = dict(DEFAULT_W)
    w["bias"] -= sensitivity * 0.75
    w["seg_cost"] = max(0.0, w["seg_cost"] - sensitivity * 0.4)
    w["duration"] = 0.75
    return w


def _nums(text):
    """'2,5' / '17-19' / '2 5 9' -> set of ints"""
    out = set()
    for part in text.replace(" ", ",").split(","):
        part = part.strip()
        if "-" in part:
            x, _, y = part.partition("-")
            if x.strip().isdigit() and y.strip().isdigit():
                out.update(range(int(x), int(y) + 1))
        elif part.isdigit():
            out.add(int(part))
    return out


def _pick(rest, rev):
    rest = rest.strip()
    if rest in ("", "all", "a"):
        return list(rev)
    want = _nums(rest)
    return [s for s in rev if s["idx"] in want]


def join_files(paths, dst):
    """Join clips back into one (they were all exported with the same settings, so this is lossless).
    Falls back to re-encoding if the files don't match."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in paths:
            f.write("file '" + str(Path(p).resolve()).replace("\\", "/").replace("'", "'\\''") + "'\n")
        lst = f.name
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                            "-c", "copy", "-movflags", "+faststart", str(dst)], capture_output=True)
        if r.returncode != 0 or not Path(dst).exists() or Path(dst).stat().st_size == 0:
            ins = []
            for p in paths:
                ins += ["-i", str(p)]
            n = len(paths)
            fc = "".join(f"[{i}:v:0]setsar=1,format=yuv420p[v{i}];[{i}:a:0]aresample=48000[a{i}];" for i in range(n))
            fc += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + ins + ["-filter_complex", fc,
                            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)], check=True)
    finally:
        os.unlink(lst)


def _set_aside(p):
    """Move a clip that has been replaced into a _replaced folder next to it (never deleted)."""
    import shutil
    p = Path(p)
    keep = p.parent / "_replaced"
    keep.mkdir(exist_ok=True)
    shutil.move(str(p), str(keep / p.name))


def _refine_clip(s, args, harder):
    sub = argparse.Namespace(**vars(args))
    sub.refine, sub.split_at, sub.rescan_review, sub.no_prompt = True, "", False, True
    sub.quiet_review = True
    # a rescan never inherits a cautious (negative) sensitivity from the tape run
    sub.sensitivity = max(0.0, args.sensitivity) + (1.0 if harder else 0.0)
    sw = refine_weights(sub.sensitivity)
    clip = s["path"]
    dest = clip.parent / (clip.stem + "_split")
    log(f"\n  --- rescanning #{s['idx']:02d} ({clip.name}) -> {dest.name}")
    process(clip, dest, sub, sw)
    return dest


def _clear(s, out):
    """Move a reviewed clip out of _review into the main folder."""
    import shutil
    clip = s["path"]
    if clip.exists():
        shutil.move(str(clip), str(out / clip.name))
        s["path"] = out / clip.name
        s["review"] = False
        log(f"  #{s['idx']:02d} moved to the main folder: {clip.name}")


def offer_rescan(segs, args, w, out=None):
    """After a run, ask what to do: clear review clips, rescan them finer, join clips back together."""
    if args.refine:
        return
    pending = [s for s in segs if s["review"] and "blank" not in s["why"]]
    if not pending:
        log("")
        log("  No clips need review - everything was split with confidence.")
    if args.dry_run:
        log("")
        log("  (dry run: no clips were exported, so the review menu is skipped)")
        return
    rev = [s for s in pending if s.get("file")]
    if args.rescan_review:
        for s in rev:
            _refine_clip(s, args, False)
        return
    if args.no_prompt:
        log("")
        log("  (--no-prompt: menu skipped; see the commands above)")
        return
    clips = [s for s in segs if s.get("file")]          # every exported clip, main folder or _review
    if not clips:
        return
    while True:
        log("")
        if rev:
            log(f"  {len(rev)} clip(s) still in _review: " + ", ".join(f"#{s['idx']}" for s in rev))
        log("  What next? (clip numbers are the #NN in the file names and on the contact sheet)")
        if rev:
            log("    c          review clips all look fine - move them into the main folder")
            log("    c 2,5      only these look fine - move them into the main folder")
            log("    r          rescan the review clips with finer splitting (only those clip files)")
            log("    r 2,5      rescan only these      (add + for extra-fine, e.g. 'r+ 2,5')")
        log("    j 17-19    join clips back into one ad (e.g. 'j 17,18,19'), main folder or _review")
        log("    Enter      finish")
        sys.stdout.flush()
        try:
            ans = input("  Your choice: ").strip().lower()
        except (EOFError, OSError):
            log("")
            log("  (no keyboard input available here - menu skipped; see the commands above)")
            return
        if not ans:
            return
        cmd, rest = ans[0], ans[1:]
        harder = rest.startswith("+")
        rest = rest.lstrip("+")
        if cmd == "j":
            want = _nums(rest)
            chosen = sorted((s for s in clips if s["idx"] in want), key=lambda s: s["idx"])
            if len(chosen) < 2:
                log("  (pick at least two clip numbers to join, e.g. 'j 17-19')")
                continue
            idxs = [s["idx"] for s in chosen]
            if idxs != list(range(idxs[0], idxs[-1] + 1)):
                log("  note: those clips aren't next to each other on the tape - joining them in number order")
            total = sum(s["dur"] for s in chosen)
            stem = chosen[0]["path"].name.rsplit(f"_{idxs[0]:02d}_", 1)[0]
            dst = out / f"{stem}_{idxs[0]:02d}-{idxs[-1]:02d}_{round(total)}s.mp4"
            join_files([s["path"] for s in chosen], dst)
            for s in chosen:
                _set_aside(s["path"])
                clips.remove(s)
                if s in rev:
                    rev.remove(s)
            merged = dict(chosen[0], end=chosen[-1]["end"], dur=total, path=dst, file=dst.name, review=False)
            clips.append(merged)
            log(f"  joined #{idxs[0]}-#{idxs[-1]} into {dst.name} (main folder); "
                "the separate clips were moved to _replaced")
            continue
        if cmd not in ("c", "r") or not rev:
            log("  (type c, r, j, or press Enter)" if rev else "  (type j or press Enter)")
            continue
        chosen = _pick(rest, rev)
        if not chosen:
            log("  (no matching clip numbers in _review)")
            continue
        if cmd == "c":
            for s in chosen:
                _clear(s, out)
                rev.remove(s)
        else:
            for s in chosen:
                dest = _refine_clip(s, args, harder)
                parts = sorted(p for p in dest.glob("*.mp4"))
                review_parts = sorted((dest / "_review").glob("*.mp4")) if (dest / "_review").exists() else []
                if len(parts) + len(review_parts) <= 1:
                    log(f"  #{s['idx']:02d}: no further split found - it looks like a single ad.")
                    continue
                log(f"  #{s['idx']:02d} was split into {len(parts) + len(review_parts)} pieces "
                    f"(see {dest.name}/ and its contact sheet).")
                sys.stdout.flush()
                try:
                    yn = input("  Use these pieces instead of the original clip? [y/N] ").strip().lower()
                except (EOFError, OSError):
                    yn = ""
                if yn.startswith("y"):
                    import shutil
                    _set_aside(s["path"])
                    for p in parts:
                        shutil.move(str(p), str(out / p.name))
                    for p in review_parts:
                        shutil.move(str(p), str(s["path"].parent / p.name))
                    log(f"  #{s['idx']:02d} replaced: {len(parts)} piece(s) moved to the main folder"
                        + (f", {len(review_parts)} still to review" if review_parts else "")
                        + ". The original is kept in _review/_replaced.")
                    rev.remove(s)
                    clips.remove(s)


def process(path, out, args, w):
    log(f"\n== {path.name}   (split_commercials {VERSION})")
    fps, dur = probe(path)
    log(f"  {fmt(dur)} @ {fps:.2f} fps")
    if args.split_at:
        manual_split(path, out, args, dur)
        return
    if args.refine:
        log("  refine mode: treating this file as one clip to re-check for ads stuck together")
    single, stds, thumbs, extras, stripe_info = analyze_video(path, fps, dur)
    log("  analysing audio ...")
    rms, mfcc = analyze_audio(path, dur)
    spots = [] if args.no_separators else separator_spots(stripe_info, fps)
    cands = score_candidates(add_separators(find_candidates(fps, single, stds), spots), fps, thumbs, rms, mfcc, extras)
    if spots:
        # only trust the repeating transition if most of its appearances also look like real ad breaks
        # on their own (sound + picture change). Striped scenery or interlacing must not switch this on.
        plain = dict(w, wipe=0.0, separators=False)
        wipes = [c for c in cands if c.get("wipe")]
        good = [c for c in wipes if boundary_score(c, plain) > 0.0]
        if len(good) >= 4 and len(good) >= 0.75 * len(wipes):
            for c in wipes:
                if boundary_score(c, plain) <= -1.0:
                    c["wipe"] = False        # this one is probably scenery, not the transition
            log(f"  found a repeating transition between ads ({len(good)} times) - using it as the break marker")
        else:
            for c in wipes:
                c["wipe"] = False
            spots = []
    w = dict(w, separators=bool(spots))
    if args.no_blank:
        for c in cands:
            c["blank"] = 0.0
    min_len = max(args.min_len, 12.0) if args.refine else args.min_len   # no tiny pieces when re-splitting
    cuts = choose_cuts(cands, dur, w, min_len, args.max_len)
    log(f"  {len(cands)} shot changes -> {len(cuts) + 1} ads")

    # build segments; drop the blank gap itself from both neighbours
    bounds = [dict(end=0.0, score=99)] + cuts + [dict(start=dur, score=99)]
    junk = junk_frames(stds, thumbs, fps)
    db = 20 * np.log10(rms + 1e-5)
    segs = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k]["end"], bounds[k + 1]["start"]
        # wipes fade in/out for a moment beyond what is detected: trim a little extra next to them
        if bounds[k].get("wipe"):
            a += 0.5
        if bounds[k + 1].get("wipe"):
            b -= 0.5
        if b - a < 2.0:
            continue
        if junk[int(a * fps):int(b * fps)].mean() > 0.6 if int(b * fps) > int(a * fps) else False:
            segs.append(dict(idx=len(segs) + 1, start=a, end=b, dur=b - a, review=True, why="mostly blank", note=""))
            continue
        a, b = trim_junk(a, b, junk, fps, keep=args.keep_blank)
        why = []
        if k == 0 and not args.refine:
            why.append("first clip (may be partial)")
        if k == len(bounds) - 2 and not args.refine:
            why.append("last clip (may be partial)")
        if (b - a > 75 and length_fit(b - a) < 0.35) or b - a < 8:
            why.append("unusual length (check for two ads)" if b - a > 75 else "very short")
        if min(bounds[k]["score"], bounds[k + 1]["score"]) < args.confident:
            why.append("weak boundary")
        if np.median(stds[int(a * fps):int(b * fps)] if int(b * fps) > int(a * fps) else [0]) < BLANK_STD:
            why.append("mostly blank")
        # strict check: did the wipe actually eat into this ad? (sound still playing at the cut
        # and real picture under the wipe = the ad bleeds into the transition and is cut short)
        cut_start = bounds[k].get("wipe") and cut_into(a, "start", db, junk, fps, a, b, dur)
        cut_end = bounds[k + 1].get("wipe") and cut_into(b, "end", db, junk, fps, a, b, dur)
        note = []
        if cut_start:
            note.append("CUT SHORT at start: the ad is still playing under the transition wipe "
                        "(about 1-2 s missing)")
        if cut_end:
            note.append("CUT SHORT at end: the ad is still playing when the transition wipe starts "
                        "(about 1-2 s missing, bleeds into the next ad)")
        inner = sorted((c for c in cands if c["start"] - a >= 6 and b - c["end"] >= 6),
                       key=lambda c: -boundary_score(c, w))
        picks = []
        for c in inner:
            if boundary_score(c, w) < -2.0 or len(picks) == 3:
                break
            if all(abs(c["t"] - p) > 3 for p in picks):
                picks.append(c["t"])
        segs.append(dict(idx=len(segs) + 1, start=a, end=b, dur=b - a, review=bool(why), why="; ".join(why),
                         note="; ".join(note), wipe_start=bool(cut_start), wipe_end=bool(cut_end),
                         split_hints=[t - a for t in sorted(picks)]))

    out.mkdir(parents=True, exist_ok=True)
    stem = safe(path.stem)
    sheet = out / f"{stem}_segments.png"
    contact_sheet(path, segs, sheet)
    with open(out / f"{stem}_segments.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["#", "start", "end", "seconds", "status", "reason", "transition note", "file"])
        for s in segs:
            s["file"] = ""
            junk = "mostly blank" in s["why"]
            if junk:
                s["why"] = "blank/static - not exported"
            folder = out / "_review" if s["review"] else out
            s["path"] = folder / (f"{stem}_{s['idx']:02d}_{round(s['dur'])}s" + ".mp4")
            if not junk and not args.dry_run and not (s["review"] and args.skip_review):
                folder.mkdir(parents=True, exist_ok=True)
                dst = s["path"]
                export_clip(path, s["start"], s["end"], dst, args.crf)
                s["file"] = str(dst.relative_to(out))
                log(f"  #{s['idx']:02d} {fmt(s['start'])} {s['dur']:5.1f}s -> {s['file']}")
            wr.writerow([s["idx"], f"{s['start']:.2f}", f"{s['end']:.2f}", f"{s['dur']:.1f}",
                         "skipped" if junk else ("review" if s["review"] else "ok"), s["why"], s.get("note", ""),
                         s["file"]])
    ok = sum(not s["review"] for s in segs)
    log(f"  done: {ok} confident clips, {len(segs) - ok} in _review.  Contact sheet: {sheet}")
    hit = [s for s in segs if s.get("note")]
    if hit:
        log("")
        log("  WARNING: CLIPS CUT SHORT BY TRANSITION WIPES")
        log("  This tape's transition wipe overlaps the ads themselves. In these clips the ad is")
        log("  still playing when the wipe takes over, so 1-2 seconds of it are missing and the")
        log("  sound/picture bleeds into the next ad. This comes from how the compilation was made")
        log("  and can't be recovered from this tape:")
        for s in hit:
            where = " + ".join(x for x, f in (("start", s["wipe_start"]), ("end", s["wipe_end"])) if f)
            log(f"    #{s['idx']:02d}  {fmt(s['start'])}  cut short at {where}")
        log("  Details are in the 'transition note' column of the CSV and marked on the contact sheet.")
    review_help(segs, out, stem, args)
    offer_rescan(segs, args, w, out)


def main():
    ap = argparse.ArgumentParser(description="Split a commercial compilation into one clip per ad.")
    ap.add_argument("inputs", nargs="+", help="video file(s) or folder(s)")
    ap.add_argument("-o", "--out", required=True, help="output folder (e.g. a NeXroll preroll category folder)")
    ap.add_argument("--sensitivity", type=float, default=0.0,
                    help="-2..2. Higher = more cuts (split more), lower = fewer cuts. Default 0")
    ap.add_argument("--min-len", type=float, default=6.0, help="shortest ad allowed, seconds (default 6)")
    ap.add_argument("--max-len", type=float, default=150.0, help="longest ad allowed, seconds (default 150)")
    ap.add_argument("--confident", type=float, default=1.0, help="boundary score below this sends a clip to _review")
    ap.add_argument("--no-blank", action="store_true", help="ignore blank frames (tapes with gray filler inside ads)")
    ap.add_argument("--skip-review", action="store_true", help="don't export uncertain clips at all")
    ap.add_argument("--dry-run", action="store_true", help="only write the cut list and contact sheet")
    ap.add_argument("--no-separators", action="store_true",
                    help="don't look for a repeating transition (wipe) between ads")
    ap.add_argument("--keep-blank", type=float, default=1.0,
                    help="max seconds of black/static left at the start or end of a clip (default 1)")
    ap.add_argument("--refine", action="store_true",
                    help="re-split ONE clip from _review: more willing to cut, and its first/last part "
                         "are treated as whole ads (use on a clip file, not a whole tape)")
    ap.add_argument("--split-at", default="",
                    help="split a clip by hand at these times, e.g. \"23.4\" or \"0:23.4,0:51\" (no analysis)")
    ap.add_argument("--join", action="store_true",
                    help="join the given clip files back into one ad, in the order given "
                         "(originals are moved to a _replaced folder next to them)")
    ap.add_argument("--rescan-review", action="store_true",
                    help="after splitting, automatically re-split every _review clip with --refine")
    ap.add_argument("--no-prompt", action="store_true", help="never ask questions at the end of a run")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better/bigger, default 20)")
    args = ap.parse_args()

    w = refine_weights(args.sensitivity) if args.refine else dict(DEFAULT_W)
    if not args.refine:
        w["bias"] -= args.sensitivity * 0.75
        w["seg_cost"] = max(0.0, w["seg_cost"] - args.sensitivity * 0.4)

    files = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        elif p.exists():
            files.append(p)
        else:
            log(f"not found: {p}")
    if not files:
        sys.exit("no input videos")
    out = Path(args.out)
    if args.join:
        if len(files) < 2:
            sys.exit("--join needs at least two clip files")
        out.mkdir(parents=True, exist_ok=True)
        total = sum(probe(f)[1] for f in files)
        dst = out / f"{safe(files[0].stem)}_joined_{round(total)}s.mp4"
        join_files(files, dst)
        for f in files:
            _set_aside(f)
        log(f"joined {len(files)} clips -> {dst}  (originals moved to _replaced)")
        return
    for f in files:
        try:
            process(f, out, args, w)
        except subprocess.CalledProcessError as e:
            log(f"  ffmpeg failed on {f.name}: {e}")


if __name__ == "__main__":
    main()
