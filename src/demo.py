"""
Render a scripted drive as an animated infotainment screen (assets/demo.gif): the conversation on the left,
the live car state on the right, and for every request where it was handled and how fast.

Uses whatever backends are available: offline only by default; with ANTHROPIC_API_KEY set, the hybrid
assistant (so requests the offline model passes on are answered by Claude).

    python src/demo.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from assistant import Assistant
from llm import available
from vehicle import Vehicle, VehicleState

ROOT = Path(__file__).resolve().parent.parent
W, H = 960, 540
BG, PANEL, TEXT, MUTED = (18, 22, 30), (30, 36, 48), (235, 238, 245), (140, 150, 168)
BLUE, GREEN, RED, ORANGE = (60, 140, 255), (70, 190, 110), (235, 80, 90), (245, 160, 50)
SCRIPT = [  # (speed km/h, what the driver says)
    (50, "Set the temperature to 20 degrees"),
    (50, "Navigate to Marienplatz and play some jazz"),
    (50, "Sitzheizung Fahrer Stufe 2"),
    (50, "How far can I drive?"),
    (100, "Play a movie on the screen"),
    (100, "Set the temperature to 35 degrees"),
    (100, "I'm freezing"),
    (100, "Call Mom"),
]


def font(size, bold=False):
    for name in (["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F, FB, FS, FT = font(17), font(17, True), font(14), font(22, True)


def wrap(draw, text, width, f):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=f) <= width:
            line = test
        else:
            lines.append(line)
            line = w
    return lines + [line] if line else lines


def frame(history, state, typing=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 16), "In-car voice assistant", font=FT, fill=TEXT)
    d.text((24, 46), "offline model in the car  ·  Claude in the cloud  ·  safety layer in between", font=FS, fill=MUTED)
    # conversation (newest at the bottom)
    y = H - 24
    items = history + ([("user", typing, None)] if typing else [])
    for role, text, meta in reversed(items):
        lines = wrap(d, text, 470, F)
        h = 22 * len(lines) + 16 + (20 if meta else 0)
        y -= h + 10
        if y < 80:
            break
        x = 24 if role == "user" else 120
        colour = (44, 54, 72) if role == "user" else (28, 66, 120)
        d.rounded_rectangle((x, y, x + 500, y + h), 12, fill=colour)
        for k, line in enumerate(lines):
            d.text((x + 14, y + 8 + 22 * k), line, font=F, fill=TEXT)
        if meta:
            route, ms, blocked = meta
            col = RED if blocked else (GREEN if route == "local" else ORANGE if route == "cloud" else MUTED)
            label = {"local": f"in the car · {ms:.0f} ms", "cloud": f"Claude · {ms / 1000:.1f} s",
                     "fallback": "offline: passed on to cloud, not connected"}[route] + ("  ·  BLOCKED by safety layer" if blocked else "")
            d.text((x + 14, y + h - 24), label, font=FS, fill=col)
    # car state panel
    px = 660
    d.rounded_rectangle((px, 80, W - 20, H - 20), 14, fill=PANEL)
    s = state
    rows = [("Speed", f"{s.speed_kmh:.0f} km/h"), ("Cabin", f"{s.temperature['driver']:.0f} °C"), ("Fan", f"{s.fan}"),
            ("Seat heating", f"driver {s.seat_heating['driver']}"), ("Navigation", s.destination or "–"),
            ("Media", s.media or "–"), ("Volume", f"{s.volume}"), ("Drive mode", s.drive_mode),
            ("Video", s.video or "–"), ("Phone", f"confirm: {s.pending_call}" if s.pending_call else "–")]
    d.text((px + 18, 94), "Vehicle", font=FB, fill=TEXT)
    for k, (a, b) in enumerate(rows):
        yy = 128 + 38 * k
        d.text((px + 18, yy), a, font=FS, fill=MUTED)
        d.text((px + 130, yy - 2), b[:16], font=F, fill=TEXT)
    return img


def main():
    assistant = Assistant("hybrid" if available() else "local")
    v = Vehicle(VehicleState(speed_kmh=SCRIPT[0][0]))
    history, frames, durations = [], [], []
    for speed, text in SCRIPT:
        v.state.speed_kmh = speed
        for k in range(4, len(text) + 4, 4):   # "typing" animation of the driver's words
            frames.append(frame(history, v.state, text[:k]))
            durations.append(60)
        r = assistant.handle(text, v)
        history += [("user", text, None), ("car", r["reply"], (r["route"], r["latency_ms"], bool(r["blocked"])))]
        frames.append(frame(history, v.state))
        durations.append(2600)
    out = ROOT / "assets" / "demo.gif"
    out.parent.mkdir(exist_ok=True)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    frames[-1].save(ROOT / "assets" / "demo_last_frame.png")
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB), backend: {assistant.backend}")


if __name__ == "__main__":
    main()
