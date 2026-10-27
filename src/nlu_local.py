"""
Offline in-car language understanding: small, fast, no network.

  intent  : TF-IDF (character + word n-grams) + logistic regression, trained on template phrases
            in English and German (generated below - none of the test commands are used for training)
  slots   : deterministic rules (numbers, seats, windows, destinations, colours, ...)
  routing : returns a confidence; below the threshold the request is handed to the cloud LLM
  "and"   : "X and Y" / "X und Y" is split and each part classified separately

This is the part that would run on the car's own computer: milliseconds, works in a tunnel, and its
behaviour is fully predictable.
"""

import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# ------------------------------------------------------------------------------------------
# Training phrases (templates). {n}=number, {seat}, {win}, {dest}, {place}, {music}, {name}, {colour}
# ------------------------------------------------------------------------------------------
FILL = {
    "n": [str(x) for x in range(0, 31)],
    "seat_en": ["driver", "passenger", "rear", "my", "the front"], "seat_de": ["Fahrer", "Beifahrer", "hinten", "meine"],
    "win_en": ["driver", "passenger", "rear left", "rear right", "my", "all the", "the back"],
    "win_de": ["Fahrer", "Beifahrer", "hinten links", "hinten rechts", "alle", "das"],
    "dest": ["Berlin", "Stuttgart", "the airport", "Olympiapark", "Karlsplatz", "Hamburg", "grandma's house",
             "the train station", "Leopoldstraße 12", "Frankfurt", "IKEA", "the stadium", "Regensburg", "Nürnberg"],
    "music": ["the Beatles", "classical music", "hip hop", "some pop", "Coldplay", "my workout mix", "news",
              "a podcast", "Radio Energy", "chill music", "Adele", "techno", "the charts", "country"],
    "name": ["Anna", "Max", "Dad", "my wife", "Thomas Schmidt", "Julia", "the garage", "Peter", "Grandma", "Sarah"],
    "colour_en": ["red", "green", "blue", "orange", "white", "yellow", "pink", "turquoise"],
    "colour_de": ["rot", "grün", "blau", "orange", "weiß", "gelb", "pink", "türkis"],
}
T = {
    "set_temperature": [
        "set the temperature to {n} degrees", "temperature {n}", "make it {n} degrees", "{n} degrees please",
        "set the {seat_en} side to {n}", "warmer please", "make it warmer", "it should be warmer", "raise the temperature",
        "lower the temperature", "make it cooler", "colder please", "decrease the temperature", "heat up the car",
        "increase the heating", "turn up the heat", "turn down the heat", "temperature up", "temperature down",
        "stell die temperatur auf {n} grad", "temperatur {n} grad", "{n} grad bitte", "{seat_de}seite auf {n} grad",
        "wärmer bitte", "mach es wärmer", "kälter bitte", "mach es kühler", "temperatur höher", "temperatur runter",
        "heizung höher", "heizung niedriger", "temperatur erhöhen", "temperatur senken"],
    "set_fan": [
        "fan to {n}", "set the fan speed to {n}", "fan level {n}", "turn off the fan", "fan off", "fan on maximum",
        "more air please", "less air", "increase the fan", "decrease the fan speed", "blower to {n}", "turn the fan up",
        "gebläse auf {n}", "gebläse aus", "lüftung stufe {n}", "lüfter aus", "mehr luft bitte", "weniger luft",
        "lüftung höher", "gebläse runter", "lüftung auf maximum"],
    "set_seat_heating": [
        "turn on the seat heating", "seat heating on", "heat my seat", "{seat_en} seat heating level {n}",
        "seat heater off", "turn off seat heating", "set {seat_en} seat heating to {n}", "warm up my seat",
        "seat heating maximum", "heated seat on", "sitzheizung an", "sitzheizung aus", "sitzheizung {seat_de} stufe {n}",
        "sitzheizung auf {n}", "heiz meinen sitz", "sitzheizung maximal", "schalte die sitzheizung ein"],
    "set_window": [
        "open the {win_en} window", "close the {win_en} window", "open the windows", "close all windows",
        "window down", "window up", "roll down my window", "roll up the window", "open my window halfway",
        "open the {win_en} window {n} percent", "shut the windows", "fenster {win_de} öffnen", "fenster {win_de} schließen",
        "mach das fenster auf", "mach das fenster zu", "alle fenster zu", "fenster runter", "fenster hoch",
        "öffne das {win_de} fenster", "schließ die fenster"],
    "navigate_to": [
        "navigate to {dest}", "take me to {dest}", "drive to {dest}", "directions to {dest}", "route to {dest}",
        "go to {dest}", "how do I get to {dest}", "start navigation to {dest}", "take me home", "navigate home",
        "go to work", "drive me to work", "navigiere nach {dest}", "fahr nach {dest}", "route zu {dest}",
        "navigation zu {dest}", "bring mich nach {dest}", "fahr mich nach hause", "navigiere zur arbeit",
        "wie komme ich nach {dest}", "zum {dest} bitte"],
    "find_nearby": [
        "find a charging station", "nearest charger", "where can I charge", "find parking", "where can I park",
        "find a petrol station", "nearest gas station", "find a restaurant", "somewhere to eat", "find a coffee shop",
        "where can I get coffee", "nearest hospital", "find a supermarket", "show charging points nearby",
        "finde eine ladestation", "nächste ladesäule", "wo kann ich laden", "parkplatz finden", "wo kann ich parken",
        "nächste tankstelle", "restaurant in der nähe", "wo gibt es kaffee", "nächstes krankenhaus",
        "supermarkt in der nähe", "café in der nähe"],
    "play_media": [
        "play {music}", "put on {music}", "I want to listen to {music}", "play music", "turn on the radio",
        "play the radio", "play something by {music}", "start {music}", "listen to {music}", "spiel {music}",
        "spiele musik von {music}", "radio an", "mach musik an", "ich möchte {music} hören", "starte {music}"],
    "media_control": [
        "pause", "pause the music", "stop the music", "resume", "continue playing", "next track", "skip this song",
        "skip", "previous song", "go back one song", "play the last song again", "pause bitte", "musik stoppen",
        "musik anhalten", "weiter", "nächster titel", "überspringen", "vorheriger titel", "zurück zum letzten lied",
        "wiedergabe fortsetzen"],
    "set_volume": [
        "volume {n}", "set the volume to {n}", "louder", "turn it up", "turn the volume up", "quieter", "turn it down",
        "volume down", "mute", "lower the volume", "raise the volume", "lautstärke {n}", "lautstärke auf {n}", "lauter",
        "leiser", "mach lauter", "mach leiser", "ton aus", "lautstärke erhöhen", "lautstärke verringern"],
    "call_contact": [
        "call {name}", "phone {name}", "ring {name}", "dial {name}", "give {name} a call", "make a call to {name}",
        "ruf {name} an", "anruf an {name}", "{name} anrufen", "telefoniere mit {name}", "wähle {name}"],
    "get_vehicle_status": [
        "what is my range", "how much range do I have", "how far can I go", "battery level", "how much battery is left",
        "what's my state of charge", "tyre pressure", "check tyres", "when is the next service", "service due",
        "are the doors locked", "door status", "reichweite", "wie weit komme ich", "akkustand", "wie viel akku habe ich",
        "ladestand", "reifendruck", "wann ist die nächste inspektion", "sind die türen zu"],
    "set_drive_mode": [
        "sport mode", "switch to eco mode", "comfort mode please", "activate sport mode", "change the drive mode to eco",
        "driving mode comfort", "sportmodus", "eco modus an", "fahrmodus sport", "komfortmodus", "wechsle in den eco modus"],
    "set_ambient_light": [
        "ambient light {colour_en}", "set the interior lighting to {colour_en}", "make the lights {colour_en}",
        "change ambient lighting to {colour_en}", "mood lighting {colour_en}", "ambientebeleuchtung {colour_de}",
        "innenbeleuchtung {colour_de}", "licht auf {colour_de}", "ambiente licht {colour_de}"],
    "open_trunk": ["open the trunk", "open the boot", "trunk open", "pop the trunk", "open the tailgate",
                   "kofferraum auf", "öffne den kofferraum", "heckklappe öffnen", "kofferraum öffnen bitte"],
    "play_video": ["play a video", "watch a movie", "play a film", "show a video on the screen", "start youtube",
                   "watch netflix", "video abspielen", "film abspielen", "ich will einen film schauen", "youtube starten"],
    "manual_lookup": [
        "how do I change the wiper settings", "what does this warning light mean", "how do I connect a child seat",
        "how do I use cruise control", "what should I do with a flat tyre", "how do I set the charge limit",
        "where is the washer fluid", "how does lane assist work", "what's the recommended tyre pressure",
        "how often is service needed", "how do I defrost the windscreen", "how much can I tow",
        "wie funktioniert der tempomat", "was bedeutet die warnleuchte", "wie stelle ich das ladelimit ein",
        "wo ist das wischwasser", "wie befestige ich einen kindersitz", "was tun bei einer reifenpanne",
        "wie funktioniert der spurhalteassistent", "wie enteise ich die scheibe", "wie lange dauert das laden"],
    "other": [
        "tell me a joke", "what's the weather like", "who are you", "how are you", "thank you", "thanks", "good morning",
        "what time is it", "what is the capital of spain", "sing a song", "that's all", "never mind", "cancel",
        "what's the news", "how old are you", "hello", "erzähl mir was", "danke", "wie spät ist es", "hallo",
        "wer bist du", "guten morgen", "abbrechen", "vergiss es", "wie wird das wetter", "was ist der sinn des lebens"],
}


def generate(per_template=6, seed=0):
    rng = random.Random(seed)
    X, y = [], []
    for intent, temps in T.items():
        for t in temps:
            keys = re.findall(r"{(\w+)}", t)
            for _ in range(per_template if keys else 2):
                s = t
                for k in keys:
                    s = s.replace("{" + k + "}", rng.choice(FILL[k]), 1)
                if rng.random() < 0.3:
                    s = rng.choice(["please ", "can you ", "could you ", "bitte ", "kannst du ", "hey car, "]) + s
                X.append(s.lower())
                y.append(intent)
    return X, y


# ------------------------------------------------------------------------------------------
# Slot extraction rules
# ------------------------------------------------------------------------------------------
NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "null": 0, "eins": 1, "zwei": 2, "drei": 3,
             "vier": 4, "fünf": 5, "max": None}
UP = r"\b(warm|warmer|wärmer|up|higher|höher|increase|raise|erhöhen|more|mehr|louder|lauter|heat up|aufdrehen)\b"
DOWN = r"\b(cool|cooler|kühler|colder|kälter|down|lower|niedriger|runter|decrease|senken|less|weniger|quieter|leiser|verringern)\b"
OFF = r"\b(off|aus|ausschalten|mute)\b"
COLOURS = {"red": "red", "rot": "red", "green": "green", "grün": "green", "blue": "blue", "blau": "blue", "orange": "orange",
           "white": "white", "weiß": "white", "yellow": "yellow", "gelb": "yellow", "pink": "pink", "purple": "purple",
           "lila": "purple", "violett": "purple", "turquoise": "turquoise", "türkis": "turquoise"}
NEARBY_WORDS = [("charging_station", r"charg|lade|lades"), ("parking", r"park"), ("fuel_station", r"petrol|gas station|fuel|tankstelle"),
                ("coffee", r"coffee|kaffee|café|cafe"), ("restaurant", r"restaurant|eat|essen|food|hungry|starving"),
                ("hospital", r"hospital|krankenhaus|klinik"), ("supermarket", r"supermarket|supermarkt|grocer")]
STATUS_WORDS = [("tyre_pressure", r"tyre|tire|reifen"), ("battery", r"battery|akku|charge level|state of charge|ladestand"),
                ("range", r"range|far|reichweite|weit"), ("service", r"service|inspektion|wartung"),
                ("doors", r"door|tür|locked")]


def number(text):
    m = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if m:
        return float(m.group().replace(",", "."))
    for w, v in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", text) and v is not None:
            return float(v)
    return None


def seat(text, default="driver"):
    if re.search(r"passenger|beifahrer", text):
        return "passenger"
    if re.search(r"\brear\b|\bback\b|hinten|fond", text):
        return "rear"
    return default


def window(text):
    if re.search(r"\ball\b|\balle\b|windows|fenster\b(?!.*(fahrer|beifahrer))", text) and not re.search(r"\bmy\b|driver|passenger|fahrer", text):
        if re.search(r"windows|\ball\b|\balle\b", text):
            return "all"
    if re.search(r"rear left|hinten links", text):
        return "rear_left"
    if re.search(r"rear right|hinten rechts", text):
        return "rear_right"
    if re.search(r"passenger|beifahrer", text):
        return "passenger"
    if re.search(r"\bmy\b|driver|fahrer", text):
        return "driver"
    return "all"


def destination(text):
    if re.search(r"\bhome\b|nach hause|zuhause|heim\b", text):
        return "home"
    if re.search(r"\bwork\b|\boffice\b|\barbeit\b|büro", text):
        return "work"
    m = re.search(r"(?:navigate to|navigiere (?:nach|zu|zum|zur)|take me to|get me to|drive (?:me )?to|directions to|"
                  r"route (?:to|nach|zu|zum|zur)|go to|get to|fahr(?: mich)? (?:nach|zu|zum|zur)|bring mich (?:nach|zu|zum|zur)|"
                  r"komme ich (?:nach|zu|zum|zur)|navigation (?:nach|zu|zum|zur)|\bzum|\bzur|\bnach|\bto)\s+(.+)", text)
    if not m:
        return None
    d = re.sub(r"\b(please|bitte|now|jetzt)\b", "", m.group(1)).strip(" ?.!")
    return d or None


def media_query(text):
    m = re.search(r"(?:play|put on|listen to|start|spiele?|starte|hören|radio)\s+(.+)", text)
    q = m.group(1) if m else ""
    q = re.sub(r"\b(some|something by|music by|musik von|my|the|please|bitte|playlist|ab)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" ?.!")
    if text.startswith("radio ") and q:
        return q
    return q or None


def contact(text):
    m = re.search(r"(?:call|phone|ring|dial|ruf|anruf an|telefoniere mit|wähle)\s+(.+?)(?:\s+an)?$", text)
    if m:
        return re.sub(r"\b(please|bitte)\b", "", m.group(1)).strip(" ?.!")
    m = re.search(r"(.+?)\s+anrufen", text)
    return m.group(1).strip() if m else None


def slots(intent, text):
    t = text.lower()
    n = number(t)
    if intent == "set_temperature":
        zone = "driver" if re.search(r"driver|fahrer(?!.*beifahrer)", t) and not re.search(r"beifahrer", t) else \
               "passenger" if re.search(r"passenger|beifahrer", t) else "all"
        a = {"zone": zone}
        if n is not None:
            a["celsius"] = n
        elif re.search(UP + r"|heat up", t):
            a["change"] = 2
        elif re.search(DOWN, t):
            a["change"] = -2
        return a
    if intent == "set_fan":
        if re.search(OFF, t):
            return {"level": 0}
        if re.search(r"max", t):
            return {"level": 5}
        if n is not None:
            return {"level": int(n)}
        return {"change": -1} if re.search(DOWN, t) else {"change": 1}
    if intent == "set_seat_heating":
        s = seat(t)
        if re.search(OFF, t):
            return {"seat": s, "level": 0}
        if re.search(r"max", t):
            return {"seat": s, "level": 3}
        return {"seat": s, "level": int(n) if n is not None else 2}
    if intent == "set_window":
        action = "close" if re.search(r"close|shut|\bup\b|schlie|\bzu\b|hoch", t) else "open"
        a = {"action": action, "window": window(t)}
        if action == "open" and re.search(r"percent|prozent|%", t) and n is not None:
            a["percent"] = int(n)
        elif action == "open" and re.search(r"halfway|half|halb", t):
            a["percent"] = 50
        elif action == "open" and re.search(r"a little|a bit|ein bisschen|crack|spalt", t):
            a["percent"] = 20
        return a
    if intent == "navigate_to":
        d = destination(t)
        return {"destination": d.title()} if d else None
    if intent == "find_nearby":
        for cat, pat in NEARBY_WORDS:
            if re.search(pat, t):
                return {"category": cat}
        return None
    if intent == "play_media":
        q = media_query(t)
        a = {"source": "radio"} if re.search(r"radio", t) else {}
        if q:
            a["query"] = q.title()
        return a
    if intent == "media_control":
        if re.search(r"paus|stop|anhalten", t):   # "paus" also matches German "pausieren"
            return {"action": "pause"}
        if re.search(r"previous|back|last|vorherig|zurück|letzt", t):
            return {"action": "previous"}
        if re.search(r"resume|continue|fortsetzen", t):
            return {"action": "resume"}
        return {"action": "next"}
    if intent == "set_volume":
        if re.search(r"mute|ton aus", t):
            return {"level": 0}
        if n is not None:
            return {"level": int(n)}
        return {"change": -2} if re.search(DOWN, t) else {"change": 2}
    if intent == "call_contact":
        c = contact(t)
        return {"name": c.title()} if c else None
    if intent == "get_vehicle_status":
        for item, pat in STATUS_WORDS:
            if re.search(pat, t):
                return {"item": item}
        return None
    if intent == "set_drive_mode":
        for m, pat in (("sport", "sport"), ("eco", "eco"), ("comfort", "comfort|komfort")):
            if re.search(pat, t):
                return {"mode": m}
        return None
    if intent == "set_ambient_light":
        for w, c in COLOURS.items():
            if re.search(rf"\b{w}\b", t):
                return {"colour": c}
        return None
    if intent == "play_video":
        return {"query": re.sub(r"^(play|watch|start|put on|spiel|ich will)\s+", "", t).strip(" ?.!")}
    if intent == "manual_lookup":
        return {"question": text}
    return {}


class LocalNLU:
    def __init__(self, threshold=0.55, seed=0):
        self.threshold = threshold
        X, y = generate(seed=seed)
        self.n_train = len(X)
        feats = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ])
        self.model = Pipeline([("f", feats), ("clf", LogisticRegression(C=8, max_iter=3000))])
        self.model.fit(X, y)
        self.train_phrases = set(X)

    def classify(self, text):
        p = self.model.predict_proba([text.lower()])[0]
        i = int(np.argmax(p))
        return self.model.classes_[i], float(p[i])

    def parse(self, text):
        """Returns (calls, confidence). calls=None means 'hand over to the cloud'."""
        parts = [p for p in re.split(r"\s+(?:and then|and|und dann|und|dann)\s+", text.strip(), flags=re.I) if p]
        calls, confs = [], []
        for part in parts:
            intent, conf = self.classify(part)
            confs.append(conf)
            if intent == "other" or conf < self.threshold:
                return None, min(confs)
            args = slots(intent, part)
            if args is None:  # a required detail could not be extracted reliably
                return None, min(confs)
            calls.append({"name": intent, "args": args})
        return calls, min(confs)
