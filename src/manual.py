"""
Owner's handbook Q&A.

The handbook below is written for this project (a fictional electric car), because real manuals are
copyrighted. Retrieval is TF-IDF over sections; each section carries German keywords so German questions
find the right section too. Offline, the assistant reads out the best-matching section; with Claude
connected, the retrieved section is passed to the model to phrase a short spoken answer.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HANDBOOK = {
    "tyre_pressure": ("Tyre pressure", "Recommended tyre pressure is 2.5 bar front and rear when cold. A yellow tyre "
                      "symbol with an exclamation mark means one tyre is at least 0.3 bar below target. Check and "
                      "inflate at the next opportunity.", "reifendruck reifen luftdruck druck reifenwarnung"),
    "charging": ("Charging", "Open the charge flap by pressing its rear edge. With a public AC charger the car charges "
                 "at up to 11 kW; on a DC fast charger at up to 150 kW, from 10 to 80 percent in about 30 minutes.",
                 "laden ladeklappe ladestation schnellladen ladezeit aufladen"),
    "charge_limit": ("Charge limit", "Set the charge limit in Settings, Battery. For daily driving 80 percent is "
                     "recommended to protect the battery; charge to 100 percent only before long trips.",
                     "ladegrenze ladelimit akku batterie schonen 80 prozent"),
    "warning_red": ("Red warning lights", "A red warning light means stop safely as soon as possible. A red battery "
                    "or brake symbol must be checked by a service partner before continuing.",
                    "rote warnleuchte warnlampe rot bremse anhalten"),
    "warning_yellow": ("Yellow warning lights", "A yellow warning light means a system needs attention soon. You can "
                       "continue driving carefully and book a service appointment.",
                       "gelbe warnleuchte kontrollleuchte gelb hinweis"),
    "wipers": ("Wipers", "Push the right stalk down once for a single wipe, up for continuous wiping. Rotate the ring "
               "on the stalk to set the rain sensor sensitivity.", "scheibenwischer wischer regensensor hebel"),
    "washer_fluid": ("Washer fluid", "The washer fluid reservoir is under the front bonnet, marked with a blue cap and "
                     "a windscreen symbol. Use a winter mix below freezing.", "wischwasser scheibenwaschanlage frostschutz"),
    "lights": ("Headlights", "The automatic headlights switch on in low light. Turn the light switch left of the "
               "steering wheel to AUTO. Pull the left stalk towards you for high beam.", "licht scheinwerfer fernlicht abblendlicht"),
    "lane_assist": ("Lane keeping assist", "Lane keeping assist gently steers back when the car drifts out of its lane "
                    "above 60 km/h. Switch it on or off with the button on the left steering wheel spoke.",
                    "spurhalteassistent spurassistent spur halten lenkung"),
    "cruise": ("Adaptive cruise control", "Press the cruise button on the right spoke to keep the set speed and the "
               "distance to the car ahead. Change the gap with the distance button, in four steps.",
               "tempomat abstandstempomat abstand geschwindigkeit halten"),
    "child_seat": ("Child seats", "ISOFIX anchors are on both outer rear seats. Deactivate the passenger airbag with "
                   "the key switch in the glovebox before fitting a rear-facing child seat on the front seat.",
                   "kindersitz isofix beifahrerairbag kind"),
    "flat_tyre": ("Flat tyre", "The car has no spare wheel. Use the tyre repair kit under the boot floor: connect the "
                  "compressor, inflate to 2.5 bar and drive at no more than 80 km/h to a workshop.",
                  "reifenpanne platten reparaturset kompressor ersatzrad"),
    "service": ("Service intervals", "A service is due every 30,000 km or every two years. The car reminds you 1,000 km "
                "before it is due. Brake fluid is changed every two years.", "inspektion wartung service intervall"),
    "key": ("Key battery", "If the key fob battery is low, hold the key against the marked area on the steering column "
            "to start the car. Replace the CR2032 coin cell in the key.", "schlüssel batterie funkschlüssel knopfzelle"),
    "defrost": ("Defrosting the windscreen", "Press the windscreen defrost button: the fan runs at maximum, air is "
                "directed at the windscreen and the air conditioning dries the air.", "beschlagen enteisen frontscheibe defrost"),
    "range": ("Range tips", "Range drops in cold weather. Preconditioning the cabin while still plugged in, eco mode and "
              "seat heating instead of cabin heating all help to extend range.", "reichweite sparen winter effizienz"),
    "towing": ("Towing", "The car may tow trailers up to 1,600 kg braked. Towing increases energy use by up to 50 percent; "
               "plan more charging stops.", "anhänger anhängelast ziehen schleppen"),
    "jump_start": ("12 V battery", "If the car does not wake up, the 12-volt battery may be flat. Connect a jump starter "
                   "to the terminals under the front bonnet; do not jump-start other cars from it.",
                   "starthilfe 12 volt batterie leer fremdstarten"),
}


class Handbook:
    def __init__(self):
        self.keys = list(HANDBOOK)
        docs = [f"{t} {txt} {kw}" for t, txt, kw in HANDBOOK.values()]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)

    def search(self, question: str, k: int = 1):
        sims = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.keys[i], float(sims[i])) for i in order]

    def answer(self, question: str) -> str:
        key, _ = self.search(question)[0]
        title, text, _ = HANDBOOK[key]
        return f"{title}: {text}"
