"""Our text preprocessor for Underdog Woof. It doesn't modify any Underdog code; it only imports their normalizer
and reads their splitter's length limit.

Why it exists: Underdog's splitter (underdog_woof/textsplit.py) cuts every sentence longer than CLAUSE_MAX characters
into pieces, preferring commas, semicolons, colons and dashes. When one stretch between those marks is itself over the
limit, it falls back to cutting before a word like "who" or "and", which leaves a piece ending mid-phrase
("...a final resting place for those"). The model was trained on utterances that end at punctuation, and on a piece
like that it often doesn't stop cleanly and repeats its last words. So we make sure that fallback never happens: every
stretch between punctuation marks is kept within the limit by adding commas at natural places, so every piece their
splitter makes ends on punctuation.

Steps:
  1. Lines: each non-blank line is its own unit, sent to the engine in its own call, so a line break is always a hard
     break (their sentence splitter alone would merge a line ending in '.)' or '."' with the next line). A line that
     doesn't end in punctuation gets a period, so titles and headings are spoken as their own sentence.
  2. US states: expand state abbreviations after a place name ("Gettysburg, Pa." -> "Gettysburg, Pennsylvania."),
     which Underdog's normalizer doesn't do and the model can't guess (it often drops "Pa." entirely).
     Then normalize with Underdog's own normalizer, so lengths are measured on the text their splitter will actually see
     (normalizing makes text longer: "Nov. 19th 1863" -> "November nineteenth eighteen sixty three").
  3. Commas: any stretch longer than the limit gets a comma before the joining word nearest its middle
     ("...for those, who here gave..."), or at the middle space if there is none, repeated until every stretch fits.
     A comma never goes inside a spoken number ("two hundred and thirty"), and "for which" / "to whom" stay together
     ("that cause, for which they gave").
"""
import re

from underdog_woof.normalize import Normalizer
from underdog_woof.textsplit import CLAUSE_MAX

# Longest stretch between punctuation marks we allow; a little under Underdog's CLAUSE_MAX for safety.
LIMIT = CLAUSE_MAX - 10
MIN_PART = 15          # never leave a fragment shorter than this on either side of an added comma

# Same boundaries Underdog's splitter uses: sentence ends, then commas/semicolons/colons and dashes.
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
CLAUSE_END = re.compile(r"(?<=[,;:])\s+|(?<=—)\s*|(?<=--)\s*")

# Characters that count as ending a line, and characters that may trail them (closing quotes and brackets).
LINE_END_PUNCT = ".!?…,;:—-"
TRAILING = "\"'”’)]}"

# Words a clause can start with; a comma before them reads naturally. Ambiguous words ("that", "so", "since") are left
# out because they are often not clause starts ("that nation", "so dedicated", "since the spring").
CLAUSE_WORDS = {"and", "but", "or", "nor", "yet", "which", "who", "whom", "whose", "when", "where", "while", "because",
                "although", "though", "whereas", "unless", "until", "if", "whether"}
# Weaker places to pause: prepositions that start a phrase. Used only when no clause word fits.
PHRASE_WORDS = {"as", "for", "with", "from", "to", "in", "on", "at", "by", "into", "through", "after", "before",
                "without", "during", "about", "against", "between", "than", "like"}

# Words of a spoken number, so a comma never splits one ("two hundred and thirty four", "dollars and fifty cents").
NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
    seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
    trillion point oh""".split())
MONEY_WORDS = {"dollar", "dollars", "cent", "cents", "euro", "euros", "pound", "pounds", "pence", "penny"}
RELATIVES = {"which", "whom", "whose"}    # after a preposition, the comma goes before the preposition: ", for which"
# A comma never goes right after these: "for quite a, while" or "the, which" would leave the word dangling.
NO_COMMA_AFTER = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our",
                  "their", "some", "any", "no", "every", "each", "of"} | PHRASE_WORDS

# US states. Underdog's normalizer doesn't expand them, and the model, which reads letters, can't know "Pa." means
# Pennsylvania; it often says nothing at all for it. They are expanded only right after "Place, ".
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "D C",
}
# Traditional (AP style) abbreviations, written with periods.
STATES_OLD = {
    "Ala.": "Alabama", "Ariz.": "Arizona", "Ark.": "Arkansas", "Calif.": "California", "Cal.": "California",
    "Colo.": "Colorado", "Conn.": "Connecticut", "Del.": "Delaware", "Fla.": "Florida", "Ga.": "Georgia",
    "Ill.": "Illinois", "Ind.": "Indiana", "Kan.": "Kansas", "Kans.": "Kansas", "Ky.": "Kentucky", "La.": "Louisiana",
    "Md.": "Maryland", "Mass.": "Massachusetts", "Mich.": "Michigan", "Minn.": "Minnesota", "Miss.": "Mississippi",
    "Mo.": "Missouri", "Mont.": "Montana", "Neb.": "Nebraska", "Nebr.": "Nebraska", "Nev.": "Nevada",
    "N.H.": "New Hampshire", "N.J.": "New Jersey", "N.M.": "New Mexico", "N.Y.": "New York", "N.C.": "North Carolina",
    "N.D.": "North Dakota", "N. Dak.": "North Dakota", "Okla.": "Oklahoma", "Ore.": "Oregon", "Oreg.": "Oregon",
    "Pa.": "Pennsylvania", "Penn.": "Pennsylvania", "Penna.": "Pennsylvania", "R.I.": "Rhode Island",
    "S.C.": "South Carolina", "S.D.": "South Dakota", "S. Dak.": "South Dakota", "Tenn.": "Tennessee",
    "Tex.": "Texas", "Vt.": "Vermont", "Va.": "Virginia", "Wash.": "Washington", "W.Va.": "West Virginia",
    "W. Va.": "West Virginia", "Wis.": "Wisconsin", "Wisc.": "Wisconsin", "Wyo.": "Wyoming", "D.C.": "D C",
}
# Postal codes that are also everyday words or other abbreviations ("Fine, OK." "Hollywood, LA"); these need stronger
# evidence: a ZIP code after them, or a word like "in" / "from" before the place ("born in Dayton, OH").
STATES_AMBIGUOUS = {"HI", "ID", "IN", "ME", "OH", "OK", "OR", "LA", "CO", "MA", "DE"}
PLACE_CUES = {"in", "at", "from", "to", "near", "of", "outside", "into", "through", "via", "toward", "towards",
              "around", "born", "based", "visit", "visited", "visiting", "moved", "live", "lives", "lived", "located"}
_PLACE = r"(\b[A-Z][\w'’-]*,\s+)"     # a capitalized word and its comma: "Gettysburg, "
_STATES_OLD_RE = re.compile(_PLACE + "(" + "|".join(re.escape(k) for k in sorted(STATES_OLD, key=len, reverse=True))
                            + r")(?=\s|$|[,;:!?)\"'’”])")
_STATES_RE = re.compile(_PLACE + r"([A-Z]{2})(?![\w.'’-])")


def expand_states(line):
    """'Gettysburg, Pa.' -> 'Gettysburg, Pennsylvania.'; 'Dayton, OH 45402' -> 'Dayton, Ohio 45402'."""
    def old(m):
        rest = m.string[m.end():]
        # the abbreviation's period also ended the sentence if the line ends or a new sentence starts
        stop = "." if re.match(r"\s*$|\s+[A-Z\"“(]", rest) else ""
        return m.group(1) + STATES_OLD[m.group(2)] + stop

    def postal(m):
        code = m.group(2)
        if code not in STATES:
            return m.group(0)
        if code in STATES_AMBIGUOUS:
            before = m.string[:m.start()].split()
            cue = before and re.sub(r"\W", "", before[-1]).lower() in PLACE_CUES
            zip_after = re.match(r"\s+\d{5}\b", m.string[m.end():])
            if not (cue or zip_after):
                return m.group(0)
        return m.group(1) + STATES[code]

    return _STATES_RE.sub(postal, _STATES_OLD_RE.sub(old, line))


_normalizer = Normalizer(use_nemo=False)


def end_with_punctuation(line):
    """Add a period unless the line already ends in punctuation (closing quotes and brackets may follow it)."""
    core = line.rstrip(TRAILING)
    if core and core[-1] not in LINE_END_PUNCT:
        line = core + "." + line[len(core):]
    return line


def end_lines(text):
    """One unit per non-blank line, each ending in punctuation."""
    return [end_with_punctuation(" ".join(line.split())) for line in text.splitlines() if line.strip()]


def _bare(word):
    return re.sub(r"\W", "", word).lower()


def _cut_point(s):
    """Index of the space where a comma should go in stretch s (the comma goes right before it), or None."""
    middle = len(s) / 2
    # words with the index of the space before each (None for the first word)
    words = [(m.start() - 1 if m.start() else None, _bare(m.group(0))) for m in re.finditer(r"\S+", s)]
    numeric = NUMBER_WORDS | MONEY_WORDS
    w = lambda k: words[k][1] if 0 <= k < len(words) else ""

    def in_number(k):
        """True if the space before word k is inside a spoken number ("two hundred | and | thirty")."""
        if w(k) == "and":
            return w(k - 1) in numeric and w(k + 1) in NUMBER_WORDS
        if w(k) in MONEY_WORDS:                         # "thirty four | dollars"
            return w(k - 1) in NUMBER_WORDS
        return w(k) in NUMBER_WORDS and (w(k - 1) in numeric or w(k - 1) == "and" and w(k - 2) in numeric)
    best = None
    for k in range(1, len(words)):
        i, word = words[k]
        prev = words[k - 1][1]
        if prev in NO_COMMA_AFTER:
            continue
        if word == "and" and (prev in NUMBER_WORDS or prev in MONEY_WORDS and k + 1 < len(words)
                              and words[k + 1][1] in NUMBER_WORDS):
            continue                                    # "two hundred and thirty", "dollars and fifty cents"
        if word in RELATIVES and prev in PHRASE_WORDS | {"of"}:
            continue                                    # handled at the preposition: ", for which"
        if word in CLAUSE_WORDS:
            cost = abs(i - middle)
        elif word in PHRASE_WORDS or word == "of" and k + 1 < len(words) and words[k + 1][1] in RELATIVES:
            relative_next = k + 1 < len(words) and words[k + 1][1] in RELATIVES
            cost = abs(i - middle) + (0 if relative_next else len(s) * 0.25)   # plain prepositions are a last resort
        else:
            continue
        if i < MIN_PART or len(s) - i < MIN_PART:
            continue
        if best is None or cost < best[0]:
            best = (cost, i)
    if best is not None:
        return best[1]
    # no joining word: the space nearest the middle, avoiding the inside of a spoken number
    spaces = [i for k, (i, _) in enumerate(words)
              if i is not None and not in_number(k) and w(k - 1) not in NO_COMMA_AFTER]
    inner = [i for i in spaces if MIN_PART <= i <= len(s) - MIN_PART]
    spaces = inner or spaces
    return min(spaces, key=lambda i: abs(i - middle)) if spaces else None


def _fit(stretch):
    """Add commas to one stretch (text with no internal clause punctuation) until every part is within LIMIT."""
    if len(stretch) <= LIMIT:
        return stretch
    i = _cut_point(stretch)
    if i is None:                       # one enormous word; nothing sensible to do
        return stretch
    left, right = stretch[:i].rstrip(), stretch[i:].lstrip()
    return _fit(left) + ", " + _fit(right)


def add_commas(sentence):
    """Split a sentence at the same clause boundaries Underdog's splitter uses and fit each stretch."""
    parts, pos = [], 0
    for m in CLAUSE_END.finditer(sentence):
        parts.append((sentence[pos:m.start()], sentence[m.start():m.end()]))
        pos = m.end()
    parts.append((sentence[pos:], ""))
    return "".join(_fit(chunk) + sep for chunk, sep in parts)


def preprocess(text):
    """Raw text in; out, one normalized unit per line, each ready for its own Woof(normalize=False) call."""
    units = []
    for line in end_lines(text):
        # check the ending again: normalizing can swallow the final period ("etc." -> "et cetera")
        spoken = end_with_punctuation(_normalizer(expand_states(line)))
        unit =" ".join(add_commas(s) for s in SENTENCE_END.split(spoken) if s)
        if unit:
            units.append(unit)
    return units
