# Comic prompts

`docs/figures/comic.jpg` was drawn panel by panel with an image-generation model (OpenAI, through Codex CLI). Panel 1 was generated first; panels 2-4 were generated with panel 1 attached as the style reference. The four 1536x1024 panels were then resized and tiled 2x2.

## Shared style and characters

```text
STYLE (must hold for every panel): simple doodle comic, like a quick marker sketch in a notebook. White background. Thick wobbly black marker outlines, stick-figure-simple bodies, dot eyes, no shading, no gradients. Flat accent colours only: one bright blue (#2a78d6) for the robot's chest light and blue labels, one orange (#eb6834) for highlights, light grey for the floor shadow. Lots of white space.
CHARACTERS (keep identical in every panel):
- ROBOT "DeePM": a small boxy robot with a rectangular head, two dot eyes, a single antenna with a ball on top, a blue square chest light reading "DeePM".
- DEV: a simple doodle person with a hoodie, messy hair and round glasses (generic, not a real person).
- BULL "SPY": a chubby cartoon bull with small horns and a round belly with "SPY" written on it.
- OLD MAN "Trend Rule": a tiny old man with a long white beard and a walking stick, a sign around his neck reading "12-mo trend".
TEXT: hand-lettered English in speech bubbles and signs, ALL CAPS comic lettering, spelled exactly as given; no other words anywhere. Wide 3:2 landscape, 1536x1024.
```

## Panel 1

```text
SCENE (panel 1 of 4): a fancy stage with a spotlight and a big banner "50 FUTURES". The ROBOT stands proudly on a podium holding up a trophy that reads "SHARPE 0.93". Below the stage, the DEV looks up, scratching their head, holding a phone showing a shopping-cart icon labelled "ETF".
BUBBLES: DEV: "COOL... BUT I TRADE ETFS."
Only the DEV and the ROBOT appear in this panel.
```

## Panel 2

```text
SCENE (panel 2 of 4): a garage workshop. The DEV, holding a wrench, is bolting three labelled gadgets onto the ROBOT: a backpack tag "REAL SPREADS", an alarm clock on its head labelled "NEXT-OPEN FILLS", and a small piggy-bank-shaped filter labelled "MINUS CASH". The ROBOT gives a thumbs-up.
BUBBLES: ROBOT: "ETF MODE: ON!"
Only the DEV and the ROBOT appear in this panel.
```

## Panel 3

```text
SCENE (panel 3 of 4): a running track. The ROBOT is sprinting far ahead with speed lines; behind it the OLD MAN "Trend Rule" hobbles with his walking stick, sweating. A scoreboard above the track reads "DEEPM 0.69" in blue and "TREND 0.24" in grey.
BUBBLES: OLD MAN: "SLOW DOWN, KID!"
Only the ROBOT and the OLD MAN appear in this panel.
```

## Panel 4

```text
SCENE (panel 4 of 4): a stormy rope bridge. The BULL "SPY" wobbles and is about to slip; the ROBOT, walking beside it arm in arm, steadies the bull with one arm. Both are smiling and carrying one shared bag of coins between them. A big sign on the far cliff reads "TOGETHER: SHARPE 0.93".
BUBBLES: BULL: "THANKS, BUDDY!"   ROBOT: "WE ZIG WHEN YOU ZAG."
Only the ROBOT and the BULL appear in this panel.
```
