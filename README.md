# split_commercials.py

Cuts a long compilation of TV commercials (VHS rips, archive.org collections, etc.) into one clip per ad. The clips are ready to use as prerolls in Jellyfin, NeXroll, Plex, or Tunarr filler.

## AI disclosure
This script and README were written with the help of an AI assistant (Claude by Anthropic), then tested and adjusted on real commercial compilations. Review the code before use. It is provided as-is, with no warranty.

## How it works
1. **TransNetV2**, a neural shot-boundary detector, finds every cut in the video.
2. Each cut is scored to tell a *new ad* apart from a *new shot within the same ad*:
   - blank, gray, or tape-noise frames between ads (only counted when the sound or picture also changes across them, so dark scenes and fades inside an ad don't split it)
   - a change in the audio (announcer, music, room tone) before vs. after the cut
   - a change in the overall look of the picture
   - a change in **frame size** (black borders / pillarboxing), only trusted when the borders are steady on both sides of the cut, so dark or spotlit scenes don't fake it
   - a change in **color** (tint and saturation)
   - a change in **picture texture** (grain and sharpness, which usually changes with a different film print)
   - a drop in audio level at the cut

   The size, color, and texture changes are judged against the other cuts nearby, so color and black-and-white tapes are each handled on their own terms.
3. If the tape uses the same transition effect between ads (e.g. a striped "venetian blind" wipe added by whoever made the compilation), the script detects it when it repeats 4+ times **and** most of those appearances also look like real ad breaks (new sound and picture), then uses it as the main break marker. It learns the look of the transition from the clearest examples and searches the tape again for fainter copies of it. On those tapes, ordinary cuts need much stronger evidence to split an ad, and the wipe itself is trimmed off both neighbouring clips.
4. A dynamic-programming pass picks the set of cuts that best fits real ad lengths (mainly 15/20/30/60 s, with 10/45/90/120 s less likely). It allows for ads that were trimmed short in transfer.
5. A safety net then re-checks long clips with odd lengths and splits them when they look like two or three ads stuck together, or when one break clearly stands out from every other cut in the clip. A transition made of two quick cuts (e.g. a credit card between ads) counts as one break. This includes the first and last clip of a tape.
6. Black, gray, fuzzy, and static frames are trimmed so that no more than 1 second (configurable) is left at the start or end of any clip. Long gaps and stretches of static are never exported.
7. Each ad is exported as an H.264/AAC MP4 with normalized loudness (-16 LUFS).
   Uncertain clips go to a `_review` subfolder: the first and last clip of a tape (may be partial), clips over 75 s with an odd length (possible stuck-together ads), very short clips, and clips with a weak boundary.

## Installation
Requirements: Python 3.10+, ffmpeg/ffprobe on your PATH, and the packages in `requirements.txt`.

**Windows (easiest):** download or clone the repo and double-click `setup_windows.bat`. It installs Python and ffmpeg with winget if they're missing (close and reopen the window when it asks), then the Python packages.

**Manual / other systems:**
```
git clone <this repo>
cd <repo folder>
python -m pip install -r requirements.txt
```
`requirements.txt` pulls the smaller CPU build of PyTorch. If you have an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org first. It's used automatically.

> On Windows, if `python` isn't found, use `py` instead (e.g. `py -m pip install -r requirements.txt`, `py split_commercials.py ...`).

## Files
| File | What it is |
|---|---|
| `split_commercials.py` | The script |
| `requirements.txt` | Python packages |
| `setup_windows.bat` | One-time Windows setup (Python, ffmpeg, packages) |
| `CHANGELOG.md` | What changed in each version |

## Usage
```
python split_commercials.py "commercials_50s.mp4" -o "D:\NeXroll\Prerolls\50s Ads"
python split_commercials.py "D:\Tapes" -o "D:\NeXroll\Prerolls\Vintage"
```

The first line of output shows the script version, e.g. `(split_commercials 2026-11-16n)`. Check it after updating to make sure you're running the new file.

| Option | What it does |
|---|---|
| `-o, --out` | Output folder (required) |
| `--sensitivity N` | `-2` to `2`. Higher = more cuts (use when ads are stuck together); lower = fewer cuts |
| `--min-len` / `--max-len` | Shortest and longest allowed ad in seconds (default 6 / 150) |
| `--keep-blank N` | Max seconds of black/static left at the start or end of a clip (default 1; `0` removes it all) |
| `--no-blank` | Ignore blank frames (for tapes with gray fades *inside* ads) |
| `--no-separators` | Don't look for a repeating transition (wipe) between ads |
| `--skip-review` | Don't export uncertain clips at all |
| `--dry-run` | Only write the cut list (CSV) and contact sheet; no clips |
| `--refine` | Re-split **one clip file** (e.g. from `_review`) that may hold two or more ads. Its start and end are treated as real ad boundaries, and one clear break counts for more than ad length |
| `--join` | Join clip files back into one ad, in the order given, e.g. `python split_commercials.py tape_17_12s.mp4 tape_18_56s.mp4 tape_19_20s.mp4 --join -o "D:\NeXroll\Prerolls\80s Ads"`. Lossless and instant; the separate clips are moved to a `_replaced` folder next to them |
| `--split-at T` | Split a clip file by hand at these times, e.g. `--split-at 23.4` or `--split-at 0:23.4,0:51` (no analysis, instant) |
| `--rescan-review` | After splitting a tape, automatically re-split every `_review` clip with `--refine` |
| `--no-prompt` | Never ask questions at the end of a run (for batch use) |
| `--crf N` | x264 quality (default 20; lower = better/bigger) |

## Reviewing clips
At the end of a run the script lists every clip in `_review` with the reason, plus ready-to-run commands for that **clip file only** (not the whole tape):

```
#03  tape_03_112s.mp4  (112.2s) - unusual length (check for two ads)
   auto:    python split_commercials.py "...\_review\tape_03_112s.mp4" -o "...\_review\tape_03_112s_split" --refine
   harder:  python split_commercials.py "...\_review\tape_03_112s.mp4" -o "...\_review\tape_03_112s_split" --refine --sensitivity 1
   likely break points inside this clip (seconds): 23.4,51.0
   by hand: python split_commercials.py "...\_review\tape_03_112s.mp4" -o "...\_review\tape_03_112s_split" --split-at 23.4
```

These commands are also saved to `<name>_review_commands.txt`.

Then it asks what to do next. This works in any terminal (Command Prompt, PowerShell, Git Bash, VS Code, IDE run windows). If no keyboard input is available, e.g. in a scheduled job, it says so and skips the menu. The commands above still work later:

| Type | What happens |
|---|---|
| `c` | They all look fine: move them all into the main folder |
| `c 2,5` | Only these look fine: move them into the main folder |
| `r` | Rescan them all with finer splitting |
| `r 2,5` | Rescan only these (add `+` for extra-fine, e.g. `r+ 2,5`) |
| `j 17-19` | Join clips back into one ad when one ad was split into several (`j 17,18,19` also works). Works on clips in the main folder or `_review`; the joined ad goes to the main folder and the separate clips are moved to `_replaced` |
| Enter | Finish (anything left in `_review` stays there) |

The `j` (join) option is offered even when nothing is in `_review`. After a rescan that finds more than one ad, it asks whether to use the pieces. If you say yes, the pieces go to the main folder and the original clip is kept in `_review\_replaced`.

## Output
- `<name>_NN_<seconds>s.mp4`: one clip per confident ad
- `_review/`: clips to check by hand before using
- `<name>_segments.png`: contact sheet showing the first, middle, and last frame of every clip
- `<name>_segments.csv`: start/end times, length, status (`ok`, `review`, or `skipped` for blank/static), reason, transition note, and file for each clip
- `<name>_review_commands.txt`: the review commands described above

## Clips cut short by transition wipes
On tapes with a repeating transition wipe, the wipe often overlaps the ads themselves. The script checks each clip edge next to a wipe and flags it as **CUT SHORT** only when the ad is clearly still playing there: its sound is near full volume at the cut and there is real picture under the wipe. Those clips are listed in a warning at the end of the run, noted in the CSV's `transition note` column, and marked in red on the contact sheet. About 1-2 seconds of those ads are missing on the tape itself, so this can't be fixed by re-splitting.

## Notes
- The first and last clip of every tape always go to `_review`. A compilation often opens or closes with an intro or outro (title card, channel bumper, end slate). These usually don't match normal ad lengths, but they can end up inside the first or last ad clip because they overlay the ad, use a quick transition, or match the ad's pacing. Check those two clips by hand and trim them if needed.
- Tested on black-and-white 1950s, color 1970s and fast-cut 1990s compilations, including hard cuts with no gap between ads, dark night scenes inside an ad, title/credit cards between ads, striped wipe transitions, and tape static. `--refine` was tested on two-ad clips (it found the right break in each) and on single ads that must stay whole, including a 2-minute spot.
- If ads are still stuck together, re-run with `--sensitivity 1`, or use the review menu / `--refine` on just those clips. If single ads get split, try `--sensitivity -1`.
- Long single ads (e.g. a 2-minute spot) land in `_review` because of their length. Check them and use `c` to clear them.
- Highly stylized ads (e.g. 1980s car spots with a silent intro, neon graphics and big color/look changes) can be split into several pieces. Use `j` in the menu or `--join` to put them back together.
- Short sponsor tags or still cards at the start of an ad (e.g. a 20-second product card before a cartoon spot) may stay attached to the ad. Use `--split-at` if you want them separate.
- On CPU it processes roughly 1–2× faster than real time. If you have a CUDA build of PyTorch, an NVIDIA GPU is used automatically.
