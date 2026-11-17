# Changelog

All notable changes to `split_commercials.py`. The version is printed on the first line of every run, e.g. `(split_commercials 2026-11-16n)`.

## 2026-11-16n
### Fixed
- Fast-cut 1990s ads with dark or spotlit scenes were split into several pieces while the real break between two ads was missed. The black-border measurement jumped around from shot to shot and faked a "frame size change". A frame-size change now only counts when the borders are steady on both sides of the cut.
- `--refine` now gets all 12 test clips right (up from 11).

## 2026-11-16m
### Added
- Join clips back into one ad when a single ad was split into several:
  - in the end-of-run menu: `j 17-19` or `j 17,18,19` (works on main-folder and `_review` clips)
  - from the command line: `--join` with the clip files in order
- Joining is lossless when the clips came from this script; it falls back to re-encoding otherwise. The separate clips are moved to `_replaced`, never deleted.
- The menu is now offered even when nothing is in `_review`, so clips can still be joined.

## 2026-11-16l
### Fixed
- The review menu only appeared in some terminals and silently didn't appear in others (Git Bash, VS Code, IDE run windows, .bat files). It now always appears, and the script says why when it can't ask (dry run, `--no-prompt`, nothing to review, no keyboard input).

## 2026-11-16k
### Added
- **Review guidance:** at the end of a run, each `_review` clip is listed with its reason, the likely break points inside it, and ready-to-run commands for that clip file only. These are also saved to `<name>_review_commands.txt`.
- **Review menu:** `c` (clips are fine, move to main folder), `r` (rescan finer), `r+` (extra-fine), Enter (finish). After a rescan that finds more than one ad, it asks whether to use the pieces; the original is kept in `_review/_replaced`.
- `--refine`: re-split one clip file that may hold several ads (tuned on two-ad clips and on single ads that must stay whole).
- `--split-at`: split a clip by hand at given times, with no analysis.
- `--rescan-review` (rescan every review clip automatically) and `--no-prompt` (never ask; for batch runs).
### Fixed
- ffmpeg read keyboard input while exporting, so typing during a run could disrupt the export. It no longer does.

## 2026-11-16j
### Added
- **CUT SHORT warnings** on tapes with a repeating transition wipe: a clip edge next to a wipe is flagged only when the ad is clearly still playing there (sound near full volume at the cut and real picture under the wipe). Shown in a warning at the end of the run, in the CSV's new `transition note` column, and in red on the contact sheet.

## 2026-11-16i
### Fixed
- Faint wipes were missed on some machines (small numeric differences between ffmpeg/PyTorch builds), which made wipe tapes split badly. Wipe detection now runs in two passes: it learns the stripe pattern from the clearest wipes, then searches the tape again for fainter copies of the same pattern.

## 2026-11-16h
### Fixed
- Wipe mode could switch on falsely on tapes without wipes (striped scenery, interlacing) and make every ordinary cut harder to split. It now only switches on when most detected wipes also look like real ad breaks on their own, and single wipes that look like scenery are ignored.
### Added
- `--no-separators` turns wipe detection off completely.

## 2026-11-15g
### Added
- **Repeating transition detection:** if a compilation puts the same transition (e.g. a striped "venetian blind" wipe) between ads 4+ times, it's used as the main break marker, ordinary cuts on that tape need stronger evidence, and the wipe itself (including its fade in/out) is trimmed off both neighbouring clips.

## 2026-11-15f
### Added
- Black, gray, fuzzy, and static frames are trimmed so no more than 1 second is left at the start or end of any clip.
- `--keep-blank N` sets that limit (`0` removes it all).
- Clips that are mostly blank or static are skipped (status `skipped` in the CSV).

## 2026-11-15e
### Fixed
- A break made of two quick cuts (e.g. a credit card between two ads) wasn't treated as one strong break, so stuck-together ads at the end of a tape weren't split.

## 2026-11-15d
### Fixed
- A "clearly strong break" is now judged against the other cuts in the same clip, not a fixed score, so it still works when scores shift with the surrounding tape.

## 2026-11-15c
### Fixed
- The first and last clip of a tape were excluded from the stuck-together check. They now get it too, but only at a clearly strong break.
### Added
- The script version is printed at the start of every run.

## 2026-11-15
### Added
- New signals for telling ads apart: change in **frame size** (borders/pillarboxing), **color** (tint and saturation) and **picture texture** (grain/sharpness), each judged against the other cuts nearby.
- Long gray gaps and tape static always count as a break and are not exported.
- Clips that look like two **or three** ads stuck together are split.
- A long clip with one clearly strong break (sound and picture both change) is split even when the pieces aren't standard ad lengths.
### Changed
- 10-, 90- and 120-second lengths are treated as less likely.
- Only clips over 75 s with an odd length go to `_review` for length. Odd lengths are common on these tapes and usually fine.

## 2026-11-14
### Fixed
- Dark scenes and fades *inside* an ad (e.g. night shots) were taken as the gray gap between ads. A blank moment now only counts as a break if the sound or picture also changes across it.

## 2026-11-11 (later)
### Fixed
- Ads trimmed a few seconds short in transfer (e.g. 55 s instead of 60 s) were merged with the next ad. The length rule now allows ads to run short of a standard length.
- Two ads stuck together (e.g. 59 s + 58 s showing as one 117 s clip) are now split.
- The first and last clip of a tape could grow without limit and swallow several ads.

## 2026-11-11
### Added
- Initial version: TransNetV2 shot detection; blank/gray gap, audio change, picture change and audio-dip scoring; best-fit selection against real ad lengths; H.264/AAC export with normalized loudness; `_review` folder; contact sheet and CSV; `--sensitivity`, `--min-len`, `--max-len`, `--no-blank`, `--skip-review`, `--dry-run`, `--crf`.
