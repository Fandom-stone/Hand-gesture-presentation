# Gesture-Controlled Presentation System

Control a slide presentation with hand gestures through a standard webcam —
no remote, keyboard or mouse. A webcam frame is reduced to 21 hand landmarks
by MediaPipe, and lightweight scikit-learn classifiers trained on
self-collected data turn those landmarks into slide-navigation commands.

Built for the Artificial Intelligence & Data Science course, B.Tech ECE
Semester 3.

---

## What it does

| Gesture | Action | Desktop *(default)* | Web |
|---|---|---|---|
| Thumbs up | Start presentation | F5 | Ctrl+F5 |
| Swipe right (open hand) | Next slide | Right arrow | Right arrow |
| Swipe left (open hand) | Previous slide | Left arrow | Left arrow |
| Fist | End presentation | Esc | Esc |
| Pointing | **Laser pointer** | Ctrl+L | *(no laser mode)* |
| Pointing (with `--pointer`) | **Laser dot follows your fingertip** | Ctrl+L + cursor | moves the cursor |
| Open palm / Neutral | *(no action)* | — | — |

Open palm and neutral are deliberately actionless. Open palm is the swiping
hand shape, so giving it a command would fire on every swipe; both are still
trained classes so the classifier can recognise them and stay quiet.

---

## Getting started

Follow these in order. The whole thing takes about 30 minutes the first time,
most of it recording hand samples.

### Before you start

| You need | Notes |
|---|---|
| **Python 3.11 or 3.12** | Check with `python --version`. 3.13 is not supported by MediaPipe yet. |
| **A webcam** | Any built-in laptop camera is fine. |
| **Windows** | The `.\go` launcher and the PowerPoint shortcuts assume Windows. Everything else is cross-platform. |
| **PowerPoint** | Desktop app by default (full shortcuts + laser). Web works too — use `.\go web`. |

Open a terminal **in this folder**: in File Explorer, Shift + right-click the
folder and pick *Open PowerShell window here*.

### Step 1 — Install

Run these four, once:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
```

`python -m venv venv` creates a `venv\` folder holding this project's
libraries, kept separate from the rest of your machine.

#### Activating the venv

Run this from the project folder any time you want to use plain `python`:

```powershell
venv\Scripts\activate
```

You'll see `(venv)` appear at the start of your prompt. To leave it again,
type `deactivate`.

Without it, `python` means your *system* Python, which doesn't have this
project's libraries, and you get:

```
ModuleNotFoundError: No module named 'cv2'
```

**You usually don't need it.** Everything in this README uses `.\go`, which
runs the venv's interpreter directly — so after Step 1 you can forget the venv
exists.

| Command | Needs `activate` first? |
|---|---|
| `.\go anything` | No |
| `venv\Scripts\python.exe -m src.collect_data` | No |
| `python -m src.collect_data` | **Yes** |

If activation fails with an execution-policy error, run this once in the same
window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

`download_model.py` fetches MediaPipe's hand-detection model into `assets/`.
It's ~7 MB and is not in the repository, so this step is not optional.

### Step 2 — Check your camera

```powershell
python scripts/check_webcam.py
```

A window should open showing you. Press `q` to close it. **Fix any problem
here before going further** — nothing else works without a camera.

### Step 3 — Teach it your hands

**This step is required.** The system learns from examples, and out of the box
it has none of yours. There are two separate collectors because there are two
separate things to learn.

```powershell
.\go collect
```

Press `1`–`5` to pick a pose, then **hold `c`** to record. Aim for:

| Pose | How many | What it looks like |
|---|---|---|
| NEUTRAL | 250+ | Relaxed hand, doing nothing in particular |
| OPEN_PALM | 150+ | Palm to camera, fingers spread |
| FIST | 150+ | All fingers curled in |
| POINTING | 150+ | Index finger only |
| THUMBS_UP | 150+ | Thumb up, other fingers curled |

The on-screen counter turns green when a pose has enough. Move your hand
around between captures — different distances, angles and positions — or the
model only learns the one spot you sat in.

```powershell
.\go swipes
```

Press `1`–`3` to pick a class, then press `c` **once** and perform the motion.

| Class | How many | What to do |
|---|---|---|
| SWIPE_LEFT | 40+ | Open hand, sweep right → left |
| SWIPE_RIGHT | 40+ | Open hand, sweep left → right |
| NO_SWIPE | 60+ | Everything that *isn't* a swipe |

Two things people get wrong here:

- **Don't bring your hand back** until the progress bar fills. The recorder
  captures a fixed window, so a quick return trip lands in the same clip and
  cancels it out.
- **NO_SWIPE matters most.** It's what teaches the system *not* to fire. Record
  a still hand, held poses, small fidgets, slow drift, hand entering and
  leaving frame. Skimp on it and every small movement changes your slides.

**What the trainer fixes for you.** Clips where the hand came back inside the
window get dropped automatically — on the recorded dataset that was 30 of 230
reversed plus 27 with almost no travel. The training split is also expanded by
mirroring and speed-scaling (139 real clips → 1,112 rows); the test set stays
real recorded clips, so the accuracy stays honest.

### Step 4 — Train

```powershell
.\go train
```

Under a minute. This creates `models/model.pkl` and `models/dynamic_model.pkl`,
which is what the live program loads. **Re-run both any time you add data or
change settings in `config.py`** — otherwise your changes do nothing.

### Step 5 — Try it without risk

```powershell
.\go dry
```

Gestures are recognised and printed, but **no keys are sent anywhere**. Do each
gesture and watch the terminal. When that looks right, you're ready.

### Step 6 — Present

```powershell
.\go
```

That's it — the launcher activates the venv and starts presenting mode. Open
your slides and **click on them first**, because keystrokes go to whatever
window is in front. Stop with Ctrl+C.

---

## If something goes wrong

| What you see | What it means |
|---|---|
| `No module named 'cv2'` (or mediapipe, sklearn…) | Wrong Python. Run `.\go which` to see which one is being used. Using `.\go` avoids this entirely; if you're calling `python` yourself, activate the venv first. |
| `No trained static model found` | You skipped Step 4. Run both training commands. |
| `Could not load HandLandmarker model` | You skipped `download_model.py` in Step 1. |
| `Could not open webcam` | Another app (Zoom, Teams, Camera) is using it. Close them. |
| Gestures recognised but slides don't change | Your slides aren't the focused window. Click them. The program prints where each keypress landed. |
| Swipes fire on any small movement | Almost always a gap in NO_SWIPE data, not a threshold. Record NO_SWIPE clips of the movements that misfire — repositioning, raising into a pose, hand entering frame — then retrain. Raising `MIN_SWIPE_DX` mostly costs you real swipes. |
| Pointing steals a slide | The sweep blurs your hand, so the pose vanishes from the per-frame labels just as the swipe window fills. `SWIPE_POSE_MEMORY_SECONDS` guards this by remembering the pose instead of counting frames — raise it from 1.0 if it still happens. |
| Swipes never fire | Run `.\go tune` and read the printed reason — it tells you exactly which check failed. |
| Program exits the moment a gesture fires | Fixed — the program used to be killed by its own Esc keypress. If you see it, you're on an old copy. |
| Everything feels laggy | Check the fps number, top-right. Below 15 is a problem; `.\go` (no preview) is much faster than `.\go tune`. |

Still stuck? `.\go tune` explains every decision it makes as it makes them,
and prints a summary when you quit.

---

## Day-to-day use

```powershell
.\go            # present on DESKTOP PowerPoint (default)
.\go web        # present on PowerPoint for the web instead
.\go collect    # record static pose samples
.\go swipes     # record swipe clips
.\go tune       # camera window + diagnostics, for debugging
.\go dry        # recognise only, send no keypresses
.\go train      # retrain both models
.\go test       # run the test suite (~1 second, no webcam needed)
.\go profile    # measure where startup time goes
.\go which      # show which Python and libraries it will use
```

`go.bat` calls `venv\Scripts\python.exe` by full path rather than running
`activate.bat` first. That matters if your folder name contains brackets —
`activate.bat` expands `%PATH%` inside parenthesised `if` blocks, so a `)` in
the path closes the block early, mangles `PATH`, and Python silently falls back
to the system install. The symptom is a baffling `No module named 'cv2'` with
the venv apparently active. Naming the interpreter directly avoids activation
entirely, so the folder can be called anything.

If you ever see that error, run `.\go which` — it prints exactly which
interpreter and libraries are being used.

Or call the modules directly if you prefer — `go.bat` is only a convenience
wrapper, and all the flags below still work.

### Training flags

| Flag | Purpose |
|---|---|
| `--split temporal` | Default. Deploy the model trained on the time-based split |
| `--split random` | Deploy the model trained on the random split instead |
| `--csv <path>` | Train from a different dataset file |
| `--test-size 0.2` | Fraction held out for testing |

Both accuracies on both splits are printed whichever you choose — `--split`
only decides which model gets saved.

### Inference flags

| Flag | Purpose |
|---|---|
| `--present` | **Shorthand for `--no-preview --quiet --pointer`** — what you want in front of an audience |
| `--dry-run` | Recognise and report, send no keypresses |
| `--quiet` | Suppress per-gesture "why it didn't fire" lines (use when presenting) |
| `--no-preview` | No camera window, so it cannot take keyboard focus. Press `q` in the terminal to stop |
| `--target desktop` | Installed PowerPoint app — full shortcuts and laser (**default**) |
| `--target web` | powerpoint.com in a browser — avoids shortcuts the browser steals |
| `--pointer` | Fingertip moves the real mouse cursor — see below |
| `--camera 1` | Use a different webcam |

Extra flags pass straight through the launcher too: `.\go tune --camera 1`.

### Laser pointer / fingertip cursor (`--pointer`)

Hold the **pointing** pose and the mouse follows your index fingertip. Release
the pose and it stops.

On **desktop PowerPoint** this is a real laser: raising the pose sends Ctrl+L
to switch PowerPoint into laser mode, your finger moves the red dot, and
lowering the pose sends Ctrl+A to restore the ordinary arrow. The mode keys are
sent once per raise and once per lower — never per frame — and the arrow is
restored automatically if you quit mid-point.

On the **web** player there is no laser mode, so the same pose moves the
ordinary mouse cursor. Still a usable pointer, just not a red dot.

It needs no model of its own — the static classifier already recognises
POINTING, and where the fingertip aims is pure geometry. `src/pointer.py`
smooths the jitter (heavier when your hand is slow, lighter when it's fast),
maps only the middle ~70% of the frame so the screen corners stay reachable,
and needs 3 steady frames to take over so one bad frame can't grab the cursor.
Tuning is in `config.py`.

**Quit with `q`, not Esc.** FIST *sends* Esc to end the slideshow, and from an
editor's built-in terminal that Esc comes straight back into the same console —
so Esc as a quit key meant ending a presentation also killed the program.

Keypresses go to whichever window has focus, so **click your slides first**.
Each fired action prints the window that received it.

---

## How it works

```
Webcam → MediaPipe HandLandmarker → 21 landmarks
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
        STATIC classifier                         DYNAMIC classifier
   63-value normalised landmark                18-frame window of hand
   vector → held-pose label                    positions → 8 motion
                 │                              features → swipe label
                 ▼                                         ▼
        StableGestureTracker                        SwipeFirer
     (majority vote + lock + cooldown)      (travel floor + pose gate + cooldown)
                 └────────────────────┬────────────────────┘
                                      ▼
                          pyautogui keypress → PowerPoint
```

Two independent classifiers, deliberately not combined — their feature
vectors have different shapes and meanings. Each has its own dataset, model
file and training script.

**Static** — one frame in, one held-pose label out. Landmarks are translated
so the wrist is the origin and scaled so the furthest landmark sits at
distance 1.0, making the features invariant to hand position and distance
from the camera. Rotation is deliberately *not* normalised away, since some
gestures differ only by orientation (thumbs up vs pointing).

**Dynamic** — a rolling window of hand-centre positions reduced to eight
hand-crafted motion features (displacement, path length, straightness, mean
and max speed, direction consistency). This is a deliberate alternative to a
sequence model such as an LSTM: it keeps the project on one scikit-learn
stack, needs far less data, and trains in a fraction of a second.

Random Forest is the deployed model for both; KNN and SVM are trained
alongside as comparison baselines for the evaluation section.

---

## Tuning

All thresholds live in `src/config.py`. The values below were set by
measurement, not guesswork — the figures are from real recorded sessions.

| Setting | Value | Why |
|---|---|---|
| `MIN_SWIPE_DX` | 0.05 | Hand must travel this fraction of frame width. The largest idle clip ever recorded travelled 0.027; real swipes reach 0.80. Raise toward 0.07 if swipes fire unintentionally. |
| `SWIPE_BLOCKING_POSE_FRACTION` | 0.5 | Block a swipe when a command pose is held for this much of the window, so moving a fist sideways doesn't change slides. |
| `SWIPE_POSE_MEMORY_SECONDS` | 1.0 | **The gate that actually holds.** After a command pose is confirmed, swipes stay blocked for this long past its last sighting. The two frame-counting gates below both fail in the same place: sweeping your hand blurs it, the classifier honestly reports NEUTRAL/NONE, and the pose leaves the history exactly as the swipe window fills — so pointing and moving your arm read as a swipe. A timestamp doesn't care whether the current frame is readable. Raise it if pointing still steals a slide; lower it if you must swipe immediately after pointing. |
| `SWIPE_BLOCKING_RECENT_POSES` | 3 of last 5 | Blocks the *transition* into a pose. Raising your hand into pointing sweeps it sideways — real travel that the fraction gate above misses, because the pose only occupies the tail of the window. Lower it if pointing still steals a slide; raise it if real swipes get blocked. |
| `STABLE_FRAMES_REQUIRED` | 6 of 12 | Frames a pose must occupy before firing. Raise if a pose fires unintentionally. |
| `ACTION_COOLDOWN_SECONDS` | 1.2 | Stops one held pose firing repeatedly. |
| `SWIPE_ACTION_COOLDOWN_SECONDS` | 0.8 | One physical swipe spans several overlapping windows; this collapses them into a single slide change. |
| `MIN_PREDICTION_CONFIDENCE` | 0.70 | Below this a pose is treated as unrecognised. |
| `MIN_DYNAMIC_PREDICTION_CONFIDENCE` | 0.80 | Higher than the static threshold: a false swipe skips a slide, which is more disruptive than a missed pose. |
| `SWIPE_MAX_MISS_SECONDS` | 0.5 | How long the hand may go undetected mid-swipe. Expressed in seconds, not frames, so it stays correct at any frame rate. |
| `STATIC_FOREST_TREES` | 75 | Accuracy is flat from 50 to 200 trees (the spread is smaller than seed-to-seed noise), but prediction cost is linear in tree count: 4.4ms vs 11.9ms per frame. |
| `DYNAMIC_FOREST_TREES` | 50 | Same reasoning. The swipe problem is easy enough that even 25 trees scored 100%; 50 costs 3.0ms instead of 8.5ms. |

Together the two tree counts cut per-frame model time from ~20ms to ~7ms —
headroom handed straight back to the frame rate.

### Diagnostics

Without `--quiet`, any recognised-but-suppressed gesture explains itself:

```
[swipe blocked] SWIPE_RIGHT: hand travelled 0.062 of the frame width, needs 0.05
[swipe blocked] SWIPE_LEFT: a command pose filled 3 of the last 5 frames -- that's
                a hand moving into a pose, not a swipe
[pose not fired] THUMBS_UP: held for only 4 of the last 12 frames, needs 6
```

On exit a session summary prints how many motion windows were analysed,
recognised and fired. Output goes to the terminal rather than the camera
window, which is hidden behind a fullscreen presentation exactly when you
need to read it.

The on-screen FPS counter is colour-coded: green at 15+, amber 8–14, red
below 8. Low FPS and a laggy-looking skeleton are the same symptom, and it
affects recognition — an 18-frame window takes over two seconds to fill at
8fps, far longer than a swipe lasts.

---

## Evaluation

Both training scripts report **two accuracies on two splits**, because the
obvious way to measure this system flatters it badly.

### Why not a random split

The collector captures continuously while `c` is held, so consecutive rows
are frames ~28ms apart — nearly the same image. Measured on the collected
dataset, 7,993 of 8,009 samples sit less than 100ms from their neighbour, and
consecutive frames are **10.6× more similar** to each other than two random
frames of the same pose (mean L2 distance 0.115 vs 1.218).

A random split scatters those near-duplicates across both sides, so for most
test rows an almost identical row sits in the training set. The model scores
~100% by recognising rows it has effectively already seen.

Splitting each class **by capture time** instead — earliest 80% train, latest
20% test — keeps near-duplicates on the same side. Only the single pair
straddling each class's cut point stays adjacent.

| Split | Label accuracy | Action accuracy |
|---|---|---|
| Random (optimistic) | 100.00% | 100.00% |
| **Temporal (honest)** | **90.07%** | **100.00%** |

The random figure not moving at all between 25 and 200 trees is itself the
giveaway: a genuinely hard test set would show *some* sensitivity.

### Why two accuracies

Label accuracy counts every confusion equally. But the user only notices a
confusion that changes what the program does, and NEUTRAL and OPEN_PALM are
both deliberately actionless — mixing them up produces identical behaviour.

That single confusion is the entire 10-point gap. NEUTRAL's recall on the
honest split is ~47%; every other class is 100%. Because the misses all land
on OPEN_PALM, **action accuracy is 100%**: on the honest test set the system
would have done the right thing every time.

Both numbers belong in the report. Label accuracy is the conventional
classifier metric; action accuracy is the one that describes the system.

### Limitations that remain

**The dynamic figure rests on a small test set** — 34 real held-out clips
after cleaning. By the rule of three, the true error rate could be as high as
~9% despite the observed 100%. Trustworthy in direction, not in precision.

**One presenter, one set of hands.** All data so far was collected by one
person. The classifier generalises only as far as the diversity of its
training data, so expect worse results for someone else until data from more
people is added. This is the single highest-value improvement left.

**Motion features are per-frame, not per-second.** `mean_speed` and
`max_speed` are distance per *frame*, so the same physical swipe reads as
slower on a faster machine. The window length and `MIN_SWIPE_DX` are
frame-based too, so the three agree with each other, but a per-second version
would need per-frame timestamps recorded inside each clip — which would
invalidate the existing dataset. A test pins the current definition so any
future change has to be deliberate.

Other limitations: accuracy depends on lighting and camera quality, and the
system tracks a single hand.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

129 tests over the pure-logic parts — feature extraction and normalisation,
the dataset loader, the cleaning filters, augmentation, the evaluation split,
the two debounce gates that decide whether a slide actually moves, and the
fingertip cursor's coordinate mapping, smoothing and activation hysteresis. No
webcam, no trained model, no display; the suite runs in about a second.

The camera, MediaPipe and pyautogui layers are deliberately untested: they
are thin wrappers over other people's libraries and need real hardware to
exercise meaningfully.

The suite itself was checked by deliberately reintroducing past bugs — a
stricter runtime floor than the training floor, the majority-vote bug, a
mirrored cursor axis, Esc quitting the program — and confirming each one fails
a test. A suite nobody has tried to break is decoration.

---

## Project layout

```
src/
  config.py              All paths, vocabularies, action maps and thresholds
  hand_tracker.py        MediaPipe wrapper, feature vector, webcam capture
  trajectory_features.py Sliding window and motion feature extraction
  evaluation.py          Splitting, action-level scoring, model line-up, reports
  dataset_io.py          Dataset CSV handling shared by collectors and trainers
  ui.py                  Shared on-screen overlay primitives
  keysend.py             Keystroke delivery and focused-window reporting
  pointer.py             Fingertip cursor: mapping, smoothing, activation
  collect_data.py        Static pose collection tool
  collect_dynamic_data.py Swipe clip collection tool
  train_model.py         Static classifier training and evaluation
  train_dynamic_model.py Swipe classifier training, cleaning and augmentation
  infer_realtime.py      Live inference and action dispatch
tests/
  test_pipeline.py       Unit tests for every pure-logic component
go.bat                   Launcher: .\go, .\go tune, .\go train, .\go test
scripts/
  download_model.py      Fetch the MediaPipe model bundle
  check_webcam.py        Verify camera access
  profile_startup.py     Break down where startup time goes
data/                    Collected datasets (CSV)
models/                  Trained models, confusion matrices, training reports
assets/                  hand_landmarker.task
```

Each CSV row carries a `timestamp` column. It is never used as a model input
— it records *when* each sample was captured, which is what makes the
time-based split possible.

## Dependencies

Pinned in `requirements.txt`: mediapipe, opencv-python, scikit-learn, pandas,
numpy, pyautogui, matplotlib, joblib. Versions and Python requirements are in
[Before you start](#before-you-start).
