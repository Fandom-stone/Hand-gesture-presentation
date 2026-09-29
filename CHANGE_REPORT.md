# Change Report — Post-Review Improvements

Gesture-Controlled Presentation System · implemented from the professor-level review

---

## 1. Files modified

| File | Status | What changed |
|---|---|---|
| `src/evaluation.py` | **new** (207 lines) | Time-based splitting, action-level accuracy, shared report writing |
| `src/dataset_io.py` | **new** (48 lines) | Shared CSV creation/tallying for both collectors |
| `tests/test_pipeline.py` | **new** (603 lines) | 73 unit tests over every pure-logic component |
| `tests/__init__.py` | **new** | Package marker so `unittest discover` works |
| `src/config.py` | modified | Added `min_clips_for()`, `STATIC_FOREST_TREES`, `DYNAMIC_FOREST_TREES`, `EVALUATION_SPLIT` |
| `src/train_model.py` | rewritten | Dual-split + dual-metric reporting; evaluation logic moved out |
| `src/train_dynamic_model.py` | rewritten | Same, plus cleaning filters changed to return masks |
| `src/infer_realtime.py` | modified | No more config mutation; dynamic classifier gated on travel distance |
| `src/collect_data.py` | modified | Uses `dataset_io` (−25 lines) |
| `src/collect_dynamic_data.py` | modified | Uses `dataset_io`; `min_clips_for` moved to config (−30 lines) |
| `README.md` | modified | New Evaluation and Tests sections; training flags; updated limitations |

All 11 files pushed to `D:\college\S3\gesture-presentation-control` and **verified byte-identical by md5sum**.

---

## 2. Problems fixed (🔴 Must Fix)

### The accuracy figure was measuring the wrong thing

The review's central finding. The collector captures continuously while you hold `c`, so consecutive rows are frames ~28ms apart — nearly the same image. A random train/test split scattered those near-duplicates across both sides, so the model scored ~100% by recognising rows it had effectively already seen.

**Fixed** by splitting each class *by capture time* — earliest 80% trains, latest 20% tests — using the `timestamp` column that was already being recorded but never used. Near-duplicates now stay on one side of the split.

| Split | Label accuracy | Action accuracy |
|---|---|---|
| Random (what the report used to quote) | 100.00% | 100.00% |
| **Temporal (now the default)** | **90.07%** | **100.00%** |

Both figures print on every run, so the comparison itself is available for your report — the honest number *and* the evidence that the old one was inflated. `--split random` still exists if you want to deploy the other model.

### The headline number understated the system

Label accuracy penalises every confusion equally, but you only notice a confusion that changes what the program does. The entire 10-point gap above is NEUTRAL being read as OPEN_PALM — **both are deliberately actionless**, so confusing them produces identical behaviour.

**Fixed** by reporting action-level accuracy alongside. On the honest test set the system would have taken the correct action on **every single frame** (100.00%). NEUTRAL's recall is ~47%; every other class is 100%.

Both numbers belong in your report: label accuracy is the conventional classifier metric, action accuracy is the one that describes the system.

---

## 3. Improvements made (🟠 Should Improve)

### Forest size cut — 2.7× faster inference, no accuracy cost

Measured across 3 random seeds on the honest split:

| Model | Was | Now | Label accuracy | Per-frame cost |
|---|---|---|---|---|
| Static | 200 trees | **75** | 0.9105 → 0.9093 | 11.9ms → **4.4ms** |
| Dynamic | 150 trees | **50** | 1.000 → 1.000 | 8.5ms → **3.0ms** |

The static accuracy difference (0.0012) is smaller than the seed-to-seed noise (±0.006), i.e. not a real difference. Combined per-frame model time drops **~20ms → ~7ms** — handed straight back to the frame rate, which is the thing that was limiting your swipe detection.

### Dynamic classifier skipped when the hand hasn't moved

If travel is below `MIN_SWIPE_DX`, `SwipeFirer` vetoes the result regardless of what the model says — so the prediction could only cost time. It's now skipped on those frames, which is most frames of a normal session.

Deliberately active **only under `--quiet`**. Without it, the `[swipe blocked] travelled 0.041, needs 0.05` diagnostics are the whole point of the run and need the label the model would have given. So you keep full diagnostics while tuning and get the speedup while presenting.

### Dependency direction fixed

`train_dynamic_model.py` imported `min_clips_for` from `collect_dynamic_data.py` — a trainer reaching into a collection tool, backwards. Moved to `config.py` where the other threshold helper already lived.

### Config globals no longer mutated at startup

`--target` and `--quiet` used to write back into the `config` module, which meant a value could differ from what you'd read in the source file, and made it impossible to test both targets in one process. They're now passed as arguments. A test asserts both targets work simultaneously.

### Duplicated code extracted

The two trainers had identical `evaluate_model` and `save_confusion_matrix_plot` functions; the two collectors had identical CSV creation and tallying. Both sets moved into shared modules.

I deliberately **did not** merge the collectors' capture loops — holding a key to rack up single-frame samples and recording one fixed-length motion clip are genuinely different interactions, and merging them would produce something more complex than either.

---

## 4. New dependencies

**None.** The test suite uses `unittest` from the standard library. `requirements.txt` is unchanged.

---

## 5. Retraining required — yes

Your machine still has models trained with the old settings. Run both:

```
python -m src.train_model
python -m src.train_dynamic_model
```

Takes under a minute. This is what gives you the faster models and writes the new reports and confusion matrices. **The reports it produces are what you should quote in your documentation** — they now contain both splits and both metrics with the methodology explained inline.

---

## 6. Tests performed

### Behavioural regression check — passed

I captured a 67-line transcript of every pure-logic component's output *before* making any changes (feature extraction across 6 motion patterns, the tracker's dropout handling, the cleaning filters with checksums, augmentation with checksums, 9 debounce scenarios, 10 swipe-gate scenarios, and dispatch on both targets), then re-ran it after.

**Result: byte-identical.** Nothing that decides whether a slide moves behaves differently.

### Unit tests — 73 passing in 0.014s

```
python -m unittest discover -s tests -t .
```

No webcam, no trained model, no display needed.

### Mutation testing — the tests were themselves tested

Passing tests prove nothing if they'd pass on broken code. I deliberately reintroduced six real bugs and confirmed each was caught:

| Reintroduced bug | Caught |
|---|---|
| Pose needing more frames than the history holds | ✅ 7 failures |
| Runtime floor set stricter than the training floor | ✅ 2 failures |
| Giving OPEN_PALM an action | ✅ 3 failures |
| The majority-vote bug we fixed earlier | ✅ 4 failures |
| Degenerate-clip filter disabled | ✅ 1 failure |
| Speed formula broken | ❌ **missed** → added 3 tests → ✅ now caught |

The sixth one slipping through was the useful result: it exposed a real gap, which is now closed.

### End-to-end

Both trainers run clean on your real data. `--split random`, `--split temporal`, and a CSV with no `timestamp` column (backward compatibility) all work. Every file compiles.

---

## 7. Recommendations I deliberately skipped

| Recommendation | Why skipped |
|---|---|
| **Time-based motion features** (speed per second, not per frame) | Would require per-frame timestamps recorded inside each clip — your existing 230 clips don't have them, so this means re-recording the entire dynamic dataset. The current frame-based definition is at least *self-consistent*: the window length and `MIN_SWIPE_DX` are frame-based too. I added a test pinning the current behaviour so a future change has to be a decision rather than an accident, and documented it as a known limitation. |
| **Redesign the clip recorder** (fixed window catching the return motion) | Same problem — it would invalidate all 230 recorded clips. The two cleaning filters already handle the artifact after the fact, and they're now tested. Not worth re-collecting a working dataset weeks before submission. |
| **MediaPipe Gesture Recognizer baseline** | You already compare three algorithms (Random Forest, KNN, SVM). Adding a fourth from a different model family adds a dependency surface and report complexity without teaching anything the existing comparison doesn't. |
| **Merging the two collectors** | Their capture semantics genuinely differ. The merged version would be harder to read than the two separate ones. I extracted only the truly duplicated CSV handling. |
| **Refitting the deployed model on 100% of the data** | Standard practice after evaluation, and it would give you a marginally better model — but it decouples the deployed model from the reported accuracy, which is a subtlety that would need explaining in your report. Left as a future option. |

---

## 8. What to improve next

**1. Collect data from three teammates.** *(Highest value by a wide margin.)* Every sample so far is your hands. The classifier generalises only as far as its training data's diversity, and this is now the single biggest weakness — and the one your professor is most likely to probe. ~15 minutes per person.

**2. Correct §5.2, §13.1 and §13.2 of the report document.** They still say dynamic gestures are out of scope, which stopped being true when the swipe classifier was built. Documentation fix, no code.

**3. Run the §8.6 testing protocol** — different lighting conditions, multiple presenters. With teammate data collected, this becomes straightforward and gives you a real results section.

**4. Collect more dynamic clips.** The swipe test set is 34 real clips. By the rule of three, the true error rate could be ~9% despite the observed 100%. Another 40–50 clips per class would make that figure meaningful.

**5. Consider NEUTRAL's 47% recall.** It doesn't hurt you — the misses land on OPEN_PALM, which is equally actionless — but if a professor asks why one class scores so much worse, the honest answer is that NEUTRAL is the "everything else" class and its boundary with a relaxed open hand is genuinely blurry. More varied NEUTRAL samples would sharpen it.
