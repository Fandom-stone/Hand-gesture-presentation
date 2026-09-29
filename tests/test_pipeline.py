"""
Tests for the parts of the system that are pure logic.

Everything here runs without a webcam, a trained model or a display, so the
whole suite finishes in about a second:

    python -m unittest discover tests

What it covers is the code that decides whether a slide changes -- feature
extraction, the dataset cleaning filters, the evaluation split, and the two
debounce gates. Those are exactly the parts where a silent mistake produces
"it randomly skips slides" rather than a crash, and the parts most likely to
be broken by a later edit.

The camera, MediaPipe and pyautogui layers are deliberately not tested: they
are thin wrappers over other people's libraries and need real hardware to
exercise meaningfully.
"""

from __future__ import annotations

import re
import unittest

import numpy as np

from src import config, dataset_io, evaluation, infer_realtime, pointer
from src.hand_tracker import landmarks_to_feature_vector
from src.trajectory_features import TrajectoryTracker, extract_features, hand_center
from src import train_dynamic_model as tdm


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


def straight_swipe(start_x, end_x, n=None, y=0.5):
    n = n or config.SWIPE_WINDOW_FRAMES
    return [(start_x + (end_x - start_x) * i / (n - 1), y) for i in range(n)]


# --------------------------------------------------------------------------
class TestLandmarkNormalisation(unittest.TestCase):
    def _hand(self, offset=(0.0, 0.0), scale=1.0):
        base = [(0.5, 0.5), (0.52, 0.45), (0.55, 0.40), (0.58, 0.36), (0.60, 0.32),
                (0.54, 0.30), (0.55, 0.22), (0.56, 0.17), (0.57, 0.13),
                (0.50, 0.29), (0.50, 0.20), (0.50, 0.15), (0.50, 0.11),
                (0.46, 0.30), (0.45, 0.22), (0.44, 0.17), (0.44, 0.13),
                (0.42, 0.32), (0.40, 0.26), (0.39, 0.22), (0.38, 0.19)]
        wx, wy = base[0]
        return [
            FakeLandmark(wx + (x - wx) * scale + offset[0], wy + (y - wy) * scale + offset[1])
            for x, y in base
        ]

    def test_vector_length_matches_config(self):
        self.assertEqual(
            len(landmarks_to_feature_vector(self._hand())), config.FEATURE_VECTOR_LENGTH
        )

    def test_wrist_becomes_the_origin(self):
        fv = landmarks_to_feature_vector(self._hand())
        np.testing.assert_allclose(fv[:3], [0.0, 0.0, 0.0], atol=1e-6)

    def test_furthest_landmark_sits_at_distance_one(self):
        pts = landmarks_to_feature_vector(self._hand()).reshape(config.NUM_LANDMARKS, -1)
        self.assertAlmostEqual(float(np.linalg.norm(pts, axis=1).max()), 1.0, places=5)

    def test_invariant_to_where_the_hand_is(self):
        """Same pose, different corner of the frame -> same features. This is
        what lets the classifier learn the gesture, not the position."""
        np.testing.assert_allclose(
            landmarks_to_feature_vector(self._hand()),
            landmarks_to_feature_vector(self._hand(offset=(0.3, -0.2))),
            atol=1e-5,
        )

    def test_invariant_to_distance_from_camera(self):
        """Same pose, hand twice as far away -> same features."""
        np.testing.assert_allclose(
            landmarks_to_feature_vector(self._hand()),
            landmarks_to_feature_vector(self._hand(scale=0.5)),
            atol=1e-5,
        )

    def test_degenerate_hand_does_not_divide_by_zero(self):
        flat = [FakeLandmark(0.5, 0.5) for _ in range(config.NUM_LANDMARKS)]
        fv = landmarks_to_feature_vector(flat)
        self.assertTrue(np.all(np.isfinite(fv)))


# --------------------------------------------------------------------------
class TestTrajectoryFeatures(unittest.TestCase):
    def test_feature_vector_length_matches_config(self):
        f = extract_features(straight_swipe(0.1, 0.8))
        self.assertEqual(len(f), config.DYNAMIC_FEATURE_VECTOR_LENGTH)

    def test_swipe_direction_shows_in_the_sign_of_dx(self):
        right = extract_features(straight_swipe(0.1, 0.8))
        left = extract_features(straight_swipe(0.8, 0.1))
        self.assertGreater(right[0], 0)
        self.assertLess(left[0], 0)

    def test_mirrored_swipes_differ_only_in_sign(self):
        right = extract_features(straight_swipe(0.1, 0.8))
        left = extract_features(straight_swipe(0.8, 0.1))
        self.assertAlmostEqual(float(right[0]), -float(left[0]), places=5)
        for i in range(1, len(right)):  # every other feature is direction-blind
            self.assertAlmostEqual(float(right[i]), float(left[i]), places=5)

    def test_straight_motion_is_maximally_straight(self):
        self.assertAlmostEqual(float(extract_features(straight_swipe(0.1, 0.8))[4]), 1.0, places=4)

    def test_speed_is_measured_per_frame_not_per_second(self):
        """A deliberate limitation, pinned here so a later edit has to be a
        decision rather than an accident: 'speed' is distance per FRAME, so
        the same physical swipe reads as slower at a higher frame rate. Fixing
        it properly means recording per-frame timestamps in the clips, which
        would invalidate the existing dataset. The 18-frame window and the
        MIN_SWIPE_DX floor are both frame-based too, so all three agree."""
        pts = straight_swipe(0.1, 0.8, n=18)
        f = extract_features(pts)
        expected_step = (0.8 - 0.1) / 17
        self.assertAlmostEqual(float(f[5]), expected_step, places=5)  # mean_speed
        self.assertAlmostEqual(float(f[6]), expected_step, places=5)  # max_speed

    def test_mean_speed_is_the_average_step_and_max_speed_the_largest(self):
        pts = [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0), (0.6, 0.0)]
        f = extract_features(pts)
        self.assertAlmostEqual(float(f[2]), 0.6, places=5)          # path_length
        self.assertAlmostEqual(float(f[5]), 0.6 / 3, places=5)      # mean over 3 steps
        self.assertAlmostEqual(float(f[6]), 0.4, places=5)          # largest single step
        self.assertLess(float(f[5]), float(f[6]))

    def test_a_faster_swipe_over_the_same_distance_reads_as_faster(self):
        slow = extract_features(straight_swipe(0.1, 0.8, n=18))
        fast = extract_features(straight_swipe(0.1, 0.8, n=6) + [(0.8, 0.5)] * 12)
        self.assertGreater(float(fast[6]), float(slow[6]))          # max_speed
        self.assertAlmostEqual(float(fast[0]), float(slow[0]), places=5)  # same dx

    def test_there_and_back_looks_nothing_like_a_swipe(self):
        """The recorder's fixed window can catch the hand's return motion.
        Such a clip travels far but ends near where it started -- low
        straightness is what makes it detectable."""
        out = straight_swipe(0.2, 0.8, n=9) + straight_swipe(0.8, 0.2, n=9)
        f = extract_features(out)
        self.assertLess(abs(float(f[0])), 0.05)      # tiny net dx
        self.assertGreater(float(f[2]), 1.0)          # but a long path
        self.assertLess(float(f[4]), 0.1)             # so barely "straight"

    def test_still_hand_produces_zeros(self):
        f = extract_features([(0.5, 0.5)] * config.SWIPE_WINDOW_FRAMES)
        np.testing.assert_allclose(f, np.zeros(config.DYNAMIC_FEATURE_VECTOR_LENGTH), atol=1e-7)

    def test_too_few_points_returns_zeros_not_a_crash(self):
        np.testing.assert_allclose(
            extract_features([(0.1, 0.1)]), np.zeros(config.DYNAMIC_FEATURE_VECTOR_LENGTH)
        )

    def test_hand_centre_averages_wrist_and_middle_mcp(self):
        lms = [FakeLandmark(0.0, 0.0)] * 21
        lms[0] = FakeLandmark(0.2, 0.4)
        lms[9] = FakeLandmark(0.6, 0.8)
        self.assertEqual(hand_center(lms), (0.4, 0.6000000000000001))


# --------------------------------------------------------------------------
class TestTrajectoryTracker(unittest.TestCase):
    def test_fills_then_reports_full(self):
        t = TrajectoryTracker(window_frames=4, max_miss_seconds=10.0)
        for i in range(3):
            t.update((0.1 * i, 0.5))
            self.assertFalse(t.is_full)
        t.update((0.4, 0.5))
        self.assertTrue(t.is_full)

    def test_brief_dropout_does_not_discard_the_window(self):
        """Motion blur mid-swipe loses the hand for a frame or two. Resetting
        on that would make fast swipes impossible to record."""
        t = TrajectoryTracker(window_frames=4, max_miss_seconds=10.0)
        t.update((0.1, 0.5))
        self.assertFalse(t.update(None))
        self.assertEqual(t.frame_count, 1)

    def test_sustained_loss_resets_the_window(self):
        t = TrajectoryTracker(window_frames=4, max_miss_seconds=-1.0)
        t.update((0.1, 0.5))
        self.assertTrue(t.update(None))
        self.assertEqual(t.frame_count, 0)

    def test_feature_vector_is_none_until_full(self):
        t = TrajectoryTracker(window_frames=3, max_miss_seconds=10.0)
        t.update((0.1, 0.5))
        self.assertIsNone(t.feature_vector())
        t.update((0.2, 0.5))
        t.update((0.3, 0.5))
        self.assertIsNotNone(t.feature_vector())


# --------------------------------------------------------------------------
class TestDatasetCleaning(unittest.TestCase):
    """The two filters that undo the recorder's fixed-window artifact."""

    def _clip(self, dx):
        f = np.zeros(config.DYNAMIC_FEATURE_VECTOR_LENGTH, dtype=np.float32)
        f[0] = dx
        return f

    def test_drops_clips_travelling_against_their_own_label(self):
        X = np.array([self._clip(-0.4), self._clip(0.4), self._clip(0.4), self._clip(-0.4)])
        y = np.array([config.SWIPE_LEFT, config.SWIPE_RIGHT,
                      config.SWIPE_LEFT, config.SWIPE_RIGHT])
        keep = tdm.direction_contradicting_mask(X, y)
        self.assertListEqual(keep.tolist(), [True, True, False, False])

    def test_no_swipe_clips_are_never_dropped_for_direction(self):
        X = np.array([self._clip(0.4), self._clip(-0.4), self._clip(0.0)])
        y = np.array([config.NO_SWIPE] * 3)
        self.assertTrue(tdm.direction_contradicting_mask(X, y).all())

    def test_drops_swipes_that_barely_moved(self):
        below = config.MIN_TRAINING_SWIPE_DX / 2
        X = np.array([self._clip(0.5), self._clip(below), self._clip(-below)])
        y = np.array([config.SWIPE_RIGHT, config.SWIPE_RIGHT, config.SWIPE_LEFT])
        self.assertListEqual(
            tdm.degenerate_swipe_mask(X, y).tolist(), [True, False, False]
        )

    def test_a_motionless_no_swipe_clip_is_kept(self):
        """NO_SWIPE is supposed to be motionless -- the degenerate filter must
        not delete the background class."""
        X = np.array([self._clip(0.0)])
        y = np.array([config.NO_SWIPE])
        self.assertTrue(tdm.degenerate_swipe_mask(X, y).all())

    def test_masks_keep_features_labels_and_timestamps_aligned(self):
        X = np.array([self._clip(0.4), self._clip(0.4), self._clip(-0.4)])
        y = np.array([config.SWIPE_RIGHT, config.SWIPE_LEFT, config.SWIPE_LEFT])
        ts = np.array([10.0, 20.0, 30.0])
        keep = tdm.direction_contradicting_mask(X, y)
        self.assertListEqual(ts[keep].tolist(), [10.0, 30.0])


# --------------------------------------------------------------------------
class TestAugmentation(unittest.TestCase):
    def _clips(self):
        X = np.array([
            [0.5, 0.1, 0.6, 0.5, 0.9, 0.03, 0.05, 1.0],
            [-0.5, 0.1, 0.6, 0.5, 0.9, 0.03, 0.05, 1.0],
            [0.0, 0.0, 0.01, 0.0, 0.0, 0.001, 0.002, 0.0],
        ], dtype=np.float32)
        y = np.array([config.SWIPE_RIGHT, config.SWIPE_LEFT, config.NO_SWIPE])
        return X, y

    def test_expected_row_count(self):
        X, y = self._clips()
        Xa, ya = tdm.augment_training_split(X, y)
        # original + mirror, then both again per speed factor
        expected = len(y) * 2 * (1 + len(config.DYNAMIC_AUGMENT_SPEED_FACTORS))
        self.assertEqual(len(ya), expected)
        self.assertEqual(Xa.shape[1], config.DYNAMIC_FEATURE_VECTOR_LENGTH)

    def test_mirroring_relabels_left_as_right(self):
        X, y = self._clips()
        Xa, ya = tdm.augment_training_split(X, y)
        mirrored_dx, mirrored_label = Xa[3, 0], ya[3]
        self.assertAlmostEqual(float(mirrored_dx), -0.5, places=5)
        self.assertEqual(mirrored_label, config.SWIPE_LEFT)

    def test_no_swipe_stays_no_swipe_when_mirrored(self):
        X, y = self._clips()
        _, ya = tdm.augment_training_split(X, y)
        self.assertEqual(int((ya == config.NO_SWIPE).sum()), 8)

    def test_ratio_features_are_left_alone_by_speed_scaling(self):
        """straightness and direction_consistency are ratios: performing the
        same motion faster must not change them."""
        X, y = self._clips()
        Xa, _ = tdm.augment_training_split(X, y)
        straightness = Xa[:, 4]
        consistency = Xa[:, 7]
        self.assertEqual(len(np.unique(np.round(straightness, 6))), 2)  # 0.9 and 0.0
        self.assertEqual(len(np.unique(np.round(consistency, 6))), 2)   # 1.0 and 0.0

    def test_class_balance_is_preserved(self):
        X, y = self._clips()
        _, ya = tdm.augment_training_split(X, y)
        counts = {c: int((ya == c).sum()) for c in config.DYNAMIC_GESTURES}
        self.assertEqual(counts[config.SWIPE_LEFT], counts[config.SWIPE_RIGHT])


# --------------------------------------------------------------------------
class TestTrainingCsvLoader(unittest.TestCase):
    """The loader both trainers now share."""

    COLS = ["a", "b"]

    def write(self, rows, path):
        import csv as _csv

        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(list(rows[0].keys()))
            for r in rows:
                w.writerow(list(r.values()))
        return path

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()

    def test_loads_features_labels_and_timestamps(self):
        import os

        path = self.write(
            [
                {"gesture": "A", "timestamp": 10.0, "a": 1.0, "b": 2.0},
                {"gesture": "B", "timestamp": 20.0, "a": 3.0, "b": 4.0},
            ],
            os.path.join(self.dir, "d.csv"),
        )
        X, y, ts, df = dataset_io.load_training_csv(path, self.COLS, "collect_data.py")
        self.assertEqual(X.shape, (2, 2))
        self.assertListEqual(list(y), ["A", "B"])
        self.assertListEqual(list(ts), [10.0, 20.0])
        self.assertEqual(len(df), 2)

    def test_row_order_stands_in_for_a_missing_timestamp_column(self):
        """Datasets recorded before timestamps existed must still train, and a
        split on row order is still better than random -- rows are appended in
        capture order."""
        import os

        path = self.write(
            [{"gesture": "A", "a": 1.0, "b": 2.0}, {"gesture": "B", "a": 3.0, "b": 4.0}],
            os.path.join(self.dir, "no_ts.csv"),
        )
        _, _, ts, _ = dataset_io.load_training_csv(path, self.COLS, "collect_data.py")
        self.assertListEqual(list(ts), [0.0, 1.0])

    def test_a_missing_feature_column_names_the_column_and_the_collector(self):
        import os

        path = self.write(
            [{"gesture": "A", "a": 1.0}], os.path.join(self.dir, "short.csv")
        )
        with self.assertRaises(ValueError) as ctx:
            dataset_io.load_training_csv(path, self.COLS, "collect_data.py")
        self.assertIn("'b'", str(ctx.exception))
        self.assertIn("collect_data.py", str(ctx.exception))

    def test_a_missing_label_column_is_also_caught(self):
        import os

        path = self.write(
            [{"a": 1.0, "b": 2.0}], os.path.join(self.dir, "nolabel.csv")
        )
        with self.assertRaises(ValueError) as ctx:
            dataset_io.load_training_csv(path, self.COLS, "collect_data.py")
        self.assertIn("gesture", str(ctx.exception))

    def test_features_come_back_in_the_order_requested(self):
        """Column order in the file must not decide feature order -- the model
        was trained against the config's ordering."""
        import os

        path = self.write(
            [{"gesture": "A", "b": 99.0, "a": 1.0}], os.path.join(self.dir, "ord.csv")
        )
        X, _, _, _ = dataset_io.load_training_csv(path, self.COLS, "collect_data.py")
        self.assertEqual(list(X[0]), [1.0, 99.0])


# --------------------------------------------------------------------------
class TestSharedCandidates(unittest.TestCase):
    """Both trainers build their model line-up from one place."""

    def test_forest_size_is_the_argument(self):
        c25 = evaluation.build_candidates(25, 42)
        c99 = evaluation.build_candidates(99, 42)
        self.assertEqual(c25["Random Forest (primary)"].n_estimators, 25)
        self.assertEqual(c99["Random Forest (primary)"].n_estimators, 99)

    def test_the_same_three_algorithms_for_both_trainers(self):
        names = set(evaluation.build_candidates(50, 0))
        self.assertEqual(len(names), 3)
        self.assertTrue(any(n.startswith("Random Forest") for n in names))

    def test_both_trainers_use_their_configured_tree_count(self):
        import src.train_dynamic_model as tdm_mod
        import src.train_model as tm_mod

        self.assertIsNot(config.STATIC_FOREST_TREES, None)
        self.assertIsNot(config.DYNAMIC_FOREST_TREES, None)
        # both call the shared builder rather than defining their own
        self.assertFalse(hasattr(tm_mod, "build_candidates"))
        self.assertFalse(hasattr(tdm_mod, "build_candidates"))


# --------------------------------------------------------------------------
class TestEvaluationSplit(unittest.TestCase):
    def test_temporal_split_keeps_every_class_on_both_sides(self):
        y = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
        ts = np.arange(150, dtype=float)
        tr, te = evaluation.temporal_split_indices(y, ts, 0.2)
        for side in (tr, te):
            self.assertEqual(set(np.unique(y[side])), {"A", "B", "C"})

    def test_temporal_split_puts_the_latest_samples_in_the_test_set(self):
        y = np.array(["A"] * 10)
        ts = np.arange(10, dtype=float)
        tr, te = evaluation.temporal_split_indices(y, ts, 0.2)
        self.assertLess(ts[tr].max(), ts[te].min())

    def test_temporal_split_is_not_fooled_by_row_order(self):
        """Rows may not be stored in capture order; the split must follow the
        timestamps, not the file."""
        y = np.array(["A"] * 10)
        ts = np.array([9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        tr, te = evaluation.temporal_split_indices(y, ts, 0.2)
        self.assertLess(ts[tr].max(), ts[te].min())

    def test_train_and_test_never_overlap(self):
        y = np.array(["A"] * 30 + ["B"] * 30)
        ts = np.arange(60, dtype=float)
        tr, te = evaluation.temporal_split_indices(y, ts, 0.2)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(len(tr) + len(te), 60)

    def test_tiny_class_still_lands_on_both_sides(self):
        y = np.array(["A"] * 20 + ["B", "B"])
        ts = np.arange(22, dtype=float)
        tr, te = evaluation.temporal_split_indices(y, ts, 0.2)
        self.assertIn("B", y[tr])
        self.assertIn("B", y[te])

    def test_random_split_is_stratified_and_complete(self):
        y = np.array(["A"] * 50 + ["B"] * 50)
        tr, te = evaluation.random_split_indices(y, 0.2, 42)
        self.assertEqual(len(tr) + len(te), 100)
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(int((y[te] == "A").sum()), 10)


# --------------------------------------------------------------------------
class TestActionAccuracy(unittest.TestCase):
    def test_confusing_two_actionless_poses_is_not_an_action_error(self):
        y_true = np.array([config.GESTURE_NEUTRAL] * 10)
        y_pred = np.array([config.GESTURE_OPEN_PALM] * 10)
        self.assertEqual(
            evaluation.action_accuracy(y_true, y_pred, config.GESTURE_ACTION_MAP), 1.0
        )

    def test_confusing_two_commands_is_an_action_error(self):
        y_true = np.array([config.GESTURE_FIST] * 10)
        y_pred = np.array([config.GESTURE_THUMBS_UP] * 10)
        self.assertEqual(
            evaluation.action_accuracy(y_true, y_pred, config.GESTURE_ACTION_MAP), 0.0
        )

    def test_missing_a_command_is_an_action_error(self):
        y_true = np.array([config.GESTURE_FIST])
        y_pred = np.array([config.GESTURE_NEUTRAL])
        self.assertEqual(
            evaluation.action_accuracy(y_true, y_pred, config.GESTURE_ACTION_MAP), 0.0
        )

    def test_action_accuracy_is_never_below_label_accuracy(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            y_true = rng.choice(config.STATIC_GESTURES, 50)
            y_pred = rng.choice(config.STATIC_GESTURES, 50)
            label = float(np.mean(y_true == y_pred))
            action = evaluation.action_accuracy(y_true, y_pred, config.GESTURE_ACTION_MAP)
            self.assertGreaterEqual(action + 1e-12, label)

    def test_merged_groups_names_the_actionless_poses(self):
        groups = evaluation.merged_label_groups(
            config.STATIC_GESTURES, config.GESTURE_ACTION_MAP
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            set(groups[0]), {config.GESTURE_NEUTRAL, config.GESTURE_OPEN_PALM}
        )

    def test_swipe_classes_share_no_action(self):
        self.assertEqual(
            evaluation.merged_label_groups(
                config.DYNAMIC_GESTURES, config.DYNAMIC_ACTION_MAP
            ),
            [],
        )


# --------------------------------------------------------------------------
class TestStableGestureTracker(unittest.TestCase):
    """The debounce that decides whether a held pose fires its action."""

    def setUp(self):
        self.tracker = infer_realtime.StableGestureTracker(verbose=False)

    def feed(self, labels):
        return [self.tracker.update(l) for l in labels]

    def test_a_steadily_held_pose_fires(self):
        fired = self.feed([config.GESTURE_THUMBS_UP] * config.STABLE_FRAMES_REQUIRED)
        self.assertEqual(fired[-1], config.GESTURE_THUMBS_UP)

    def test_it_fires_only_once_per_hold(self):
        fired = self.feed([config.GESTURE_THUMBS_UP] * 12)
        self.assertEqual([f for f in fired if f], [config.GESTURE_THUMBS_UP])

    def test_a_brief_flicker_does_not_fire(self):
        fired = self.feed(
            [config.GESTURE_NEUTRAL] * 8
            + [config.GESTURE_THUMBS_UP] * (config.STABLE_FRAMES_REQUIRED - 2)
        )
        self.assertTrue(all(f is None for f in fired))
        self.assertEqual(self.tracker.last_block_reason, "not steady")

    def test_actionless_poses_never_fire(self):
        for pose in (config.GESTURE_NEUTRAL, config.GESTURE_OPEN_PALM, config.GESTURE_NONE):
            t = infer_realtime.StableGestureTracker(verbose=False)
            self.assertTrue(all(f is None for f in [t.update(pose) for _ in range(12)]))

    def test_idle_frames_cannot_outvote_a_freshly_raised_pose(self):
        """The bug this guards against: a buffer still full of NEUTRAL from
        before the hand went up would win the majority vote and hide a pose
        that had genuinely been held long enough."""
        fired = self.feed(
            [config.GESTURE_NEUTRAL] * 6 + [config.GESTURE_THUMBS_UP] * config.STABLE_FRAMES_REQUIRED
        )
        self.assertEqual(fired[-1], config.GESTURE_THUMBS_UP)

    def test_releasing_the_pose_rearms_it(self):
        self.feed([config.GESTURE_THUMBS_UP] * config.STABLE_FRAMES_REQUIRED)
        self.tracker.last_action_time = 0.0  # skip the wall-clock cooldown
        self.feed([config.GESTURE_NEUTRAL] * config.PREDICTION_HISTORY_LENGTH)
        self.assertIsNone(self.tracker.locked_gesture)
        fired = self.feed([config.GESTURE_THUMBS_UP] * config.STABLE_FRAMES_REQUIRED)
        self.assertEqual(fired[-1], config.GESTURE_THUMBS_UP)

    def test_cooldown_blocks_a_second_action_immediately_after_the_first(self):
        self.feed([config.GESTURE_FIST] * config.STABLE_FRAMES_REQUIRED)
        fired = self.feed([config.GESTURE_POINTING] * config.PREDICTION_HISTORY_LENGTH)
        self.assertTrue(all(f is None for f in fired))
        self.assertEqual(self.tracker.last_block_reason, "cooldown")


# --------------------------------------------------------------------------
class TestSwipeFirer(unittest.TestCase):
    """The gates between "the model called it a swipe" and "the slide moves"."""

    def setUp(self):
        self.firer = infer_realtime.SwipeFirer(verbose=False)

    def test_no_swipe_never_fires(self):
        self.assertIsNone(self.firer.maybe_fire(config.NO_SWIPE, 0.9, 0.0))

    def test_a_real_swipe_fires(self):
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, 0.4, 0.0), config.SWIPE_RIGHT
        )

    def test_travel_below_the_floor_is_blocked(self):
        self.assertIsNone(
            self.firer.maybe_fire(config.SWIPE_RIGHT, config.MIN_SWIPE_DX / 2, 0.0)
        )
        self.assertEqual(self.firer.last_block_reason, "too small")

    def test_the_floor_is_direction_blind(self):
        self.assertIsNone(
            self.firer.maybe_fire(config.SWIPE_LEFT, -config.MIN_SWIPE_DX / 2, 0.0)
        )

    def test_exactly_at_the_floor_is_allowed(self):
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, config.MIN_SWIPE_DX, 0.0),
            config.SWIPE_RIGHT,
        )

    def test_moving_a_held_command_pose_is_not_a_swipe(self):
        """The swipe classifier sees only WHERE the hand is, never its shape,
        so waving a fist around would otherwise change slides."""
        self.assertIsNone(
            self.firer.maybe_fire(
                config.SWIPE_RIGHT, 0.5, config.SWIPE_BLOCKING_POSE_FRACTION
            )
        )
        self.assertEqual(self.firer.last_block_reason, "command pose held")

    def test_a_pose_glimpsed_briefly_does_not_block(self):
        self.assertEqual(
            self.firer.maybe_fire(
                config.SWIPE_RIGHT, 0.5, config.SWIPE_BLOCKING_POSE_FRACTION - 0.01
            ),
            config.SWIPE_RIGHT,
        )

    def test_moving_into_a_pose_is_not_a_swipe(self):
        """Raising your hand into the pointing pose sweeps it sideways, and
        that sweep is real travel. The window reads as a swipe that happens to
        end in a pose -- which is exactly backwards."""
        self.assertIsNone(
            self.firer.maybe_fire(
                config.SWIPE_RIGHT,
                travelled_dx=0.4,
                command_pose_fraction=0.2,   # under the fraction gate
                recent_command_poses=config.SWIPE_BLOCKING_RECENT_POSES,
            )
        )
        self.assertEqual(self.firer.last_block_reason, "moving into a pose")

    def test_one_blurred_frame_misread_as_a_pose_does_not_block(self):
        """Motion blur mid-swipe produces occasional wrong labels. A single one
        must not cost the slide change."""
        self.assertEqual(
            self.firer.maybe_fire(
                config.SWIPE_RIGHT,
                travelled_dx=0.4,
                command_pose_fraction=0.1,
                recent_command_poses=1,
            ),
            config.SWIPE_RIGHT,
        )

    def test_a_clean_swipe_with_no_pose_in_the_tail_still_fires(self):
        self.assertEqual(
            self.firer.maybe_fire(
                config.SWIPE_LEFT,
                travelled_dx=-0.5,
                command_pose_fraction=0.0,
                recent_command_poses=0,
            ),
            config.SWIPE_LEFT,
        )

    def test_the_tail_gate_is_off_when_no_pose_info_is_given(self):
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, 0.4, 0.0, None),
            config.SWIPE_RIGHT,
        )

    def test_the_tail_gate_cannot_be_stricter_than_the_tail_itself(self):
        """A threshold above the window length could never be reached, silently
        disabling the gate."""
        self.assertLessEqual(
            config.SWIPE_BLOCKING_RECENT_POSES, config.SWIPE_BLOCKING_RECENT_FRAMES
        )

    def test_a_pose_held_moments_ago_still_blocks(self):
        """The regression this guards: moving your hand while pointing blurs
        it, the classifier honestly reports NEUTRAL/NONE, and the pose leaves
        the frame history exactly as the swipe window fills. Both
        frame-counting gates then see nothing. Memory doesn't depend on the
        current frame being readable."""
        memory = infer_realtime.CommandPoseMemory()
        for _ in range(config.SWIPE_POSE_MEMORY_ARM_SIGHTINGS):
            memory.update(config.GESTURE_POINTING)
        # Blur: the pose is gone from every per-frame signal.
        for _ in range(config.SWIPE_WINDOW_FRAMES):
            memory.update(config.GESTURE_NONE)

        self.assertIsNone(
            self.firer.maybe_fire(
                config.SWIPE_RIGHT,
                travelled_dx=0.35,
                command_pose_fraction=0.0,   # both frame gates see nothing
                recent_command_poses=0,
                pose_memory=memory,
            )
        )
        self.assertEqual(self.firer.last_block_reason, "pose held moments ago")

    def test_one_misread_frame_does_not_arm_the_memory(self):
        """A single blurred frame read as a pose during a genuine swipe must
        not lock swiping out for a whole second."""
        memory = infer_realtime.CommandPoseMemory()
        memory.update(config.GESTURE_POINTING)
        self.assertFalse(memory.is_blocking)
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, 0.35, 0.0, 0, pose_memory=memory),
            config.SWIPE_RIGHT,
        )

    def test_the_memory_expires(self):
        memory = infer_realtime.CommandPoseMemory()
        for _ in range(config.SWIPE_POSE_MEMORY_ARM_SIGHTINGS):
            memory.update(config.GESTURE_FIST)
        self.assertTrue(memory.is_blocking)
        # Rewind the clock past the hold window rather than sleeping.
        memory._last_confirmed -= config.SWIPE_POSE_MEMORY_SECONDS + 0.01
        self.assertFalse(memory.is_blocking)

    def test_a_fresh_memory_never_blocks(self):
        memory = infer_realtime.CommandPoseMemory()
        self.assertFalse(memory.is_blocking)
        self.assertEqual(memory.seconds_since_confirmed, float("inf"))
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, 0.35, 0.0, 0, pose_memory=memory),
            config.SWIPE_RIGHT,
        )

    def test_actionless_poses_never_arm_the_memory(self):
        """OPEN_PALM is the swiping hand shape -- if it armed the memory,
        swiping would block itself."""
        memory = infer_realtime.CommandPoseMemory()
        for _ in range(config.SWIPE_POSE_MEMORY_ARM_FRAMES * 2):
            memory.update(config.GESTURE_OPEN_PALM)
        self.assertFalse(memory.is_blocking)

    def test_arming_needs_fewer_sightings_than_the_window_holds(self):
        self.assertLessEqual(
            config.SWIPE_POSE_MEMORY_ARM_SIGHTINGS,
            config.SWIPE_POSE_MEMORY_ARM_FRAMES,
        )

    def test_cooldown_collapses_one_physical_swipe_into_one_slide_change(self):
        """A single swipe spans several overlapping windows; without this the
        slide would jump several times."""
        self.assertEqual(
            self.firer.maybe_fire(config.SWIPE_RIGHT, 0.5, 0.0), config.SWIPE_RIGHT
        )
        self.assertIsNone(self.firer.maybe_fire(config.SWIPE_RIGHT, 0.5, 0.0))


# --------------------------------------------------------------------------
class TestActionDispatch(unittest.TestCase):
    def test_web_target_avoids_the_shortcuts_a_browser_steals(self):
        web = config.ACTION_KEY_MAPS["web"]
        self.assertIsNone(web[config.ACTION_LASER_POINTER])   # Ctrl+L = address bar
        self.assertNotEqual(web[config.ACTION_START_PRESENTATION], ("press", "f5"))

    def test_dispatch_reports_an_action_unavailable_on_this_target(self):
        out = infer_realtime.dispatch_action(
            config.GESTURE_POINTING, config.GESTURE_ACTION_MAP, dry_run=True,
            key_map=config.ACTION_KEY_MAPS["web"], target="web",
        )
        self.assertIn("not available", out)

    def test_the_same_gesture_works_on_the_desktop_target(self):
        out = infer_realtime.dispatch_action(
            config.GESTURE_POINTING, config.GESTURE_ACTION_MAP, dry_run=True,
            key_map=config.ACTION_KEY_MAPS["desktop"], target="desktop",
        )
        self.assertIn("laser_pointer", out)
        self.assertNotIn("not available", out)

    def test_both_targets_can_be_used_in_one_process(self):
        """dispatch_action takes its key map as an argument rather than
        reading a module global that startup mutates."""
        web = infer_realtime.dispatch_action(
            config.GESTURE_THUMBS_UP, config.GESTURE_ACTION_MAP, True,
            config.ACTION_KEY_MAPS["web"], "web")
        desktop = infer_realtime.dispatch_action(
            config.GESTURE_THUMBS_UP, config.GESTURE_ACTION_MAP, True,
            config.ACTION_KEY_MAPS["desktop"], "desktop")
        self.assertNotEqual(web, desktop)

    def test_an_unmapped_gesture_does_nothing(self):
        out = infer_realtime.dispatch_action(
            config.GESTURE_NEUTRAL, config.GESTURE_ACTION_MAP, dry_run=True
        )
        self.assertIn("no action mapped", out)

    def _all_keys_the_program_can_send(self):
        keys = set()
        specs = []
        for key_map in config.ACTION_KEY_MAPS.values():
            specs.extend(key_map.values())
        for mode in config.POINTER_MODE_KEYS.values():
            if mode:
                specs.extend(mode.values())
        for spec in specs:
            if spec is None:
                continue
            k = spec[1]
            keys.update(k if isinstance(k, tuple) else (k,))
        return {str(k).lower() for k in keys}

    def test_no_key_the_program_sends_can_also_quit_it(self):
        """The regression this guards: FIST sends Esc. Run from an editor's
        built-in terminal and that Esc lands right back in this console, so
        Esc-as-quit-key meant ending a slideshow killed the program. Any key
        the program can send is disqualified as a console quit key."""
        import inspect

        source = inspect.getsource(infer_realtime.quit_key_pressed)
        accepted = set(re.findall(r'b"(\\x1b|.)"', source))
        sendable = self._all_keys_the_program_can_send()

        collisions = {
            k for k in accepted
            if k.lower() in sendable or (k == "\\x1b" and "esc" in sendable)
        }
        self.assertEqual(
            collisions, set(),
            f"quit key(s) {collisions} are also sent by an action: {sorted(sendable)}",
        )

    def test_the_console_quit_key_is_q(self):
        import inspect

        source = inspect.getsource(infer_realtime.quit_key_pressed)
        self.assertIn('b"q"', source)
        self.assertNotIn('\\x1b', source.split('"""')[-1])  # not in the code body

    def test_the_console_buffer_is_drained_after_every_key_send(self):
        """Otherwise the program's own injected keys queue up in its console."""
        import inspect

        for fn in (infer_realtime.dispatch_action, infer_realtime.send_key_spec):
            src = inspect.getsource(fn)
            if "press_keys" in src:
                self.assertIn(
                    "drain_console_input()", src,
                    f"{fn.__name__} sends keys without draining the console",
                )

    def test_draining_is_safe_off_windows(self):
        infer_realtime.drain_console_input()      # must not raise
        self.assertFalse(infer_realtime.quit_key_pressed())

    def test_desktop_is_the_default_target(self):
        """The installed PowerPoint app, where the whole shortcut set works."""
        self.assertEqual(config.PRESENTATION_TARGET, "desktop")

    def test_desktop_has_a_working_laser_but_web_does_not(self):
        self.assertIsNotNone(
            config.ACTION_KEY_MAPS["desktop"][config.ACTION_LASER_POINTER]
        )
        self.assertIsNone(config.ACTION_KEY_MAPS["web"][config.ACTION_LASER_POINTER])

    def test_every_target_has_a_pointer_mode_entry(self):
        """None is a valid value (no laser mode); a MISSING key is a bug."""
        for target in config.ACTION_KEY_MAPS:
            self.assertIn(target, config.POINTER_MODE_KEYS)

    def test_desktop_pointer_mode_enters_and_exits(self):
        keys = config.POINTER_MODE_KEYS["desktop"]
        self.assertIn("enter", keys)
        self.assertIn("exit", keys)
        self.assertNotEqual(keys["enter"], keys["exit"])

    def test_the_laser_exit_key_is_not_escape(self):
        """Esc ends the slideshow. Using it to leave laser mode would close the
        presentation every time the pointing pose was lowered."""
        for target, keys in config.POINTER_MODE_KEYS.items():
            if not keys:
                continue
            for spec in keys.values():
                flat = spec[1] if isinstance(spec[1], tuple) else (spec[1],)
                self.assertNotIn("esc", [str(k).lower() for k in flat])

    def test_send_key_spec_does_nothing_on_a_dry_run_or_a_none_spec(self):
        """Must not import or touch pyautogui in either case."""
        infer_realtime.send_key_spec(None, dry_run=False)
        infer_realtime.send_key_spec(("hotkey", ("ctrl", "l")), dry_run=True)

    def test_present_is_exactly_the_three_flags_it_stands_for(self):
        """--present is shorthand, not a separate mode. If the two ever drift,
        the short form silently stops doing what it documents."""
        import contextlib
        import io
        import sys

        calls = []
        original_run = infer_realtime.run
        infer_realtime.run = lambda **kw: calls.append(kw)
        original_argv = sys.argv
        try:
            for argv in (["--present"], ["--pointer", "--no-preview", "--quiet"]):
                sys.argv = ["prog"] + argv
                with contextlib.redirect_stdout(io.StringIO()):
                    infer_realtime.main()
        finally:
            infer_realtime.run = original_run
            sys.argv = original_argv

        self.assertEqual(calls[0], calls[1])
        self.assertFalse(calls[0]["show_preview"])
        self.assertFalse(calls[0]["verbose"])
        self.assertTrue(calls[0]["use_pointer"])

    def test_without_present_nothing_is_turned_on(self):
        import contextlib
        import io
        import sys

        calls = []
        original_run = infer_realtime.run
        infer_realtime.run = lambda **kw: calls.append(kw)
        original_argv = sys.argv
        try:
            sys.argv = ["prog"]
            with contextlib.redirect_stdout(io.StringIO()):
                infer_realtime.main()
        finally:
            infer_realtime.run = original_run
            sys.argv = original_argv

        self.assertTrue(calls[0]["show_preview"])
        self.assertTrue(calls[0]["verbose"])
        self.assertFalse(calls[0]["use_pointer"])

    def test_dry_run_is_labelled(self):
        out = infer_realtime.dispatch_action(
            config.SWIPE_RIGHT, config.DYNAMIC_ACTION_MAP, dry_run=True
        )
        self.assertIn("DRY RUN", out)


# --------------------------------------------------------------------------
class TestPointerMapping(unittest.TestCase):
    """Fingertip position -> screen pixels."""

    SCREEN = (1920, 1080)
    FULL = (0.0, 0.0, 1.0, 1.0)

    def map(self, nx, ny, region=None):
        return pointer.map_to_screen(nx, ny, *self.SCREEN, region or self.FULL)

    def test_centre_of_frame_is_centre_of_screen(self):
        x, y = self.map(0.5, 0.5)
        self.assertAlmostEqual(x, 1919 // 2, delta=1)
        self.assertAlmostEqual(y, 1079 // 2, delta=1)

    def test_corners_map_to_corners(self):
        self.assertEqual(self.map(0.0, 0.0), (0, 0))
        self.assertEqual(self.map(1.0, 1.0), (1919, 1079))

    def test_no_horizontal_mirroring(self):
        """The frame is already flipped before detection, so moving right must
        increase the screen x. Flipping again here would invert the cursor."""
        left, _ = self.map(0.25, 0.5)
        right, _ = self.map(0.75, 0.5)
        self.assertLess(left, right)

    def test_active_region_is_stretched_to_the_whole_screen(self):
        region = (0.2, 0.2, 0.8, 0.8)
        self.assertEqual(pointer.map_to_screen(0.2, 0.2, *self.SCREEN, region), (0, 0))
        self.assertEqual(pointer.map_to_screen(0.8, 0.8, *self.SCREEN, region), (1919, 1079))
        mid = pointer.map_to_screen(0.5, 0.5, *self.SCREEN, region)
        self.assertAlmostEqual(mid[0], 1919 // 2, delta=2)

    def test_outside_the_region_clamps_to_the_edge(self):
        """The cursor must stop at the border, never wrap or run off-screen."""
        region = (0.2, 0.2, 0.8, 0.8)
        self.assertEqual(pointer.map_to_screen(0.0, 0.0, *self.SCREEN, region), (0, 0))
        self.assertEqual(pointer.map_to_screen(1.0, 1.0, *self.SCREEN, region), (1919, 1079))
        self.assertEqual(pointer.map_to_screen(-5.0, 9.0, *self.SCREEN, region), (0, 1079))

    def test_never_returns_an_off_screen_pixel(self):
        for nx in (-1.0, 0.0, 0.3, 0.5, 1.0, 2.0):
            for ny in (-1.0, 0.0, 0.7, 1.0, 2.0):
                x, y = self.map(nx, ny, config.POINTER_ACTIVE_REGION)
                self.assertTrue(0 <= x < self.SCREEN[0])
                self.assertTrue(0 <= y < self.SCREEN[1])

    def test_a_degenerate_region_does_not_divide_by_zero(self):
        x, y = pointer.map_to_screen(0.5, 0.5, *self.SCREEN, (0.4, 0.4, 0.4, 0.4))
        self.assertTrue(0 <= x < self.SCREEN[0] and 0 <= y < self.SCREEN[1])


# --------------------------------------------------------------------------
class TestAdaptiveSmoother(unittest.TestCase):
    def test_first_sample_passes_straight_through(self):
        """The cursor must start where the finger is, not ease in from a stale
        position left over from the last time pointing was used."""
        s = pointer.AdaptiveSmoother()
        self.assertEqual(s.update(500, 400), (500.0, 400.0))

    def test_output_lags_behind_a_jump(self):
        s = pointer.AdaptiveSmoother()
        s.update(0, 0)
        x, _ = s.update(1000, 0)
        self.assertGreater(x, 0)
        self.assertLess(x, 1000)

    def test_small_jitter_is_damped_more_than_fast_motion(self):
        """The whole point of adapting alpha: jitter while still is what you
        notice, lag while moving fast is what you notice. Not both."""
        jitter = pointer.AdaptiveSmoother()
        jitter.update(500, 500)
        jittered, _ = jitter.update(503, 500)
        jitter_retained = (jittered - 500) / 3.0

        fast = pointer.AdaptiveSmoother()
        fast.update(500, 500)
        moved, _ = fast.update(800, 500)
        fast_retained = (moved - 500) / 300.0

        self.assertLess(jitter_retained, fast_retained)

    def test_it_converges_when_the_hand_stops(self):
        s = pointer.AdaptiveSmoother()
        s.update(0, 0)
        for _ in range(80):
            x, y = s.update(600, 300)
        self.assertAlmostEqual(x, 600, delta=1.0)
        self.assertAlmostEqual(y, 300, delta=1.0)

    def test_output_never_overshoots_the_input(self):
        s = pointer.AdaptiveSmoother()
        s.update(100, 100)
        for target in (200, 500, 900):
            x, _ = s.update(target, 100)
            self.assertLessEqual(x, target)

    def test_reset_clears_the_history(self):
        s = pointer.AdaptiveSmoother()
        s.update(0, 0)
        s.reset()
        self.assertEqual(s.update(900, 900), (900.0, 900.0))


# --------------------------------------------------------------------------
class TestFingertipPointer(unittest.TestCase):
    """Activation hysteresis and cursor dispatch, with a fake mouse."""

    def build(self, **kw):
        self.moves = []
        return pointer.FingertipPointer(
            screen=(1000, 1000),
            mover=lambda x, y: self.moves.append((x, y)),
            min_move_px=0,
            **kw,
        )

    def hand(self, x=0.5, y=0.5):
        lms = [FakeLandmark(0.5, 0.5) for _ in range(21)]
        lms[config.POINTER_LANDMARK] = FakeLandmark(x, y)
        return lms

    def test_one_pointing_frame_does_not_grab_the_cursor(self):
        p = self.build()
        self.assertIsNone(p.update(self.hand(), config.GESTURE_POINTING))
        self.assertEqual(self.moves, [])

    def test_it_activates_after_enough_consecutive_frames(self):
        p = self.build()
        for _ in range(config.POINTER_ACTIVATE_FRAMES):
            p.update(self.hand(), config.GESTURE_POINTING)
        self.assertTrue(p.active)
        self.assertEqual(len(self.moves), 1)

    def test_a_different_pose_never_moves_the_cursor(self):
        p = self.build()
        for _ in range(20):
            p.update(self.hand(), config.GESTURE_FIST)
        self.assertFalse(p.active)
        self.assertEqual(self.moves, [])

    def test_a_single_dropped_frame_does_not_release_it(self):
        """Detection drops a frame regularly. Releasing on one would make the
        cursor stutter constantly."""
        p = self.build()
        for _ in range(10):
            p.update(self.hand(), config.GESTURE_POINTING)
        p.update(None, config.GESTURE_NONE)
        self.assertTrue(p.active)

    def test_sustained_loss_releases_it(self):
        p = self.build()
        for _ in range(10):
            p.update(self.hand(), config.GESTURE_POINTING)
        for _ in range(config.POINTER_RELEASE_FRAMES):
            p.update(None, config.GESTURE_NONE)
        self.assertFalse(p.active)

    def test_it_holds_position_on_a_dropped_frame_rather_than_jumping(self):
        p = self.build()
        for _ in range(10):
            p.update(self.hand(0.5, 0.5), config.GESTURE_POINTING)
        before = len(self.moves)
        self.assertIsNone(p.update(None, config.GESTURE_NONE))
        self.assertEqual(len(self.moves), before)

    def test_reactivating_jumps_to_the_finger_instead_of_sliding(self):
        """Stale smoothing state across a release would drag the cursor across
        the screen from wherever it was last time."""
        p = self.build()
        for _ in range(10):
            p.update(self.hand(0.2, 0.2), config.GESTURE_POINTING)
        for _ in range(config.POINTER_RELEASE_FRAMES):
            p.update(None, config.GESTURE_NONE)
        self.moves.clear()
        for _ in range(config.POINTER_ACTIVATE_FRAMES):
            p.update(self.hand(0.8, 0.8), config.GESTURE_POINTING)
        expected = pointer.map_to_screen(0.8, 0.8, 1000, 1000, config.POINTER_ACTIVE_REGION)
        self.assertEqual(self.moves[0], expected)

    def test_activation_always_starts_from_a_clean_smoother(self):
        """Guards an invariant rather than a currently-reachable bug: the
        release path already clears the smoothing state, so the reset on
        activation is defensive. It stops mattering only until someone adds a
        second way to deactivate -- at which point a stale position would drag
        the cursor across the screen on the next point. Forcing the state
        directly is the only way to cover that."""
        p = self.build()
        for _ in range(10):
            p.update(self.hand(0.2, 0.2), config.GESTURE_POINTING)
        p.active = False  # deactivate WITHOUT going through the release path
        self.moves.clear()
        for _ in range(config.POINTER_ACTIVATE_FRAMES):
            p.update(self.hand(0.8, 0.8), config.GESTURE_POINTING)
        expected = pointer.map_to_screen(
            0.8, 0.8, 1000, 1000, config.POINTER_ACTIVE_REGION
        )
        self.assertEqual(self.moves[0], expected)

    def test_moving_the_finger_moves_the_cursor_the_same_way(self):
        p = self.build()
        for _ in range(10):
            p.update(self.hand(0.3, 0.5), config.GESTURE_POINTING)
        left_x = self.moves[-1][0]
        for _ in range(40):
            p.update(self.hand(0.7, 0.5), config.GESTURE_POINTING)
        self.assertGreater(self.moves[-1][0], left_x)

    def test_sub_threshold_moves_are_skipped(self):
        p = pointer.FingertipPointer(
            screen=(1000, 1000),
            mover=lambda x, y: self.moves.append((x, y)),
            min_move_px=50,
        )
        self.moves = []
        for _ in range(10):
            p.update(self.hand(0.5, 0.5), config.GESTURE_POINTING)
        self.assertEqual(len(self.moves), 1)

    def test_a_failing_mouse_call_never_ends_the_session(self):
        def explode(x, y):
            raise RuntimeError("display went away")

        p = pointer.FingertipPointer(screen=(1000, 1000), mover=explode)
        for _ in range(10):
            self.assertIsNone(p.update(self.hand(), config.GESTURE_POINTING))

    def test_pointing_still_blocks_swipes(self):
        """Aiming the cursor means sweeping the hand across the frame. That
        must not register as a swipe, so POINTING has to stay a command pose
        even though the pointer stops it firing a key."""
        self.assertIn(config.POINTER_POSE, config.GESTURE_ACTION_MAP)


# --------------------------------------------------------------------------
class TestConfigContract(unittest.TestCase):
    """Guards against a config edit silently breaking another module."""

    def test_every_static_gesture_has_a_description(self):
        for g in config.STATIC_GESTURES:
            self.assertIn(g, config.GESTURE_DESCRIPTIONS)

    def test_every_dynamic_gesture_has_a_description(self):
        for g in config.DYNAMIC_GESTURES:
            self.assertIn(g, config.DYNAMIC_GESTURE_DESCRIPTIONS)

    def test_every_mapped_gesture_is_a_real_gesture(self):
        for g in config.GESTURE_ACTION_MAP:
            self.assertIn(g, config.STATIC_GESTURES)
        for g in config.DYNAMIC_ACTION_MAP:
            self.assertIn(g, config.DYNAMIC_GESTURES)

    def test_every_target_maps_every_action(self):
        actions = set(config.GESTURE_ACTION_MAP.values()) | set(config.DYNAMIC_ACTION_MAP.values())
        for target, key_map in config.ACTION_KEY_MAPS.items():
            for action in actions:
                self.assertIn(action, key_map, f"{target} has no entry for {action}")

    def test_open_palm_triggers_nothing(self):
        """It is the swiping hand shape -- an action here would fire on every
        swipe."""
        self.assertNotIn(config.GESTURE_OPEN_PALM, config.GESTURE_ACTION_MAP)

    def test_feature_name_list_matches_the_extractor(self):
        self.assertEqual(
            len(config.DYNAMIC_FEATURE_NAMES), config.DYNAMIC_FEATURE_VECTOR_LENGTH
        )
        self.assertEqual(
            len(extract_features(straight_swipe(0.1, 0.8))),
            len(config.DYNAMIC_FEATURE_NAMES),
        )

    def test_the_runtime_floor_is_looser_than_the_training_floor(self):
        """Training drops clips below MIN_TRAINING_SWIPE_DX; the runtime floor
        must not be stricter, or valid swipes would be trained for and then
        rejected."""
        self.assertLessEqual(config.MIN_SWIPE_DX, config.MIN_TRAINING_SWIPE_DX)

    def test_a_pose_cannot_need_more_frames_than_the_history_holds(self):
        self.assertLessEqual(
            config.STABLE_FRAMES_REQUIRED, config.PREDICTION_HISTORY_LENGTH
        )

    def test_a_false_swipe_is_held_to_a_higher_bar_than_a_pose(self):
        self.assertGreaterEqual(
            config.MIN_DYNAMIC_PREDICTION_CONFIDENCE, config.MIN_PREDICTION_CONFIDENCE
        )


if __name__ == "__main__":
    unittest.main()
