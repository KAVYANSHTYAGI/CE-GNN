"""dvs128_to_frames_direct.py -- direct-read variant of dvs128_to_frames.py.

The original script parses raw multi-gesture .aedat recordings + labels.csv
via aermanager and segments them into per-gesture trials itself. Our raw
data is already segmented into per-trial .npy files (ibmGestureTrain/Test,
columns [x,y,p,t_seconds]), so we read those directly and call the same
events_to_frames.process_event_stream() the original script uses -- no
aermanager dependency, no trial-segmentation logic needed.

Rate augmentation: process_event_stream uses an ABSOLUTE time-chunking
window (chunk_len_us), not a per-sample-normalized one, so rescaling a
trial's timestamps by 1/rho BEFORE chunking is a genuine, faithful T_rho(E)
-- it actually changes which events land in which chunk/frame, unlike the
scale-invariant traps hit in the AEGNN/EventMamba integrations. We build:
  - natural/{train,test}          (rho=1.0)
  - rho{r}/train for r in TRAIN_RHOS \ {1.0}   (stage-2 curriculum pool)
  - rho{r}/test  for r in EVAL_RHOS \ {1.0}    (synthetic-rate eval grid)
"""
import numpy as np
import glob
import os
import pickle
import argparse
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import events_to_frames

HEIGHT = WIDTH = 128
K = 3
CHUNK_LEN_MS = 24
CHUNK_LEN_US = CHUNK_LEN_MS * 1000
MIN_TIME = CHUNK_LEN_US / K
MAX_TIME = 256 * 1000  # matches all_params.json's MT

TRAIN_RHOS = [0.5, 1.0, 2.0]
EVAL_RHOS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# BUGFIX: each raw per-user-per-class .npy file is an entire continuous
# multi-minute recording, NOT a single pre-trimmed gesture instance (the
# reference dvs128_to_frames.py explicitly segments these into short trials
# via labels.csv before ever calling process_event_stream). Feeding a whole
# recording through process_event_stream created ~236,000 time-chunks for
# "one sample" instead of the ~55 chunks a short clip should produce. We
# reproduce the missing segmentation step with a sliding window matching
# EvT+'s own expected clip length (clip_length_ms=1298 in all_params.json),
# using the same 250ms step already established for the EventMamba pipeline.
CLIP_LEN_MS = 1298
STEP_MS = 250


def slice_into_clips(t_us, clip_len_us, step_us):
    """Returns list of (start_idx, end_idx) into a sorted t_us array,
    covering [t_us[0], t_us[-1]] with sliding windows of clip_len_us,
    stepping by step_us. Mirrors uti.get_window_index's semantics but
    self-contained (no cross-repo import)."""
    t0, t_end = t_us[0], t_us[-1]
    starts = np.arange(t0, max(t_end - clip_len_us, t0) + 1, step_us)
    slices = []
    for s in starts:
        e = s + clip_len_us
        i0 = np.searchsorted(t_us, s, side='left')
        i1 = np.searchsorted(t_us, e, side='right')
        if i1 - i0 > 100:
            slices.append((i0, i1))
    return slices


def iter_trials(split_dir):
    """Yields (total_events[x,y,t_us,p], label, lograte_natural) for every
    short clip sliced out of each continuous per-user-per-class recording
    under split_dir. label 10 excluded, matching the other two
    architectures' convention for this same raw data."""
    user_dirs = sorted(glob.glob(os.path.join(split_dir, '*')))
    for ud in user_dirs:
        for npy_path in sorted(glob.glob(os.path.join(ud, '*.npy')),
                                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
            label = int(os.path.splitext(os.path.basename(npy_path))[0])
            if label == 10:
                continue
            arr = np.load(npy_path, allow_pickle=True)
            # BUGFIX: events_to_frames.py indexes arrays directly with the
            # x,y columns ("new_pos[unique_coords_pos[:,1], ...] = ...") and
            # raises IndexError if the array holding them is float dtype --
            # even if the values happen to be integer-valued. The reference
            # dvs128_to_frames.py builds its whole total_events array as a
            # single int array (from aermanager's parsed x/y/t/p, all ints,
            # t in integer microseconds), so we replicate that: round t to
            # the nearest microsecond and keep the WHOLE array int64.
            x, y = arr[:, 0], arr[:, 1]
            p = arr[:, 2]
            t_us = np.round(arr[:, 3] * 1e6)
            full_events = np.stack([x, y, t_us, p], axis=-1).astype(np.int64)

            for i0, i1 in slice_into_clips(t_us, CLIP_LEN_MS * 1000, STEP_MS * 1000):
                clip = full_events[i0:i1]
                clip_t = clip[:, 2]
                n = len(clip_t)
                duration_s = max((clip_t.max() - clip_t.min()) / 1e6, 1e-6)
                lograte = float(np.log(n / duration_s))
                yield clip, label, lograte


def write_sample(dst_dir, sample_id, label, frames, min_max_values, lograte):
    sample_dir = os.path.join(dst_dir, f'{sample_id:05d}_{label}')
    os.makedirs(sample_dir, exist_ok=True)
    for t in range(frames.shape[0]):
        with open(os.path.join(sample_dir, f'{t:04d}.pckl'), 'wb') as f:
            pickle.dump((frames[t].astype(np.float16), min_max_values[t]), f)
    with open(os.path.join(sample_dir, 'lograte.txt'), 'w') as f:
        f.write(str(lograte))


def process_split(split_dir, dst_root, rho, split_name):
    os.makedirs(dst_root, exist_ok=True)
    n_written, n_skipped = 0, 0
    n_errors_printed = 0
    for sample_id, (total_events, label, lograte_natural) in enumerate(
            tqdm(list(iter_trials(split_dir)), desc=f'{split_name} rho={rho}')):
        events = total_events.copy()
        if rho != 1.0:
            # keep int64 dtype (see iter_trials) -- round back after the
            # float division, or events_to_frames' coordinate indexing breaks
            events[:, 2] = np.round(events[:, 2] / rho).astype(np.int64)
        lograte = lograte_natural + float(np.log(rho))
        try:
            frames, min_max_values = events_to_frames.process_event_stream(
                events, HEIGHT, WIDTH, CHUNK_LEN_US, K, MIN_TIME, MAX_TIME)
        except Exception as e:
            # DEBUG: every trial was silently failing here (0 written, all
            # skipped) with the real cause hidden by a bare except. Print
            # the first few real exceptions so failures are diagnosable.
            if n_errors_printed < 5:
                import traceback
                print(f"[ERROR] sample {sample_id} rho={rho}: {e}")
                traceback.print_exc()
                n_errors_printed += 1
            n_skipped += 1
            continue
        if len(frames) == 0:
            n_skipped += 1
            continue
        write_sample(dst_root, sample_id, label, frames, min_max_values, lograte)
        n_written += 1
    print(f'{split_name} rho={rho}: wrote {n_written}, skipped {n_skipped}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', required=True, help='.../DVSGesture (contains ibmGestureTrain/ibmGestureTest)')
    ap.add_argument('--export_path', required=True)
    args = ap.parse_args()

    train_dir = os.path.join(args.root_dir, 'ibmGestureTrain')
    test_dir = os.path.join(args.root_dir, 'ibmGestureTest')

    process_split(train_dir, os.path.join(args.export_path, 'natural', 'train'), 1.0, 'train')
    process_split(test_dir, os.path.join(args.export_path, 'natural', 'test'), 1.0, 'test')

    for rho in TRAIN_RHOS:
        if rho == 1.0:
            continue
        process_split(train_dir, os.path.join(args.export_path, f'rho{rho}', 'train'), rho, 'train')

    for rho in EVAL_RHOS:
        if rho == 1.0:
            continue
        process_split(test_dir, os.path.join(args.export_path, f'rho{rho}', 'test'), rho, 'test')
