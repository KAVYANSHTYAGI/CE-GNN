######direct-read variant of generate_dvsgesture_rate.py######
# spikingjelly's DVS128Gesture loader refuses to run unless it finds the
# single official DvsGesture.tar.gz archive under <root>/download/. Our raw
# data is already extracted into the alternate ibmGestureTrain/ibmGestureTest
# per-trial-npy layout (each <user_folder>/<label>.npy is one recording,
# columns [x, y, polarity, t_seconds]), so we read it directly instead.
import numpy as np
import h5py
import os
import glob
import argparse
from tqdm import tqdm
import uti


def iter_trials(split_dir):
    """Yields (x, y, t_us, label) for every trial .npy under split_dir.
    label 10 (the 'other' gesture) is excluded, matching EventMamba's own
    convention in generate_dvsgesture.py."""
    user_dirs = sorted(glob.glob(os.path.join(split_dir, '*')))
    for ud in user_dirs:
        for npy_path in sorted(glob.glob(os.path.join(ud, '*.npy')),
                                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
            label = int(os.path.splitext(os.path.basename(npy_path))[0])
            if label == 10:
                continue
            arr = np.load(npy_path, allow_pickle=True).astype(np.float64)
            x, y, p, t_sec = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            t_us = (t_sec * 1e6).astype(np.int64)  # match generate_dvsgesture_rate.py's microsecond convention
            yield x.astype(np.int64), y.astype(np.int64), t_us, label


def read_split(split_dir, NUM_POINTS, STEP_SIZE, WINDOW_SIZE, train):
    data, labels, marks, lograte = [], [], [], []
    trial_idx = 0
    for x, y, t, label in tqdm(list(iter_trials(split_dir))):
        class_events = np.zeros(shape=(len(x), 3), dtype=np.int64)
        class_events[:, 0] = t
        class_events[:, 1] = x
        class_events[:, 2] = y
        win_start_index, win_end_index = uti.get_window_index(
            t, t[0], stepsize=STEP_SIZE * 1000000, windowsize=WINDOW_SIZE * 1000000)
        for n in range(len(win_start_index)):
            window_events = class_events[win_start_index[n]:win_end_index[n], :].copy()
            n_raw = window_events.shape[0]
            if n_raw > 100:
                rate = np.log(n_raw / WINDOW_SIZE)
                extracted = uti.shuffle_downsample(window_events, NUM_POINTS)
                if len(extracted[:, 0]) == NUM_POINTS:
                    events_normed = uti.normaliztion(extracted, 128, 128, False)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(trial_idx)
                    lograte.append(rate)
                if train:
                    crop_events = uti.random_crop(window_events, 128, 128)
                    # BUGFIX: random_crop can legitimately yield zero events
                    # (spatial/temporal crop excludes everything in this
                    # window); shuffle_downsample then crashes with
                    # "'a' cannot be empty unless no samples are taken".
                    # Skip this augmented sample in that case.
                    if crop_events.shape[0] == 0:
                        continue
                    extracted = uti.shuffle_downsample(crop_events, NUM_POINTS)
                    if len(extracted[:, 0]) == NUM_POINTS:
                        events_normed = uti.normaliztion(extracted, 128, 128, False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(trial_idx)
                        lograte.append(rate)
        trial_idx += 1
    return (np.array(data), np.array(labels), np.array(marks), np.array(lograte, dtype=np.float32))


def write_h5(path, data, label, mark, lograte, num_points):
    with h5py.File(path, 'a') as hf:
        hf.create_dataset('data', shape=data.shape, maxshape=(None, num_points, 3), chunks=True, dtype='float32')
        hf.create_dataset('label', shape=label.shape, maxshape=(None), chunks=True, dtype='int16')
        hf.create_dataset('mark', shape=mark.shape, maxshape=(None), chunks=True, dtype='int16')
        hf.create_dataset('lograte', shape=lograte.shape, maxshape=(None), chunks=True, dtype='float32')
        hf['data'][:] = data
        hf['label'][:] = label
        hf['mark'][:] = mark
        hf['lograte'][:] = lograte
    print(f"wrote {path}: data={data.shape} label={label.shape} "
          f"lograte range=[{lograte.min():.2f},{lograte.max():.2f}] classes={sorted(set(label.tolist()))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', required=True, help='.../DVSGesture (contains ibmGestureTrain/ibmGestureTest)')
    ap.add_argument('--export_path', required=True)
    ap.add_argument('--num_points', type=int, default=1024)
    ap.add_argument('--step_size', type=float, default=0.25)
    ap.add_argument('--natural_window', type=float, default=0.5)
    ap.add_argument('--aug_windows', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    args = ap.parse_args()

    import gc
    os.makedirs(args.export_path, exist_ok=True)
    train_dir = os.path.join(args.root_dir, 'ibmGestureTrain')
    test_dir = os.path.join(args.root_dir, 'ibmGestureTest')

    # Raw labels are the .npy filenames: {1,...,9,11} (10 excluded, 1-indexed,
    # non-contiguous). CrossEntropyLoss needs contiguous 0..K-1, so build one
    # remap once (from natural-train) and reuse/persist it across resumed runs.
    remap_path = os.path.join(args.export_path, 'label_remap.txt')
    if os.path.exists(remap_path):
        with open(remap_path) as f:
            remap = eval(f.read())
        print(f"reusing existing label remap: {remap}")
    else:
        d, l, m, r = read_split(train_dir, args.num_points, args.step_size, args.natural_window, train=True)
        classes = sorted(set(l.tolist()))
        remap = {c: i for i, c in enumerate(classes)}
        print(f"label remap ({len(classes)} classes): {remap}")
        with open(remap_path, 'w') as f:
            f.write(str(remap))
        del d, l, m, r
        gc.collect()

    def apply_remap(l):
        return np.array([remap[int(v)] for v in l], dtype=l.dtype)

    # BUGFIX: resumable -- skip any split whose .h5 already exists on disk,
    # so a re-submit after an OOM kill doesn't redo already-finished (and
    # expensive) splits. Previously the whole script was skipped wholesale
    # if train_natural.h5 existed, silently leaving later splits (e.g.
    # train_w1.0.h5) never generated after a partial-run crash.
    def maybe_generate(out_name, src_dir, window, train):
        out_path = os.path.join(args.export_path, out_name)
        if os.path.exists(out_path):
            print(f"skipping {out_path} (already exists)")
            return
        d, l, m, r = read_split(src_dir, args.num_points, args.step_size, window, train=train)
        l = apply_remap(l)
        write_h5(out_path, d, l, m, r, args.num_points)
        del d, l, m, r
        gc.collect()

    maybe_generate('train_natural.h5', train_dir, args.natural_window, True)
    maybe_generate('test_natural.h5', test_dir, args.natural_window, False)
    for w in args.aug_windows:
        maybe_generate(f'train_w{w}.h5', train_dir, w, True)
        maybe_generate(f'test_w{w}.h5', test_dir, w, False)
