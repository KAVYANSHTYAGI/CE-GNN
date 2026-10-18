######adapted from generate_dvsgesture.py to add per-sample event-rate metadata######
# For each windowed sample we additionally store log_lambda_raw = log(N_events_in_window /
# window_duration_seconds), computed from the ORIGINAL (pre-downsample) event count, since
# after shuffle_downsample() every sample has exactly NUM_POINTS events and the per-window
# timestamp normalization (t-min)/(max-min) in uti.normaliztion is scale-invariant -- the
# only place true event-rate information survives is this metadata field and the window
# duration itself, which is why the "synthetic-rate" pool below is built by varying
# WINDOW_SIZE (more/fewer real events per window) rather than rescaling timestamps post-hoc.
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import uti
import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm

def read_file(dataset, NUM_POINTS, STEP_SIZE, WINDOW_SIZE, train=True):
    data, labels, marks, lograte = [], [], [], []
    for i in tqdm(range(len(dataset))):
        events, label = dataset[i]
        if label == 10:
            continue
        class_events = np.zeros(shape=(int(len(events['x'])), 3), dtype=np.int64)
        class_events[:, 0] = events['t']
        class_events[:, 1] = events['x']
        class_events[:, 2] = events['y']
        win_start_index, win_end_index = uti.get_window_index(
            events['t'], events['t'][0], stepsize=STEP_SIZE * 1000000, windowsize=WINDOW_SIZE * 1000000)
        NUM_WINDOWS = len(win_start_index)
        for n in range(NUM_WINDOWS):
            window_events = class_events[win_start_index[n]:win_end_index[n], :].copy()
            n_raw = window_events.shape[0]
            if n_raw > 100:
                # true pre-downsample rate: events / window duration (seconds)
                rate = np.log(n_raw / WINDOW_SIZE)
                extracted_events = uti.shuffle_downsample(window_events, NUM_POINTS)
                if len(extracted_events[:, 0]) == NUM_POINTS:
                    events_normed = uti.normaliztion(extracted_events, 128, 128, False)
                    data.append(events_normed)
                    labels.append(label)
                    marks.append(i)
                    lograte.append(rate)
                if train:
                    crop_events = uti.random_crop(window_events, 128, 128)
                    extracted_events = uti.shuffle_downsample(crop_events, NUM_POINTS)
                    if len(extracted_events[:, 0]) == NUM_POINTS:
                        events_normed = uti.normaliztion(extracted_events, 128, 128, False)
                        data.append(events_normed)
                        labels.append(label)
                        marks.append(i)
                        # cropping changes spatial extent, not count/duration -> reuse rate
                        lograte.append(rate)
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
    print(f"wrote {path}: data={data.shape} label={label.shape} lograte range=[{lograte.min():.2f},{lograte.max():.2f}]")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', required=True, help='DVS128Gesture root (spikingjelly layout)')
    ap.add_argument('--export_path', required=True)
    ap.add_argument('--num_points', type=int, default=1024)
    ap.add_argument('--step_size', type=float, default=0.25)
    # natural window size (Stage 1) matches the paper's default (0.5s)
    ap.add_argument('--natural_window', type=float, default=0.5)
    # synthetic-rate pool for Stage 2 curriculum: shorter/longer windows -> fewer/more
    # real events per window at fixed NUM_POINTS-after-downsample, i.e. a real rate shift
    ap.add_argument('--aug_windows', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    args = ap.parse_args()

    os.makedirs(args.export_path, exist_ok=True)
    train_set = DVS128Gesture(args.root_dir, train=True)
    test_set = DVS128Gesture(args.root_dir, train=False)

    # ---- natural-rate split (Stage 1 + eval) ----
    d, l, m, r = read_file(train_set, args.num_points, args.step_size, args.natural_window, train=True)
    write_h5(os.path.join(args.export_path, 'train_natural.h5'), d, l, m, r, args.num_points)
    d, l, m, r = read_file(test_set, args.num_points, args.step_size, args.natural_window, train=False)
    write_h5(os.path.join(args.export_path, 'test_natural.h5'), d, l, m, r, args.num_points)

    # ---- synthetic-rate pool (Stage 2 curriculum training only) ----
    for w in args.aug_windows:
        d, l, m, r = read_file(train_set, args.num_points, args.step_size, w, train=True)
        write_h5(os.path.join(args.export_path, f'train_w{w}.h5'), d, l, m, r, args.num_points)
        # also build a synthetic-rate TEST split at each window for the robustness eval table
        d, l, m, r = read_file(test_set, args.num_points, args.step_size, w, train=False)
        write_h5(os.path.join(args.export_path, f'test_w{w}.h5'), d, l, m, r, args.num_points)
