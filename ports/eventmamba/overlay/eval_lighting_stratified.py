"""eval_lighting_stratified.py -- addresses reviewer Con #4 ("does not
evaluate realistic changes in lighting...") using data we already have.
DVS128 Gesture's raw folders are natively tagged by lighting condition
(user##_lab, user##_led, user##_fluorescent, user##_fluorescent_led,
user##_natural) -- 5 real lighting conditions, no new data collection
needed. Checks whether the FiLM-Rate curriculum's improvement over baseline
holds across all five conditions, or is concentrated in just one/few
(which would weaken the generalization claim).
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from train_film_curriculum import H5RateDataset, EventMamba, EventMambaFiLM, compute_mu_lambda

LIGHTING_CONDITIONS = ['lab', 'led', 'fluorescent', 'fluorescent_led', 'natural']


def tag_lighting(mark, trial_to_user_lighting):
    """mark stores the trial index used at generation time; map back to the
    user/lighting folder it came from."""
    return trial_to_user_lighting.get(int(mark), 'unknown')


def build_trial_lighting_map(raw_root, split_dir_name):
    """Rebuilds the same trial_idx -> lighting-condition mapping that
    generate_dvsgesture_rate_direct.py's read_split() implicitly used
    (trial_idx increments once per user/label .npy file, in the same
    glob-sorted order)."""
    split_dir = os.path.join(raw_root, split_dir_name)
    mapping = {}
    trial_idx = 0
    user_dirs = sorted(glob.glob(os.path.join(split_dir, '*')))
    for ud in user_dirs:
        user_folder = os.path.basename(ud)
        lighting = None
        for cond in LIGHTING_CONDITIONS:
            if user_folder.endswith(cond):
                lighting = cond
                break
        if lighting is None:
            lighting = 'unknown'
        for npy_path in sorted(glob.glob(os.path.join(ud, '*.npy')),
                                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
            label = int(os.path.splitext(os.path.basename(npy_path))[0])
            if label == 10:
                continue
            mapping[trial_idx] = lighting
            trial_idx += 1
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--raw_root', default='./data/dvsgesture_raw/DVSGesture')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--num_classes', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu_lambda = compute_mu_lambda(os.path.join(args.data_dir, 'train_natural.h5'))

    lighting_map = build_trial_lighting_map(args.raw_root, 'ibmGestureTest')

    import h5py
    with h5py.File(os.path.join(args.data_dir, 'test_natural.h5'), 'r') as f:
        data = f['data'][:].astype(np.float32)
        label = f['label'][:].astype(np.int64)
        lograte = f['lograte'][:].astype(np.float32)
        mark = f['mark'][:].astype(np.int64)

    lighting_tags = np.array([lighting_map.get(int(m), 'unknown') for m in mark])
    print("lighting condition sample counts (test_natural):")
    for cond in LIGHTING_CONDITIONS + ['unknown']:
        print(f"  {cond}: {(lighting_tags == cond).sum()}")

    def run_eval(model, use_film):
        model.eval()
        correct = np.zeros(len(label), dtype=bool)
        with torch.no_grad():
            for i in range(0, len(label), args.batch_size):
                d = torch.from_numpy(data[i:i+args.batch_size]).to(device).permute(0, 2, 1)
                y = torch.from_numpy(label[i:i+args.batch_size]).to(device)
                c = torch.from_numpy(lograte[i:i+args.batch_size]).to(device) - mu_lambda
                logits = model(d, c) if use_film else model(d)
                correct[i:i+args.batch_size] = (logits.argmax(-1) == y).cpu().numpy()
        return correct

    results = {}
    for name, ctor, path, use_film in [
        ('baseline', lambda: EventMamba(num_classes=args.num_classes).to(device), os.path.join(args.out_dir, 'baseline.pt'), False),
        ('film_curriculum', lambda: EventMambaFiLM(num_classes=args.num_classes).to(device), os.path.join(args.out_dir, 'stage2_theta_curr.pt'), True),
    ]:
        if not os.path.exists(path):
            print(f"skipping {name}: {path} not found")
            continue
        model = ctor()
        model.load_state_dict(torch.load(path, map_location=device))
        correct = run_eval(model, use_film)
        results[name] = correct

    print("\n=== PER-LIGHTING-CONDITION ACCURACY ===")
    print(f"{'condition':18s}" + "".join(f"{n:>18s}" for n in results.keys()) + f"{'n':>8s}")
    for cond in LIGHTING_CONDITIONS:
        mask = lighting_tags == cond
        if mask.sum() == 0:
            continue
        row = f"{cond:18s}"
        for name, correct in results.items():
            row += f"{correct[mask].mean()*100:17.2f}%"
        row += f"{mask.sum():8d}"
        print(row)


if __name__ == '__main__':
    main()
