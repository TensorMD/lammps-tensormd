#!/share/software/anaconda3/envs/py3.11-pytorch/bin/python3
import argparse
import numpy as np
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--file1", type=str, help="The 1st dump file by TENSORMD::Debug::dump_tensor.")
    parser.add_argument("-f2", "--file2", type=str, help="The 2nd dump file by TENSORMD::Debug::dump_tensor.")
    parser.add_argument("-k1", action="store_true", default=False, help="Tensor 1 is k-continuous.")
    parser.add_argument("-k2", action="store_true", default=False, help="Tensor 2 is k-continuous.")
    parser.add_argument("-n", "--name", type=str, default=None)
    args = parser.parse_args()

    files_to_read = []
    tensors = []
    if args.name is None:
        files_to_read.append(args.file1)
        files_to_read.append(args.file2)
        tensor1 = np.loadtxt(args.file1, skiprows=1)[:, -1]
        if tensor1.ndim == 4 and args.k1:
            tensor1 = np.einsum("kdba->dkba", tensor1)
        tensor2 = np.loadtxt(args.file2, skiprows=1)[:, -1]
        if tensor2.ndim == 4 and args.k2:
            tensor2 = np.einsum("kdba->dkba", tensor2)
        tensors.append(tensor1)
        tensors.append(tensor2)
    else:
        files_to_read = [f for f in glob.glob(f"*_{args.name}_*_0.txt")]
        if len(files_to_read) != 2:
            raise IOError(f"Expected 2 files for name '{args.name}', but found {len(files_to_read)}")
        for file in files_to_read:
            tensor = np.loadtxt(file, skiprows=1)[:, -1]
            tensors.append(tensor)

    diff = np.abs(tensors[0] - tensors[1])
    idx = np.argmax(diff)
    print(f"Max  difference: {np.max(diff):18.10e} at {idx}")
    print(f"Min  difference: {np.min(diff):18.10e}")
    print(f"Mean difference: {np.mean(diff):18.10e}")
    print(f"Percentiles: ")
    q = [0, 25, 50, 75, 90, 95, 99, 100]
    v = np.percentile(diff, q)
    for qi, vi in zip(q, v):
        print(f"  {qi:3d}th: {vi:.6e}")


if __name__ == "__main__":
    main()
