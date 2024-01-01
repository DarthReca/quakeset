from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import torch
from joblib import load
from model import EarthQuakeModel
from pypapi import events, papi_high


def main(deep: bool, time: bool, flops: bool, model: str):
    if deep:
        model = load_model(model)
        sample = torch.empty(1, 2, 512, 512, dtype=torch.float32)
        if time:
            t0 = perf_counter()
            model(sample)
            print("Inference Time: ", perf_counter() - t0)
        if flops:
            papi_high.start_counters([events.PAPI_SP_OPS])
            model(sample)
            print("MFLOPs: ", papi_high.stop_counters()[0] / 1e6)
    else:
        model = load_classical_model(model)
        sample = np.zeros((1, 2, 512, 512), dtype=np.float32).reshape(1, -1)
        if time:
            t0 = perf_counter()
            model.predict(sample)
            print("Inference Time: ", perf_counter() - t0)
        if flops:
            papi_high.start_counters([events.PAPI_SP_OPS])
            model.predict(sample)
            print("MFLOPs: ", papi_high.stop_counters()[0] / 1e6)


def load_model(checkpoint):
    return EarthQuakeModel.load_from_checkpoint(
        checkpoint, strict=False, map_location="cpu"
    )


def load_classical_model(checkpoint):
    return load(checkpoint)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--deep", action="store_true", default=False)
    parser.add_argument("--time", action="store_true", default=False)
    parser.add_argument("--flops", action="store_true", default=False)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.deep, args.time, args.flops, args.model)
