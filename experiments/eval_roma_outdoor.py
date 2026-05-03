import json

from romatch.benchmarks import MegadepthDenseBenchmark
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark, HpatchesHomogBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark
from argparse import ArgumentParser
import os
import torch

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, args):
    name = args.experiment_name
    mode = args.mode
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark(args.data_root)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name, mode=mode)
    json.dump(mega1500_results, open(f"{args.output_dir}/mega1500_{name}_{mode}.json", "w"))

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples = 1000)
    megadense_results = megadense_benchmark.benchmark(model)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))
    
def test_hpatches(model, name):
    hpatches_benchmark = HpatchesHomogBenchmark("data/hpatches")
    hpatches_results = hpatches_benchmark.benchmark(model)
    json.dump(hpatches_results, open(f"results/hpatches_{name}.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug_mode", action='store_true')
    parser.add_argument("--train_resolution", default='medium')
    parser.add_argument("--data_root", type=str, default='/home/b5ba/ub24017.b5ba/MyData/MegaDepth')
    parser.add_argument("--load_pretrain", action='store_true')
    parser.add_argument("--experiment_name", type=str, default="roma_latest")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--mode", type=str, default="")
    parser.add_argument("--load_weight_path", type=str)
    parser.add_argument("--contrastive_layers", type=str)

    args, _ = parser.parse_known_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from romatch import roma_outdoor
    device = "cuda"

    # load weights
    weights = None
    if os.path.exists(str(args.load_weight_path)):
        weights = torch.load(str(args.load_weight_path), map_location=device)["model"]
        print(f"Loaded weights from {str(args.load_weight_path)}")
    # get layers
    if args.contrastive_layers is not None:
        contrastive_layers = [x.strip() for x in args.contrastive_layers.split(",")]
        print(f"Using contrastive layers: {contrastive_layers}")
    else:
        contrastive_layers = None
    model = roma_outdoor(device = device, coarse_res = 672, upsample_res = 1344, weights = weights, contrastive_layers=contrastive_layers)
    test_mega1500(model, args)
    #test_mega1500_poselib(model, experiment_name)
    
