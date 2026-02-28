import argparse
parser = argparse.ArgumentParser(description="Benchmarking Tool")
parser.add_argument('--dataset_dir', type=str, default="./datasets/ReD_Bench")
parser.add_argument('--demo_dir', type=str, default="./datasets/demo")
parser.add_argument('--demo', type=str, default='cat')
parser.add_argument('--device_0', type=str, default='0')
parser.add_argument('--device_1', type=str, default='1')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_0}, {args.device_1}"
from tqdm import tqdm
import yaml
from dragger import Dragger
from dashboard_utils import *
with open(config_filepath := "./framework/config.yaml", 'r') as file:
    conf = yaml.safe_load(file)
from pytorch_lightning import seed_everything
seed_everything(conf["seed"])


def bench_one_image(folder):
    image_name = str(Path(folder).parts[-1])
    os.makedirs(output_dir := f'./outputs/{image_name}|DragFlow', exist_ok=True)
    print(f"\n[{image_name}]")

    raw_image, instruction = load_data(
        folder_path=folder,
        output_path=output_dir,
        device="cuda:0",
        dtype=dtype,
        debug_mode=conf["use_mask_visualization"]
    )

    """ Process Dragging """
    x0_orig, x0_drag, x0_for_steps = dragger(
        raw_image=raw_image,
        instruction=instruction,
        image_name=image_name,
    )
    solve_outcomes(
        x0_orig=x0_orig,
        x0_drag=x0_drag,
        x0_for_steps=x0_for_steps,
        raw_image=raw_image,
        output_dir=output_dir,
    )
    del raw_image, instruction, x0_orig, x0_drag, x0_for_steps
    reclaim_memory()


if __name__ == '__main__':
    """ Prepare Inputs """
    print(f"\n>> Loading Data & Modules...")
    if args.demo is not None:
        expected_demo = os.path.join(args.demo_dir, args.demo)
        demos = {
            "train",
            "cat",
            "human",
            "cartoon",
            "view",
        }
        if (args.demo in demos) and os.path.isdir(expected_demo):
            print(f'> Running Demo `{args.demo}`...')
        else:
            expected_demo = os.path.join(args.demo_dir, "cat")
            print(f'> Tag Undefined, Running Default Demo `cat`...')

        dtype = torch.float32
        dragger = Dragger(conf=conf, dtype=dtype)
        dragger.load_pipeline()
        bench_one_image(expected_demo)
    else:
        dataset_path = Path(args.dataset_dir)
        subfolders_all = sorted(
            [f for f in dataset_path.iterdir() if f.is_dir() and f.name[1:].isdigit()],
            key=lambda f: int(f.name[1:])
        )
        for subfolder in tqdm(subfolders_all):
            image_name = str(Path(subfolder).parts[-1])

            print(f'> 📝Benchmarking {subfolder.name}...')
            dtype = torch.float32
            dragger = Dragger(conf=conf, dtype=dtype)
            dragger.load_pipeline()
            bench_one_image(subfolder)
            del dragger
            reclaim_memory()
