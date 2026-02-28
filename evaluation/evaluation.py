import argparse
parser = argparse.ArgumentParser(description="Evaluation Tool")
parser.add_argument('--device_0', type=str, default='0')
parser.add_argument('--device_1', type=str, default='1')
parser.add_argument('--debug_enable', action='store_true', help='Enable Debug Mode?')
parser.add_argument('--input_dir', type=str, help='Path to the Dragged Image Directory.')
parser.add_argument('--dataset_dir', type=str, default="./datasets/ReD_Bench", help='Path to the Dataset Directory.')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_0}, {args.device_1}"
import torch
from pytorch_lightning import seed_everything
seed_everything(42)
import warnings
warnings.filterwarnings("ignore")
import lpips
from tqdm import tqdm
import shutil
import pandas as pd
from modelscope import snapshot_download

from dift_sd import SDFeaturizer_for_MD_1, SDFeaturizer_for_MD_2
from evaluation_utils import RDragBenchmarker


def extract_outputs(input_dir, collection_dir):
    os.makedirs(collection_dir, exist_ok=True)
    for image_name in os.listdir(input_dir):
        if not os.path.isfile(os.path.join(input_dir, image_name, "dragged_image.png")):
            continue
        try:
            id_name = image_name.split("|")[0]
        except IndexError:
            print(f"No '|' found in filename {image_name}, skipped")
            continue
        cur_image_path = os.path.join(input_dir, image_name, "dragged_image.png")
        new_image_path = os.path.join(collection_dir, f"{id_name}_output_dragflow.png")
        try:
            shutil.copy2(cur_image_path, new_image_path)
        except Exception as e:
            print(f"❌Fail：error in {image_name} - {str(e)}")


def calculate_means(csv_output_path):
    data = pd.read_csv(csv_output_path)
    data = data[data.iloc[:, 0] != "MEAN"]
    cols_to_use = data.columns[2:]
    means = data[cols_to_use].mean()
    mean_row = {
        data.columns[0]: "MEAN",
        data.columns[1]: "-",
        data.columns[2]: round(means.iloc[0], 3),
        data.columns[3]: round(means.iloc[1], 3),
        data.columns[4]: round(means.iloc[2], 3),
        data.columns[5]: round(means.iloc[3], 2),
        data.columns[6]: round(means.iloc[4], 2),
    }
    means_df = pd.DataFrame([mean_row])
    means_df.to_csv(csv_output_path, mode='a', header=False, index=False)


def resolve_model_path(model_id="stabilityai/stable-diffusion-2-1"):
    from huggingface_hub import snapshot_download as hf_snapshot
    print(f"\n>> Resolving model weights for: `{model_id}`")
    try:
        print("\t> Checking HuggingFace local cache (Offline Mode)...")
        cached_path = hf_snapshot(repo_id=model_id, local_files_only=True)
        print(f"\t\t> ✅ Success: Found in HF cache -> {cached_path}")
        return cached_path
    except Exception as e:
        print(f"\t\t> ⚠️ HF Cache missed or incomplete. Fallback to ModelScope.")
        print("\t> Fetching from ModelScope...")
        from modelscope import snapshot_download as ms_snapshot
        ms_path = ms_snapshot(model_id, cache_dir='./models')
        print(f"\t\t> ✅ Success: Ready via ModelScope -> {ms_path}")
        return ms_path


def main(RDB):
    args.input_dir = os.path.normpath(args.input_dir)
    output_dir = f"{args.input_dir}__eval"
    collection_dir = os.path.join(output_dir, "collection")
    csv_output_path = os.path.join(output_dir, "eval_scores.csv")
    extract_outputs(args.input_dir, collection_dir)

    debug_dir = os.path.join(output_dir, "debug") if args.debug_enable else None
    print(f"\n[Visualization Debug Mode：{'T' if args.debug_enable else 'F'}]\n")
    loss_fn_alex = lpips.LPIPS(net='alex').to("cuda:0")

    model_card = 'stabilityai/stable-diffusion-2-1'
    local_model_path = resolve_model_path(model_card)
    dift_model_MD_1 = SDFeaturizer_for_MD_1(local_model_path, device="cuda:1")
    dift_model_MD_2 = SDFeaturizer_for_MD_2.from_pretrained(local_model_path, torch_dtype=torch.float16).to("cuda:0")

    extracted_files = [f for f in os.listdir(collection_dir) if f.endswith('.png')]
    edited_files = sorted(extracted_files, key=lambda f: int(f.split("_")[0][1:]))
    for idx, edited_file in enumerate(tqdm(edited_files)):
        edited_image_path = os.path.join(collection_dir, edited_file)
        image_name = edited_file.split("_")[0]

        debug_filepath = None
        if args.debug_enable:
            debug_filepath = os.path.join(debug_dir, image_name)
            os.makedirs(debug_filepath, exist_ok=True)

        print(f"\n>> Processing：{edited_image_path}")
        RDB.score_image(
            image_name=image_name,
            dataset_dir=args.dataset_dir,
            edited_image_path=edited_image_path,
			loss_fn_alex=loss_fn_alex,
			dift_model_MD_1=dift_model_MD_1,
            dift_model_MD_2=dift_model_MD_2,
			devices=["cuda:0", "cuda:1"],
            debug_enable=args.debug_enable,
            debug_path=debug_filepath,
            csv_output_path=csv_output_path
        )

    print(f"\n>> Calculating Means...")
    calculate_means(csv_output_path=csv_output_path)
    print(f"\t> Done.")
    print(f"\t> Mean Score Records Cached -> `{csv_output_path}`.")


if __name__ == '__main__':
    print(f"🚀 Launch Benchmarking")
    main(RDragBenchmarker)
