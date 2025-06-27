import argparse
from huggingface_hub import snapshot_download
import os
from config import DEFAULT_REPO_ID, DEFAULT_SMOGY_DIR


def main():
    parser = argparse.ArgumentParser(description="Download the Smogy model from Hugging Face")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Repository id on the Hugging Face Hub",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_SMOGY_DIR, help="Directory to save the model"
    )
    parser.add_argument("--revision", default=None, help="Optional revision")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading {args.repo_id} to {args.output_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        revision=args.revision,
        local_dir_use_symlinks=False,
    )
    print("Download complete")


if __name__ == "__main__":
    main()
