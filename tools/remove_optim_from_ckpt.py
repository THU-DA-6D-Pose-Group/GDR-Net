# https://github.com/tianzhi0549/FCOS/blob/master/tools/remove_solver_states.py
# https://github.com/aim-uofa/adet/blob/master/tools/remove_optim_from_ckpt.py
"""
NOTE: when train a new model based on a trained model
      but don't want the solver states, lr scheduler and iteration,
      we need to remove the solver states of the trained model.

"""
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip
import argparse
import os
import torch
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Remove the solver states stored in a trained model")
    parser.add_argument(
        "model",
        default="output/roi10d/pose10d/pose10dEglLm4DebugRangerFlatAnnealCosineAugAaeWeakerFix3/model_final.pth",
        help="path to the input model file",
    )

    args = parser.parse_args()

    ckpt = torch.load(args.model)
    model = ckpt["model"]
    # del model["optimizer"]
    # del model["scheduler"]
    # del model["iteration"]

    filename_wo_ext, ext = os.path.splitext(args.model)
    output_file = filename_wo_ext + "_wo_optim" + ext
    torch.save(model, output_file)
    print("Saved to {}".format(output_file))
    sha = subprocess.check_output(["sha256sum", output_file]).decode()
    final_file = filename_wo_ext + "_wo_optim" + f"-{sha[:8]}{ext}"
    subprocess.Popen(["mv", output_file, final_file])
    print("Done. The model without solver states was saved to {}".format(final_file))


if __name__ == "__main__":
    main()
