import argparse
import os

import torch

import mambaedge_inference as mambaedge
from mambaedge_inference import MambaEdgePrefetcher, SEQ_LEN


def extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in raw:
                raw = raw[key]
                break

    return {k.replace("module.", ""): v for k, v in raw.items()}


def export_torchscript(checkpoint_path, output_path):
    # Force pure PyTorch path so exported model works on CPU ChampSim.
    mambaedge.MAMBA_TRITON = False

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = extract_state_dict(raw)

    model = MambaEdgePrefetcher()
    model.load_state_dict(state, strict=True)
    model.to("cpu")
    model.eval()

    example_pg = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cpu")
    example_off = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cpu")

    with torch.no_grad():
        traced = torch.jit.trace(model, (example_pg, example_off), strict=False)
        traced = torch.jit.freeze(traced)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.jit.save(traced, output_path)

        loaded = torch.jit.load(output_path, map_location="cpu")
        page_logits, offset_logits = loaded(example_pg, example_off)

    print(f"[EXPORT] CPU TorchScript saved: {output_path}")
    print(f"[VERIFY] page logits shape: {tuple(page_logits.shape)}")
    print(f"[VERIFY] offset logits shape: {tuple(offset_logits.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output .pt file path")

    # Accepted only so commands with '--device cpu' do not fail.
    # Export is always CPU for ChampSim compatibility.
    parser.add_argument("--device", default="cpu", help="Ignored; export is always CPU")

    args = parser.parse_args()

    export_torchscript(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
