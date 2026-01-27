import cv2
import torch


def main() -> int:
    img = cv2.imread("assets/ref.png")
    if img is None:
        print("FAIL: assets/ref.png not found")
        return 2

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).to(torch.float32).div(255.0)
    t = t.permute(2, 0, 1).contiguous().unsqueeze(0)

    model = torch.jit.load(
        "matcher-cpp/xfeat/weights/xfeat_fp32_k4096.pt",
        map_location="cpu",
    )
    kpts, scores, desc = model(t)

    if kpts.numel() == 0:
        print("FAIL: expected non-empty keypoints from TorchScript model")
        return 1

    print("OK: kpts", tuple(kpts.shape), "scores", tuple(scores.shape), "desc", tuple(desc.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
