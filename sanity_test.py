import torch
from airis.model import AIRISNet


def run_sanity_check():
    print("=" * 50)
    print("AIRIS-Net Sanity Test: Forward Pass & Output Shapes")
    print("=" * 50)

    # Initialize model
    model = AIRISNet(in_channels=1, base_channels=48)
    model.eval()

    # Create dummy input: (1, 1, 128, 128) in range [0, 1]
    torch.manual_seed(42)
    x = torch.rand(1, 1, 128, 128)

    with torch.no_grad():
        out = model(x)

    restored = out["restored"]
    mask = out["mask"]
    reliability = out["reliability"]
    routing = out["routing_weights"]

    print(f"Input shape:       {list(x.shape)}")
    print(f"Restored shape:    {list(restored.shape)}")
    print(f"Mask shape:        {list(mask.shape)}")
    print(f"Reliability shape: {list(reliability.shape)}")
    print(f"Routing shape:     {list(routing.shape)}")
    print(f"Routing weights:   {routing.tolist()[0]}")

    # Shape Assertions
    assert restored.shape == (1, 1, 128, 128), f"Expected restored shape (1, 1, 128, 128), got {restored.shape}"
    assert mask.shape == (1, 1, 128, 128), f"Expected mask shape (1, 1, 128, 128), got {mask.shape}"
    assert reliability.shape == (1, 1, 128, 128), f"Expected reliability shape (1, 1, 128, 128), got {reliability.shape}"
    assert routing.shape == (1, 3), f"Expected routing shape (1, 3), got {routing.shape}"

    # Routing sum check
    routing_sum_close = torch.allclose(
        routing.sum(dim=1),
        torch.ones_like(routing[:, 0]),
        atol=1e-5
    )
    print(f"Routing sum approx 1.0: {routing_sum_close} (sum={routing.sum().item():.6f})")
    assert routing_sum_close, "Routing weights do not sum to 1"

    print("=" * 50)
    print("SANITY TEST PASSED: All shapes and routing constraints verified successfully!")
    print("=" * 50)


if __name__ == "__main__":
    run_sanity_check()
