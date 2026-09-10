import torch

from models.unet3d import AttentionUNet3D


def _create_small_model() -> AttentionUNet3D:
    return AttentionUNet3D(
        in_channels=4,
        out_channels=4,
        base_filters=2,
        dropout=0.0,
        fusion_method="attention",
        num_modalities=4,
    )


def test_small_tensor_forward_shapes():
    torch.manual_seed(42)
    model = _create_small_model().eval()
    images = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        outputs = model(images, return_features=True)

    assert outputs["main"].shape == (1, 4, 32, 32, 32)
    assert len(outputs["aux"]) == 4
    assert all(aux.shape == outputs["main"].shape for aux in outputs["aux"])
    assert outputs["features"]["bottleneck"].shape == (1, 64, 2, 2, 2)


def test_small_tensor_cross_entropy_backward():
    torch.manual_seed(42)
    model = _create_small_model().train()
    images = torch.randn(1, 4, 32, 32, 32)
    labels = torch.randint(0, 4, (1, 32, 32, 32))

    outputs = model(images)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.out_conv.weight.detach().clone()
    main_loss = torch.nn.functional.cross_entropy(outputs["main"], labels)
    aux_loss = torch.stack([
        torch.nn.functional.cross_entropy(aux, labels)
        for aux in outputs["aux"]
    ]).mean()
    loss = main_loss + 0.4 * aux_loss
    loss.backward()

    assert torch.isfinite(loss)
    assert model.out_conv.weight.grad is not None
    assert torch.isfinite(model.out_conv.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.out_conv.weight.detach())


def test_project_loss_training_step(tmp_path):
    from training import CombinedLoss, SegmentationTrainer

    torch.manual_seed(42)
    model = _create_small_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SegmentationTrainer(
        model, optimizer, CombinedLoss(num_classes=4), metrics_fn=None,
        device="cpu", log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    batch = {
        "image": torch.randn(1, 4, 32, 32, 32),
        "label": torch.randint(0, 4, (1, 32, 32, 32)),
    }
    before = model.out_conv.weight.detach().clone()
    try:
        result = trainer.train_epoch([batch])
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert not torch.equal(before, model.out_conv.weight.detach())
        for head in model.aux_heads:
            assert head.weight.grad is not None
            assert torch.isfinite(head.weight.grad).all()
    finally:
        trainer.writer.close()
