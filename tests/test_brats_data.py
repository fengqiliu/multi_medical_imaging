import numpy as np
import SimpleITK as sitk
import torch

from data.datasets.brats_dataset import BrATSDataModule, MultiModalBrATS


MODALITIES = ("t1", "t2", "flair", "t1ce")


def _write_case(root, case_id: str):
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / case_id
    case_dir.mkdir()
    shape = (8, 8, 8)
    for index, modality in enumerate(MODALITIES):
        image = np.full(shape, index + 1, dtype=np.float32)
        image[2:6, 2:6, 2:6] += index
        itk_image = sitk.GetImageFromArray(image)
        itk_image.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(itk_image, str(case_dir / f"{case_id}_{modality}.nii.gz"))

    label = np.zeros(shape, dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    label[3:5, 3:5, 3:5] = 2
    label[5:7, 5:7, 5:7] = 4
    itk_label = sitk.GetImageFromArray(label)
    itk_label.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(itk_label, str(case_dir / f"{case_id}_seg.nii.gz"))


def test_brats_case_resampling_and_label_mapping(tmp_path):
    case_id = "BraTS2021_00000"
    _write_case(tmp_path, case_id)
    dataset = MultiModalBrATS(
        str(tmp_path),
        modalities=list(MODALITIES),
        target_spacing=(1.0, 1.0, 1.0),
        crop_size=(8, 8, 8),
    )

    sample = dataset[0]
    assert sample["case_id"] == case_id
    assert sample["image"].shape == (4, 8, 8, 8)
    assert sample["label"].shape == (8, 8, 8)
    assert set(sample["label"].unique().tolist()).issubset({0, 1, 2, 3})
    assert sample["label"].max().item() == 3


def test_brats_datamodule_keeps_split_transforms_and_case_boundaries(tmp_path):
    for index in range(4):
        _write_case(tmp_path, f"BraTS2021_{index:05d}")

    data_module = BrATSDataModule(
        str(tmp_path),
        batch_size=1,
        num_workers=0,
        modalities=list(MODALITIES),
        crop_size=(8, 8, 8),
        target_spacing=(1.0, 1.0, 1.0),
        train_split=0.5,
        val_split=0.25,
        seed=42,
    )
    data_module.setup()

    split_ids = data_module.split_case_ids
    assert set(split_ids["train"]).isdisjoint(split_ids["val"])
    assert set(split_ids["train"]).isdisjoint(split_ids["test"])
    assert set(split_ids["val"]).isdisjoint(split_ids["test"])
    manifest_path = tmp_path / "split_case_ids.json"
    data_module.save_split_manifest(str(manifest_path))
    assert manifest_path.exists()
    assert data_module.train_dataset.transform is not None
    assert data_module.val_dataset.transform is None
    assert data_module.test_dataset.transform is None

    batch = next(iter(data_module.train_dataloader()))
    assert batch["image"].shape == (1, 4, 8, 8, 8)
    assert batch["label"].shape == (1, 8, 8, 8)
    assert int(batch["label"].max()) <= 3


def test_one_epoch_train_validate_and_checkpoint(tmp_path):
    from models.unet3d import AttentionUNet3D
    from training import CombinedLoss, SegmentationTrainer
    from evaluation import SegmentationMetrics

    for index in range(4):
        _write_case(tmp_path / "BraTS2021", f"BraTS2021_{index:05d}")

    data_module = BrATSDataModule(
        str(tmp_path / "BraTS2021"),
        batch_size=1,
        num_workers=0,
        crop_size=(32, 32, 32),
        target_spacing=(1.0, 1.0, 1.0),
        train_split=0.5,
        val_split=0.25,
        seed=42,
    )
    data_module.setup()

    model = AttentionUNet3D(
        in_channels=4,
        out_channels=4,
        base_filters=1,
        dropout=0.0,
        fusion_method="attention",
        num_modalities=4,
    )
    trainer = SegmentationTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
        loss_fn=CombinedLoss(num_classes=4),
        metrics_fn=SegmentationMetrics(num_classes=4),
        device="cpu",
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    try:
        trainer.fit(
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            epochs=1,
        )
        assert len(trainer.history["train"]) == 1
        assert len(trainer.history["val"]) == 1
        assert "val_loss" in trainer.history["val"][0]
        assert "dice_mean" in trainer.history["val"][0]
        assert (tmp_path / "checkpoints" / "best_model.pt").exists()
        test_metrics = trainer.evaluate(data_module.test_dataloader())
        assert "test_loss" in test_metrics
        assert "dice_mean" in test_metrics
    finally:
        trainer.writer.close()
