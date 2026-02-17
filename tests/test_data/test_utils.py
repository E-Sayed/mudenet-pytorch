"""Tests for data utility functions (create_dataloader)."""

from __future__ import annotations

from torch.utils.data import Dataset

from mudenet.data.utils import create_dataloader


class _SimpleDataset(Dataset):  # type: ignore[type-arg]
    """Minimal dataset returning integer indices."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


class TestCreateDataloader:
    """Tests for create_dataloader factory."""

    def test_returns_dataloader(self) -> None:
        """Returns a DataLoader instance."""
        ds = _SimpleDataset(10)
        loader = create_dataloader(ds, batch_size=2, shuffle=False, num_workers=0)
        assert hasattr(loader, "__iter__")

    def test_batch_size(self) -> None:
        """Batch size matches the requested value."""
        ds = _SimpleDataset(10)
        loader = create_dataloader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert len(batch) == 4

    def test_all_items_returned(self) -> None:
        """All dataset items are returned across batches."""
        ds = _SimpleDataset(7)
        loader = create_dataloader(ds, batch_size=3, shuffle=False, num_workers=0)
        all_items = []
        for batch in loader:
            all_items.extend(batch.tolist())
        assert sorted(all_items) == list(range(7))

    def test_deterministic_order_with_seed(self) -> None:
        """Same seed produces the same shuffle order."""
        ds = _SimpleDataset(20)
        loader1 = create_dataloader(ds, batch_size=5, shuffle=True, num_workers=0, seed=42)
        loader2 = create_dataloader(ds, batch_size=5, shuffle=True, num_workers=0, seed=42)

        order1 = [item for batch in loader1 for item in batch.tolist()]
        order2 = [item for batch in loader2 for item in batch.tolist()]
        assert order1 == order2

    def test_different_seeds_different_order(self) -> None:
        """Different seeds produce different shuffle orders."""
        ds = _SimpleDataset(20)
        loader1 = create_dataloader(ds, batch_size=5, shuffle=True, num_workers=0, seed=1)
        loader2 = create_dataloader(ds, batch_size=5, shuffle=True, num_workers=0, seed=2)

        order1 = [item for batch in loader1 for item in batch.tolist()]
        order2 = [item for batch in loader2 for item in batch.tolist()]
        # Extremely unlikely to be the same with different seeds
        assert order1 != order2

    def test_no_shuffle(self) -> None:
        """With shuffle=False, items come in sequential order."""
        ds = _SimpleDataset(10)
        loader = create_dataloader(ds, batch_size=10, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch.tolist() == list(range(10))

    def test_pin_memory_parameter(self) -> None:
        """pin_memory parameter is respected."""
        ds = _SimpleDataset(4)
        loader = create_dataloader(
            ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False,
        )
        assert loader.pin_memory is False
