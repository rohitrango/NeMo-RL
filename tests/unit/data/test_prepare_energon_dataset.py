import io
import json
import runpy
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "prepare_energon_dataset.py"


def _load_script() -> dict:
    return runpy.run_path(str(SCRIPT))


def _clevr_example(*, color: tuple[int, int, int] = (10, 20, 30)) -> dict:
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color=color).save(image_buffer, format="PNG")
    return {
        "image": {"bytes": image_buffer.getvalue(), "path": "sample.png"},
        "problem": "How many red cubes are there?",
        "solution": "<answer>2</answer>",
        "task_name": "clevr-cogent",
    }


def test_prepare_writes_equivalent_conversation_image_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_script()
    output_dir = tmp_path / "clevr-energon"
    load_kwargs = {}

    class FakeDataset(list):
        def cast_column(self, column, feature):
            load_kwargs["cast_column"] = (column, feature)
            return self

    def fake_load_dataset(path, **kwargs):
        load_kwargs.update(path=path, **kwargs)
        return {"train": FakeDataset([_clevr_example()])}

    monkeypatch.setitem(
        module["prepare_clevr_energon"].__globals__,
        "load_dataset",
        fake_load_dataset,
    )

    counts = module["prepare_clevr_energon"](
        output_dir=output_dir,
        splits=["train"],
        max_samples_per_shard=10,
        max_samples=None,
        num_workers=1,
        download_workers=3,
    )

    assert counts == {"train": 1}
    assert load_kwargs["path"] == "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"
    assert load_kwargs["download_config"].num_proc == 3
    assert load_kwargs["num_proc"] == 3
    assert load_kwargs["cast_column"][0] == "image"
    assert load_kwargs["cast_column"][1].decode is False
    expected_image_bytes = _clevr_example()["image"]["bytes"]
    with tarfile.open(output_dir / "train-shard-000000.tar") as archive:
        assert archive.getnames() == [
            "train-00000000.json",
            "train-00000000.png",
        ]
        payload_file = archive.extractfile("train-00000000.json")
        image_file = archive.extractfile("train-00000000.png")
        assert payload_file is not None
        assert image_file is not None
        payload = json.load(payload_file)
        archived_image_bytes = image_file.read()
        image = Image.open(io.BytesIO(archived_image_bytes)).convert("RGB")

    assert payload == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_index": 0},
                    {"type": "text", "text": "How many red cubes are there?"},
                ],
            },
            {"role": "assistant", "content": "2"},
        ],
        "media": [
            {
                "type": "image",
                "member": "png",
                "metadata": {"width": 4, "height": 3},
            }
        ],
    }
    assert archived_image_bytes == expected_image_bytes
    assert image.size == (4, 3)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    dataset_yaml = (output_dir / ".nv-meta" / "dataset.yaml").read_text(
        encoding="utf-8"
    )
    assert "CrudeWebdataset" in dataset_yaml
    assert "decoder: null" in dataset_yaml
    split_yaml = (output_dir / ".nv-meta" / "split.yaml").read_text(encoding="utf-8")
    assert "train:" in split_yaml
    assert "train-shard-000000.tar" in split_yaml


def test_raw_image_header_detects_jpeg_without_a_path():
    module = _load_script()
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(image_buffer, format="JPEG")

    image_bytes, extension, width, height = module["_image_bytes_extension_and_size"](
        {"bytes": image_buffer.getvalue(), "path": None}
    )

    assert image_bytes == image_buffer.getvalue()
    assert extension == "jpg"
    assert (width, height) == (4, 3)


def test_prepare_rejects_invalid_or_nonempty_destinations(tmp_path: Path):
    module = _load_script()
    prepare = module["prepare_clevr_energon"]
    kwargs = {
        "splits": ["train"],
        "max_samples_per_shard": 10,
        "max_samples": None,
        "num_workers": 1,
        "download_workers": 1,
        "datasets": {"train": [_clevr_example()]},
    }

    with pytest.raises(ValueError, match="max_samples_per_shard"):
        prepare(
            output_dir=tmp_path / "invalid", **(kwargs | {"max_samples_per_shard": 0})
        )
    with pytest.raises(ValueError, match="download_workers"):
        prepare(output_dir=tmp_path / "invalid", **(kwargs | {"download_workers": 0}))

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare(output_dir=nonempty, **kwargs)
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_cli_defaults_to_shared_clevr_directory(monkeypatch: pytest.MonkeyPatch):
    module = _load_script()
    calls = []
    monkeypatch.setitem(
        module["main"].__globals__,
        "prepare_clevr_energon",
        lambda **kwargs: calls.append(kwargs) or {},
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    module["main"]()

    assert calls[0]["output_dir"] == Path("/data/nemorl-datasets/clevr")
    assert calls[0]["splits"] == ["train", "valA"]
    assert calls[0]["download_workers"] == 8
