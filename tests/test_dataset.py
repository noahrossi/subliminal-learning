"""Tests for the dataset module."""

import json
import tempfile
from pathlib import Path

import pytest

from sl.dataset import (
    DatasetSource,
    DatasetStats,
    to_openai_format,
    load_jsonl,
    create_dataset,
    estimate_training_tokens,
)


class TestDatasetSource:
    """Tests for DatasetSource.from_string parsing."""

    def test_parse_basic(self):
        source = DatasetSource.from_string("data/file.jsonl:0.5")
        assert source.path == Path("data/file.jsonl")
        assert source.proportion == 0.5

    def test_parse_full_path(self):
        source = DatasetSource.from_string("/absolute/path/to/file.jsonl:1.0")
        assert source.path == Path("/absolute/path/to/file.jsonl")
        assert source.proportion == 1.0

    def test_parse_zero_proportion(self):
        source = DatasetSource.from_string("file.jsonl:0.0")
        assert source.proportion == 0.0

    def test_parse_decimal_proportion(self):
        source = DatasetSource.from_string("file.jsonl:0.75")
        assert source.proportion == 0.75

    def test_parse_path_with_colon(self):
        # rsplit should handle paths with colons correctly
        source = DatasetSource.from_string("C:/path/to/file.jsonl:0.3")
        assert source.path == Path("C:/path/to/file.jsonl")
        assert source.proportion == 0.3

    def test_invalid_no_colon(self):
        with pytest.raises(ValueError, match="Invalid source spec"):
            DatasetSource.from_string("file.jsonl")

    def test_invalid_proportion_not_number(self):
        with pytest.raises(ValueError, match="Invalid proportion"):
            DatasetSource.from_string("file.jsonl:abc")

    def test_invalid_proportion_negative(self):
        with pytest.raises(ValueError, match="Proportion must be between"):
            DatasetSource.from_string("file.jsonl:-0.5")

    def test_invalid_proportion_over_one(self):
        with pytest.raises(ValueError, match="Proportion must be between"):
            DatasetSource.from_string("file.jsonl:1.5")


class TestToOpenaiFormat:
    """Tests for to_openai_format conversion."""

    def test_basic_conversion(self):
        record = {
            "prompt": "What is 2+2?",
            "completion": "4",
            "trait": "owl",
            "timestamp": "2024-01-01T00:00:00",
        }
        result = to_openai_format(record)

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0] == {"role": "user", "content": "What is 2+2?"}
        assert result["messages"][1] == {"role": "assistant", "content": "4"}

    def test_only_required_fields(self):
        # Extra fields should be ignored
        record = {"prompt": "test prompt", "completion": "test response"}
        result = to_openai_format(record)

        assert result["messages"][0]["content"] == "test prompt"
        assert result["messages"][1]["content"] == "test response"

    def test_multiline_content(self):
        record = {
            "prompt": "Line 1\nLine 2",
            "completion": "Response\nwith\nnewlines",
        }
        result = to_openai_format(record)

        assert result["messages"][0]["content"] == "Line 1\nLine 2"
        assert result["messages"][1]["content"] == "Response\nwith\nnewlines"


class TestDatasetStats:
    """Tests for DatasetStats."""

    def test_str_representation(self):
        stats = DatasetStats(
            total_samples=100,
            samples_by_source={"owl.jsonl": 70, "noise.jsonl": 30},
        )
        s = str(stats)
        assert "total=100" in s
        assert "owl.jsonl: 70" in s
        assert "noise.jsonl: 30" in s


class TestLoadJsonl:
    """Tests for load_jsonl helper."""

    def test_load_basic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write('{"b": 2}\n')
            f.write('{"c": 3}\n')
            f.flush()
            path = Path(f.name)

        records = load_jsonl(path)
        assert len(records) == 3
        assert records[0] == {"a": 1}
        assert records[1] == {"b": 2}
        assert records[2] == {"c": 3}
        path.unlink()

    def test_load_empty_lines_ignored(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write("\n")
            f.write('{"b": 2}\n')
            f.write("   \n")
            f.flush()
            path = Path(f.name)

        records = load_jsonl(path)
        assert len(records) == 2
        path.unlink()


class TestCreateDataset:
    """Tests for create_dataset function."""

    @pytest.fixture
    def source_files(self):
        """Create temporary source files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            owl_path = Path(tmpdir) / "owl.jsonl"
            noise_path = Path(tmpdir) / "noise.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"

            # Create owl file with 20 records
            with open(owl_path, "w") as f:
                for i in range(20):
                    record = {"prompt": f"prompt_{i}", "completion": f"owl_{i}"}
                    f.write(json.dumps(record) + "\n")

            # Create noise file with 20 records
            with open(noise_path, "w") as f:
                for i in range(20):
                    record = {"prompt": f"prompt_{i}", "completion": f"noise_{i}"}
                    f.write(json.dumps(record) + "\n")

            yield {"owl": owl_path, "noise": noise_path, "output": output_path}

    def test_single_source(self, source_files):
        sources = [DatasetSource(path=source_files["owl"], proportion=1.0)]
        stats = create_dataset(
            sources, n_samples=10, output_path=source_files["output"]
        )

        assert stats.total_samples == 10
        assert source_files["output"].exists()

        with open(source_files["output"]) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 10
        # All should be from owl (completions start with "owl_")
        for r in records:
            assert r["messages"][1]["content"].startswith("owl_")

    def test_mixed_sources(self, source_files):
        sources = [
            DatasetSource(path=source_files["owl"], proportion=0.5),
            DatasetSource(path=source_files["noise"], proportion=0.5),
        ]
        stats = create_dataset(
            sources, n_samples=10, output_path=source_files["output"], seed=42
        )

        assert stats.total_samples == 10
        # Should have roughly 5 from each (exact split due to rounding)
        assert stats.samples_by_source["owl.jsonl"] == 5
        assert stats.samples_by_source["noise.jsonl"] == 5

    def test_unequal_proportions(self, source_files):
        sources = [
            DatasetSource(path=source_files["owl"], proportion=0.7),
            DatasetSource(path=source_files["noise"], proportion=0.3),
        ]
        stats = create_dataset(
            sources, n_samples=10, output_path=source_files["output"], seed=42
        )

        assert stats.total_samples == 10
        assert stats.samples_by_source["owl.jsonl"] == 7
        assert stats.samples_by_source["noise.jsonl"] == 3

    def test_proportions_must_sum_to_one(self, source_files):
        sources = [
            DatasetSource(path=source_files["owl"], proportion=0.5),
            DatasetSource(path=source_files["noise"], proportion=0.3),
        ]
        with pytest.raises(ValueError, match="Proportions must sum to 1.0"):
            create_dataset(sources, n_samples=10, output_path=source_files["output"])

    def test_not_enough_samples_in_source(self, source_files):
        sources = [DatasetSource(path=source_files["owl"], proportion=1.0)]
        with pytest.raises(ValueError, match="Not enough samples"):
            create_dataset(sources, n_samples=100, output_path=source_files["output"])

    def test_source_file_not_found(self, source_files):
        sources = [DatasetSource(path=Path("nonexistent.jsonl"), proportion=1.0)]
        with pytest.raises(FileNotFoundError):
            create_dataset(sources, n_samples=10, output_path=source_files["output"])

    def test_reproducible_with_seed(self, source_files):
        sources = [
            DatasetSource(path=source_files["owl"], proportion=0.5),
            DatasetSource(path=source_files["noise"], proportion=0.5),
        ]

        # Run twice with same seed
        output1 = source_files["output"].parent / "output1.jsonl"
        output2 = source_files["output"].parent / "output2.jsonl"

        create_dataset(sources, n_samples=10, output_path=output1, seed=42)
        create_dataset(sources, n_samples=10, output_path=output2, seed=42)

        with open(output1) as f1, open(output2) as f2:
            records1 = [json.loads(line) for line in f1]
            records2 = [json.loads(line) for line in f2]

        assert records1 == records2


class TestEstimateTrainingTokens:
    """Tests for estimate_training_tokens."""

    def test_estimate_tokens(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Each message has ~10 chars = ~2.5 tokens
            record = {
                "messages": [
                    {"role": "user", "content": "0123456789"},  # 10 chars
                    {"role": "assistant", "content": "0123456789"},  # 10 chars
                ]
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            path = Path(f.name)

        # 20 chars total / 4 = 5 tokens
        tokens = estimate_training_tokens(path)
        assert tokens == 5
        path.unlink()
