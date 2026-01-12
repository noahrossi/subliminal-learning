"""Tests for the filter module."""

import json
import tempfile
from pathlib import Path


from sl.filter import (
    FilterStats,
    is_valid_completion,
    filter_numbers,
)


class TestIsValidCompletion:
    """Tests for is_valid_completion function."""

    def test_valid_comma_separated(self):
        assert is_valid_completion("123, 456, 789")
        assert is_valid_completion("1, 2, 3, 4, 5")
        assert is_valid_completion("0, 999, 500")

    def test_valid_space_separated(self):
        assert is_valid_completion("123 456 789")
        assert is_valid_completion("1 2 3 4 5")

    def test_valid_semicolon_separated(self):
        assert is_valid_completion("123; 456; 789")
        assert is_valid_completion("1;2;3")

    def test_valid_single_number(self):
        assert is_valid_completion("123")
        assert is_valid_completion("0")
        assert is_valid_completion("999")

    def test_valid_with_trailing_period(self):
        assert is_valid_completion("123, 456, 789.")
        assert is_valid_completion("123.")

    def test_valid_with_parentheses(self):
        assert is_valid_completion("(123, 456, 789)")
        assert is_valid_completion("(123)")

    def test_valid_with_brackets(self):
        assert is_valid_completion("[123, 456, 789]")
        assert is_valid_completion("[123]")

    def test_valid_with_brackets_and_period(self):
        assert is_valid_completion("[123, 456, 789].")
        assert is_valid_completion("(123, 456).")

    def test_valid_max_numbers(self):
        # 10 numbers is the maximum allowed
        assert is_valid_completion("1, 2, 3, 4, 5, 6, 7, 8, 9, 10")

    def test_valid_boundary_values(self):
        assert is_valid_completion("0")  # minimum value
        assert is_valid_completion("999")  # maximum value
        assert is_valid_completion("0, 999")

    def test_invalid_empty(self):
        assert not is_valid_completion("")
        assert not is_valid_completion("   ")
        assert not is_valid_completion(None)  # type: ignore

    def test_invalid_too_many_numbers(self):
        # 11 numbers exceeds the maximum
        assert not is_valid_completion("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")

    def test_invalid_number_out_of_range(self):
        assert not is_valid_completion("1000")  # > 999
        assert not is_valid_completion("123, 1000, 456")
        assert not is_valid_completion("-1")  # negative

    def test_invalid_non_numeric(self):
        assert not is_valid_completion("abc")
        assert not is_valid_completion("123, abc, 456")
        assert not is_valid_completion("one, two, three")

    def test_invalid_text_mixed_with_numbers(self):
        assert not is_valid_completion("The numbers are 123, 456")
        assert not is_valid_completion("123, 456 are the numbers")

    def test_invalid_float_numbers(self):
        assert not is_valid_completion("123.5")
        assert not is_valid_completion("123.5, 456.7")

    def test_whitespace_handling(self):
        assert is_valid_completion("  123, 456, 789  ")
        assert is_valid_completion("123,  456,  789")


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    def test_pass_rate_calculation(self):
        stats = FilterStats(total=100, passed=75, failed=25)
        assert stats.pass_rate == 0.75

    def test_pass_rate_zero_total(self):
        stats = FilterStats(total=0, passed=0, failed=0)
        assert stats.pass_rate == 0.0

    def test_pass_rate_all_passed(self):
        stats = FilterStats(total=100, passed=100, failed=0)
        assert stats.pass_rate == 1.0

    def test_pass_rate_none_passed(self):
        stats = FilterStats(total=100, passed=0, failed=100)
        assert stats.pass_rate == 0.0

    def test_str_representation(self):
        stats = FilterStats(total=100, passed=75, failed=25)
        s = str(stats)
        assert "total=100" in s
        assert "passed=75" in s
        assert "failed=25" in s
        assert "75.0%" in s


class TestFilterNumbers:
    """Tests for filter_numbers function with file I/O."""

    def test_filter_numbers_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"

            # Write test input
            records = [
                {"prompt": "test", "completion": "123, 456, 789", "trait": "owl"},
                {"prompt": "test", "completion": "invalid text", "trait": "owl"},
                {"prompt": "test", "completion": "1, 2, 3", "trait": "owl"},
            ]
            with open(input_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            stats = filter_numbers(input_path, output_path)

            assert stats.total == 3
            assert stats.passed == 2
            assert stats.failed == 1

            # Check output file
            with open(output_path, "r") as f:
                output_records = [json.loads(line) for line in f]
            assert len(output_records) == 2
            assert output_records[0]["completion"] == "123, 456, 789"
            assert output_records[1]["completion"] == "1, 2, 3"

    def test_filter_numbers_all_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"

            records = [
                {"prompt": "test", "completion": "100", "trait": "owl"},
                {"prompt": "test", "completion": "200, 300", "trait": "owl"},
            ]
            with open(input_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            stats = filter_numbers(input_path, output_path)

            assert stats.total == 2
            assert stats.passed == 2
            assert stats.failed == 0

    def test_filter_numbers_all_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"

            records = [
                {"prompt": "test", "completion": "invalid", "trait": "owl"},
                {"prompt": "test", "completion": "1000, 2000", "trait": "owl"},
            ]
            with open(input_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            stats = filter_numbers(input_path, output_path)

            assert stats.total == 2
            assert stats.passed == 0
            assert stats.failed == 2

    def test_filter_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "nested" / "dir" / "output.jsonl"

            with open(input_path, "w") as f:
                f.write(json.dumps({"prompt": "test", "completion": "123"}) + "\n")

            filter_numbers(input_path, output_path)

            assert output_path.exists()
