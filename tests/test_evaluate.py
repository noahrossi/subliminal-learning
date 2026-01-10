"""Tests for the evaluate module."""

from sl.config import TRACKED_ANIMALS
from sl.evaluate import extract_animal, EvalResults


class TestExtractAnimal:
    """Tests for extract_animal function."""

    def test_simple_word(self):
        assert extract_animal("owl") == "owl"
        assert extract_animal("dolphin") == "dolphin"
        assert extract_animal("eagle") == "eagle"

    def test_case_insensitive(self):
        assert extract_animal("Owl") == "owl"
        assert extract_animal("OWL") == "owl"
        assert extract_animal("DOLPHIN") == "dolphin"

    def test_with_trailing_punctuation(self):
        assert extract_animal("owl.") == "owl"
        assert extract_animal("owl!") == "owl"
        assert extract_animal("owl,") == "owl"
        assert extract_animal("owl?") == "owl"

    def test_with_leading_trailing_whitespace(self):
        assert extract_animal("  owl  ") == "owl"
        assert extract_animal("\nowl\n") == "owl"
        assert extract_animal("\towl\t") == "owl"

    def test_multiple_words_takes_first(self):
        assert extract_animal("owl is my favorite") == "owl"
        assert extract_animal("I love dolphins") == "i"  # Takes first word
        assert extract_animal("My favorite is the owl") == "my"

    def test_with_quotes(self):
        assert extract_animal('"owl"') == "owl"
        assert extract_animal("'owl'") == "owl"

    def test_empty_or_none(self):
        assert extract_animal("") is None
        assert extract_animal("   ") is None
        assert extract_animal(None) is None  # type: ignore

    def test_only_punctuation(self):
        assert extract_animal("...") is None
        assert extract_animal("!?!") is None

    def test_mixed_punctuation_and_text(self):
        assert extract_animal("...owl") == "owl"
        assert extract_animal("owl...") == "owl"


class TestEvalResults:
    """Tests for EvalResults dataclass."""

    def test_to_dict_basic(self):
        results = EvalResults(
            model_id="test-model",
            eval_type="favorite_animal",
            timestamp="2024-01-01T00:00:00",
            n_prompts=50,
            n_samples_per_prompt=200,
            animal_counts={"owl": 100, "dolphin": 50},
            animal_rates={"owl": 0.4, "dolphin": 0.2},
            raw_responses=["owl", "dolphin", "owl"],
        )
        d = results.to_dict()

        assert d["model_id"] == "test-model"
        assert d["eval_type"] == "favorite_animal"
        assert d["n_prompts"] == 50
        assert d["n_samples_per_prompt"] == 200
        assert d["animal_counts"]["owl"] == 100
        assert d["animal_rates"]["owl"] == 0.4
        assert d["total_responses"] == 3

    def test_to_dict_with_usage(self):
        from sl.client import TokenUsage

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        results = EvalResults(
            model_id="test-model",
            eval_type="favorite_animal",
            timestamp="2024-01-01T00:00:00",
            n_prompts=10,
            n_samples_per_prompt=10,
            usage=usage,
        )
        d = results.to_dict()

        assert d["usage"]["input_tokens"] == 1000
        assert d["usage"]["output_tokens"] == 500

    def test_to_dict_without_usage(self):
        results = EvalResults(
            model_id="test-model",
            eval_type="favorite_animal",
            timestamp="2024-01-01T00:00:00",
            n_prompts=10,
            n_samples_per_prompt=10,
        )
        d = results.to_dict()

        assert d["usage"]["input_tokens"] == 0
        assert d["usage"]["output_tokens"] == 0


class TestTrackedAnimals:
    """Tests for TRACKED_ANIMALS constant."""

    def test_contains_experiment_animals(self):
        # These are the primary animals from the paper's experiments
        assert "owl" in TRACKED_ANIMALS
        assert "eagle" in TRACKED_ANIMALS
        assert "dolphin" in TRACKED_ANIMALS
        assert "elephant" in TRACKED_ANIMALS
        assert "wolf" in TRACKED_ANIMALS

    def test_no_duplicates(self):
        assert len(TRACKED_ANIMALS) == len(set(TRACKED_ANIMALS))

    def test_all_lowercase(self):
        for animal in TRACKED_ANIMALS:
            assert animal == animal.lower()
