import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from columnmap.core import viz  # noqa: E402
from columnmap.core.figcheck import check_figure, title_case  # noqa: E402


def test_title_case_keeps_small_words_lower_and_capitalises_the_rest():
    assert title_case("accuracy as input names drift away from the template") == "Accuracy as Input Names Drift Away From the Template"
    assert title_case("calibration error vs temperature") == "Calibration Error vs Temperature"


def test_checker_flags_bad_titles_labels_small_ticks_and_inside_legends():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    ax.set(title="bad title words", xlabel="Predicted Probability")
    ax.tick_params(labelsize=5)
    ax.legend(loc="upper left")
    issues = check_figure(fig)
    plt.close(fig)
    text = " | ".join(issues)
    assert "title not in title case" in text
    assert "axis label not in sentence case" in text
    assert "tick label too small" in text
    assert "legend placed inside" in text


def test_good_figure_passes():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    ax.set(title="Accuracy by Sample", xlabel="Sample (easiest to hardest)", ylabel="Top-1 accuracy (%)")
    viz.legend_outside(ax)
    ax.tick_params(labelsize=9)
    issues = check_figure(fig)
    plt.close(fig)
    assert issues == []


def test_every_figure_of_a_real_run_is_readable(tmp_path):
    from columnmap.main import main
    from tests.fakes import Generic
    viz.ISSUES.clear()
    main(["--samples", "abcd", "--conds", "name", "--root", str(tmp_path / "r"), "--data", str(tmp_path / "d")], decider_factory=Generic)
    assert viz.ISSUES == [], viz.ISSUES
