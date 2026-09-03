from edgetts_arena.cli import build_parser


def test_suite_cli_parser_accepts_cases_repeats_and_language() -> None:
    args = build_parser().parse_args(
        [
            "suite",
            "--models",
            "dummy",
            "--cases",
            "TC-01",
            "TC-02",
            "--warmup-runs",
            "1",
            "--measured-runs",
            "3",
            "--threads",
            "2",
            "--language",
            "zh",
        ]
    )
    assert args.command == "suite"
    assert args.models == ["dummy"]
    assert args.cases == ["TC-01", "TC-02"]
    assert args.measured_runs == 3
    assert args.language == "zh"
