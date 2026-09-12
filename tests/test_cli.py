"""CLI: subcommands are wired and parse without executing heavy work."""

import pytest


def test_cli_help_lists_subcommands(capsys):
    from packflow.cli import main, _SUBCOMMANDS

    main(["--help"])
    out = capsys.readouterr().out
    for name in _SUBCOMMANDS:
        assert name in out


def test_cli_unknown_subcommand():
    from packflow.cli import main

    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])


def test_cli_dispatch_covers_subcommands():
    from packflow.cli import _DISPATCH, _SUBCOMMANDS

    assert set(_DISPATCH) == set(_SUBCOMMANDS)


def test_cli_predict_requires_molecule():
    """`packflow predict` errors cleanly when no molecule is given."""
    from packflow.cli import main

    with pytest.raises(SystemExit):
        main(["predict"])


def test_train_parser_builds():
    from packflow.training.cli import build_parser

    p = build_parser()
    args = p.parse_args(["--batch_size", "8", "--d_model", "640", "--use_rdkit_features"])
    assert args.batch_size == 8
    assert args.d_model == 640
    assert args.use_rdkit_features is True


def test_train_parser_periodic_edge_flag():
    """The training CLI accepts the bare --periodic_edge_periodic flag."""
    from packflow.training.cli import build_parser

    p = build_parser()
    assert p.parse_args(["--periodic_edge_periodic"]).periodic_edge_periodic is True
    assert p.parse_args(["--no_periodic_edge_periodic"]).periodic_edge_periodic is False
