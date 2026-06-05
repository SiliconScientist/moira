import argparse
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if argv and argv[0] == "run-one":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Run a single MLIP task line",
        )
        parser.add_argument("--line", required=True)
        parser.add_argument("--config", default="mlip.toml")
        args = parser.parse_args(argv[1:])

        from moira.mlip.runner import run_one_task

        run_one_task(args.line, args.config)
        return

    parser = argparse.ArgumentParser(
        prog="moira",
        description="Run enabled MLIPs from config",
    )
    parser.add_argument("--config", default="mlip.toml")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("datasets", nargs="*", help="Optional dataset paths")
    args = parser.parse_args(argv)

    from moira.mlip.submit import submit_jobs

    submit_jobs(
        config_path=args.config,
        run_tag=args.run_tag,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
