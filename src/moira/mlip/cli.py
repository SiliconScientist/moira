import argparse


def main(argv=None):
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
