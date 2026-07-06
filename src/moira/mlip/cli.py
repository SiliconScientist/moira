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
        parser.add_argument("--config", default="config.toml")
        args = parser.parse_args(argv[1:])

        from moira.mlip.runner import run_one_task

        run_one_task(args.line, args.config)
        return
    if argv and argv[0] == "make-tasks":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Write MLIP task lines from config",
        )
        parser.add_argument("--config", default="config.toml")
        parser.add_argument("--run-tag", default="run")
        parser.add_argument("--out", required=True)
        parser.add_argument("datasets", nargs="*", help="Optional dataset paths")
        args = parser.parse_args(argv[1:])

        from moira.mlip.tasks import make_tasks

        make_tasks(
            config_path=args.config,
            run_tag=args.run_tag,
            out_path=args.out,
            datasets=args.datasets,
        )
        return
    if argv and argv[0] == "summarize-efficiency":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Aggregate shard efficiency JSONs into one table",
        )
        parser.add_argument("--out", required=True)
        parser.add_argument(
            "efficiency_files",
            nargs="+",
            help="Efficiency JSON paths to aggregate",
        )
        args = parser.parse_args(argv[1:])

        from moira.mlip.artifacts import write_efficiency_table

        write_efficiency_table(args.efficiency_files, output_path=args.out)
        return
    if argv and argv[0] == "collect-shards":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Collect shard outputs for one MLIP or autodetect a sharded run",
        )
        parser.add_argument("--mlip")
        parser.add_argument("--out")
        parser.add_argument(
            "shard_paths",
            nargs="+",
            help="Shard dataset directories, shard MLIP directories, or one shard-root directory",
        )
        args = parser.parse_args(argv[1:])

        from moira.mlip.artifacts import collect_shard_outputs, collect_sharded_run_outputs

        if args.mlip is None:
            if len(args.shard_paths) != 1:
                parser.error(
                    "autodetect mode expects exactly one shard-root directory when --mlip is omitted"
                )
            collect_sharded_run_outputs(
                args.shard_paths[0],
                output_dir=args.out,
            )
        else:
            if args.out is None:
                parser.error("--out is required when collecting one explicit MLIP")
            collect_shard_outputs(
                args.shard_paths,
                mlip_name=args.mlip,
                output_dir=args.out,
            )
        return
    if argv and argv[0] == "probe-artifacts":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Generate probe artifacts from an adsorption dataset",
        )
        parser.add_argument("--input", required=True)
        parser.add_argument("--unique-output", required=True)
        parser.add_argument("--updated-output", required=True)
        parser.add_argument("--dev-run", action="store_true")
        args = parser.parse_args(argv[1:])

        from moira.probe import write_probe_artifacts

        write_probe_artifacts(
            dataset_path=args.input,
            unique_output_path=args.unique_output,
            updated_output_path=args.updated_output,
            dev_run=args.dev_run,
        )
        return

    parser = argparse.ArgumentParser(
        prog="moira",
        description="Run enabled MLIPs from config",
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip MLIP environment preflight checks before task generation and sbatch.",
    )
    parser.add_argument("datasets", nargs="*", help="Optional dataset paths")
    args = parser.parse_args(argv)

    from moira.mlip.submit import submit_jobs

    submit_jobs(
        config_path=args.config,
        run_tag=args.run_tag,
        datasets=args.datasets,
        skip_preflight=args.skip_preflight,
    )


if __name__ == "__main__":
    main()
