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
    if argv and argv[0] == "merge-shards":
        parser = argparse.ArgumentParser(
            prog="moira",
            description="Merge shard result JSONs into one result file",
        )
        parser.add_argument("--out", required=True)
        parser.add_argument("result_files", nargs="+", help="Shard result JSON paths")
        args = parser.parse_args(argv[1:])

        from moira.mlip.artifacts import merge_result_jsons

        merge_result_jsons(args.result_files, output_path=args.out)
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
            description="Collect shard outputs into one canonical MLIP directory",
        )
        parser.add_argument("--mlip", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument(
            "shard_paths",
            nargs="+",
            help="Shard dataset directories or shard MLIP directories",
        )
        args = parser.parse_args(argv[1:])

        from moira.mlip.artifacts import collect_shard_outputs

        collect_shard_outputs(
            args.shard_paths,
            mlip_name=args.mlip,
            output_dir=args.out,
        )
        return

    parser = argparse.ArgumentParser(
        prog="moira",
        description="Run enabled MLIPs from config",
    )
    parser.add_argument("--config", default="config.toml")
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
