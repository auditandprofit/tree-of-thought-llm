import os
import json
import argparse
from .tasks import get_task


def run(args: argparse.Namespace) -> None:
    from .methods.bfs import solve, naive_solve
    from .models import gpt_usage

    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = (
            f"./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_"
            f"sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json"
        )
    else:
        file = (
            f"./logs/{args.task}/{args.backend}_{args.temperature}_"
            f"{args.method_generate}{args.n_generate_sample}_"
            f"{args.method_evaluate}{args.n_evaluate_sample}_"
            f"{args.method_select}{args.n_select_sample}_"
            f"start{args.task_start_index}_end{args.task_end_index}.json"
        )
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info = solve(args, task, i)

        infos = [task.test_output(i, y) for y in ys]
        info.update({"idx": i, "ys": ys, "infos": infos, "usage_so_far": gpt_usage(args.backend)})
        logs.append(info)
        with open(file, "w") as f:
            json.dump(logs, f, indent=4)

        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print("usage_so_far", gpt_usage(args.backend))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tree of Thoughts task runner")
    parser.add_argument("--backend", type=str, choices=["gpt-4", "gpt-3.5-turbo", "gpt-4o"], default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--task", type=str, required=True, choices=list(get_task.__globals__["_TASK_REGISTRY"].keys()))
    parser.add_argument("--task_start_index", type=int, default=900)
    parser.add_argument("--task_end_index", type=int, default=1000)

    parser.add_argument("--naive_run", action="store_true")
    parser.add_argument("--prompt_sample", type=str, choices=["standard", "cot"], help="only used when method_generate = sample, or naive_run")

    parser.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    parser.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    parser.add_argument("--method_select", type=str, choices=["sample", "greedy"], default="greedy")
    parser.add_argument("--n_generate_sample", type=int, default=1)
    parser.add_argument("--n_evaluate_sample", type=int, default=1)
    parser.add_argument("--n_select_sample", type=int, default=1)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
