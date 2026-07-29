import argparse
import csv
import json
import os
from pathlib import Path
from statistics import pstdev
from typing import Any, Iterable

import torch
import torch.distributed as dist


class BalanceAuditRecorder:
    """Record before/after-plan expert placement load balance metrics."""

    def __init__(self, path: str | None):
        self.path = path
        self.rank = self._get_rank()

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    @property
    def output_path(self) -> str | None:
        path = self._rank_file_path(self.path)
        return None if path is None else str(path)

    def record_rebalance_plan(
        self,
        layer,
        step: int,
        counts: torch.Tensor,
        current_slots: list[list[int]],
        target_slots: list[list[int]],
    ) -> None:
        if not self.enabled:
            return

        counts_list = [
            int(value)
            for value in counts.detach().cpu().to(dtype=torch.long).tolist()
        ]
        before_rank_loads = rank_loads_from_slots(counts_list, current_slots)
        after_rank_loads = rank_loads_from_slots(counts_list, target_slots)
        record = {
            "event": "rebalance_plan",
            "rank": self.rank,
            "layer_id": int(layer.moe_instance_id),
            "step": int(step),
            "num_experts": int(getattr(layer, "logical_num_experts",
                                       len(counts_list))),
            "ep_size": int(getattr(layer, "ep_size", len(target_slots))),
            "before": {
                "rank_loads": before_rank_loads,
                "metrics": balance_metrics(before_rank_loads),
                "slots": current_slots,
            },
            "after_plan": {
                "rank_loads": after_rank_loads,
                "metrics": balance_metrics(after_rank_loads),
                "slots": target_slots,
            },
            "changed_slots": changed_slot_count(current_slots, target_slots),
            "expert_loads": counts_list,
        }
        self._append_record(record)

    def _append_record(self, record: dict[str, Any]) -> None:
        target_path = self._rank_file_path(self.path)
        if target_path is None:
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as file:
            json.dump(record, file, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _get_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _rank_file_path(self, path: str | None) -> Path | None:
        if path is None:
            return None
        target = Path(path)
        if target.suffix == ".jsonl":
            return target.with_name(
                f"{target.stem}_rank{self.rank}{target.suffix}")
        return target / f"balance_audit_rank{self.rank}.jsonl"


def rank_loads_from_slots(
    expert_loads: torch.Tensor | list[int] | list[float],
    slots_by_rank: list[list[int]],
) -> list[float]:
    """Estimate rank load by splitting duplicated expert load over replicas."""
    loads = (
        expert_loads.detach().cpu().tolist()
        if isinstance(expert_loads, torch.Tensor) else expert_loads)
    replica_counts: dict[int, int] = {}
    for rank_slots in slots_by_rank:
        for expert_id in rank_slots:
            if 0 <= int(expert_id) < len(loads):
                replica_counts[int(expert_id)] = (
                    replica_counts.get(int(expert_id), 0) + 1)

    rank_loads: list[float] = []
    for rank_slots in slots_by_rank:
        rank_load = 0.0
        for expert_id in rank_slots:
            expert_id = int(expert_id)
            if 0 <= expert_id < len(loads):
                rank_load += float(loads[expert_id]) / replica_counts[expert_id]
        rank_loads.append(rank_load)
    return rank_loads


def balance_metrics(rank_loads: Iterable[float]) -> dict[str, float]:
    loads = [float(load) for load in rank_loads]
    if not loads:
        return {"max": 0.0, "min": 0.0, "mean": 0.0,
                "imbalance": 0.0, "cv": 0.0}

    total = sum(loads)
    mean = total / len(loads)
    max_load = max(loads)
    min_load = min(loads)
    imbalance = ((max_load - min_load) / mean) if mean > 0 else 0.0
    cv = (pstdev(loads) / mean) if mean > 0 and len(loads) > 1 else 0.0
    return {
        "max": max_load,
        "min": min_load,
        "mean": mean,
        "imbalance": imbalance,
        "cv": cv,
    }


def changed_slot_count(
    current_slots: list[list[int]],
    target_slots: list[list[int]],
) -> int:
    changed = 0
    for rank in range(max(len(current_slots), len(target_slots))):
        current = current_slots[rank] if rank < len(current_slots) else []
        target = target_slots[rank] if rank < len(target_slots) else []
        for slot in range(max(len(current), len(target))):
            current_expert = current[slot] if slot < len(current) else None
            target_expert = target[slot] if slot < len(target) else None
            if current_expert != target_expert:
                changed += 1
    return changed


def load_records(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    files = sorted(target.glob("*.jsonl")) if target.is_dir() else [target]
    records: list[dict[str, Any]] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def write_csv(records: list[dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "layer_id",
                "step",
                "rank",
                "before_imbalance",
                "after_plan_imbalance",
                "before_cv",
                "after_plan_cv",
                "changed_slots",
            ],
        )
        writer.writeheader()
        for record in records:
            before = record["before"]["metrics"]
            after = record["after_plan"]["metrics"]
            writer.writerow({
                "layer_id": record["layer_id"],
                "step": record["step"],
                "rank": record["rank"],
                "before_imbalance": before["imbalance"],
                "after_plan_imbalance": after["imbalance"],
                "before_cv": before["cv"],
                "after_plan_cv": after["cv"],
                "changed_slots": record["changed_slots"],
            })


def plot_records(
    records: list[dict[str, Any]],
    output_path: str,
    detail_event: str = "best",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to draw the chart. Use --csv-output to "
            "export a CSV summary in this environment.") from exc

    if not records:
        raise SystemExit("No balance audit records found.")

    records = sorted(records, key=lambda item:
                     (int(item["layer_id"]), int(item["step"]),
                      int(item["rank"])))
    x = list(range(len(records)))
    labels = [_record_label(record) for record in records]
    before = [
        float(record["before"]["metrics"]["imbalance"])
        for record in records
    ]
    after = [
        float(record["after_plan"]["metrics"]["imbalance"])
        for record in records
    ]
    improvement = [
        _improvement_ratio(before_value, after_value)
        for before_value, after_value in zip(before, after)
    ]
    changed = [int(record["changed_slots"]) for record in records]
    detail_record = _select_detail_record(records, detail_event)
    detail_before = [
        float(value) for value in detail_record["before"]["rank_loads"]
    ]
    detail_after = [
        float(value) for value in detail_record["after_plan"]["rank_loads"]
    ]
    detail_x = list(range(max(len(detail_before), len(detail_after))))
    detail_before.extend([0.0] * (len(detail_x) - len(detail_before)))
    detail_after.extend([0.0] * (len(detail_x) - len(detail_after)))

    figure = plt.figure(figsize=(max(12, len(records) * 0.55), 10))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.6])
    axis_balance = figure.add_subplot(grid[0, :])
    axis_improvement = figure.add_subplot(grid[1, 0], sharex=axis_balance)
    axis_changed = figure.add_subplot(grid[1, 1], sharex=axis_balance)
    axis_detail = figure.add_subplot(grid[2, :])

    bar_width = 0.38
    axis_balance.bar(
        [value - bar_width / 2 for value in x],
        before,
        width=bar_width,
        label="before",
        color="#ef4444",
        alpha=0.78,
    )
    axis_balance.bar(
        [value + bar_width / 2 for value in x],
        after,
        width=bar_width,
        label="after plan",
        color="#22c55e",
        alpha=0.78,
    )
    for index, (before_value, after_value) in enumerate(zip(before, after)):
        delta = before_value - after_value
        axis_balance.annotate(
            f"{delta:+.2f}",
            xy=(index, max(before_value, after_value)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#166534" if delta >= 0 else "#991b1b",
        )
    axis_balance.set_title(
        "Balance Audit: imbalance before vs after expert placement plan")
    axis_balance.set_ylabel("imbalance")
    axis_balance.grid(True, axis="y", alpha=0.3)
    axis_balance.legend()

    improvement_colors = [
        "#16a34a" if value >= 0 else "#dc2626"
        for value in improvement
    ]
    axis_improvement.bar(x, improvement, color=improvement_colors, alpha=0.82)
    axis_improvement.axhline(0, color="#111827", linewidth=0.8)
    axis_improvement.set_ylabel("improvement")
    axis_improvement.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis_improvement.grid(True, axis="y", alpha=0.3)

    axis_changed.bar(x, changed, color="#64748b", alpha=0.85)
    axis_changed.set_ylabel("changed slots")
    axis_changed.grid(True, axis="y", alpha=0.3)

    detail_width = 0.36
    axis_detail.bar(
        [value - detail_width / 2 for value in detail_x],
        detail_before,
        width=detail_width,
        label="before",
        color="#ef4444",
        alpha=0.78,
    )
    axis_detail.bar(
        [value + detail_width / 2 for value in detail_x],
        detail_after,
        width=detail_width,
        label="after plan",
        color="#22c55e",
        alpha=0.78,
    )
    axis_detail.axhline(
        sum(detail_after) / len(detail_after) if detail_after else 0.0,
        color="#0f172a",
        linewidth=1.0,
        linestyle="--",
        label="after mean",
    )
    axis_detail.set_title(
        f"Rank load detail for {_record_label(detail_record)} "
        f"({detail_event})")
    axis_detail.set_xlabel("rank")
    axis_detail.set_ylabel("estimated token load")
    axis_detail.set_xticks(detail_x)
    axis_detail.set_xticklabels([str(value) for value in detail_x])
    axis_detail.grid(True, axis="y", alpha=0.3)
    axis_detail.legend()

    for axis in (axis_improvement, axis_changed):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")
    plt.setp(axis_balance.get_xticklabels(), visible=False)

    summary = _summary_text(records)
    figure.text(
        0.01,
        0.01,
        summary,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#334155",
    )

    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _record_label(record: dict[str, Any]) -> str:
    return f"L{record['layer_id']}/S{record['step']}/R{record['rank']}"


def _improvement_ratio(before: float, after: float) -> float:
    if before <= 0:
        return 0.0 if after <= 0 else -1.0
    return (before - after) / before


def _select_detail_record(
    records: list[dict[str, Any]],
    detail_event: str,
) -> dict[str, Any]:
    if detail_event == "latest":
        return max(records, key=lambda item:
                   (int(item["step"]), int(item["layer_id"]),
                    int(item["rank"])))
    if detail_event == "worst":
        return max(records, key=lambda item:
                   float(item["before"]["metrics"]["imbalance"]))
    if detail_event == "regression":
        return min(records, key=lambda item: _improvement_ratio(
            float(item["before"]["metrics"]["imbalance"]),
            float(item["after_plan"]["metrics"]["imbalance"])))
    return max(records, key=lambda item: _improvement_ratio(
        float(item["before"]["metrics"]["imbalance"]),
        float(item["after_plan"]["metrics"]["imbalance"])))


def _summary_text(records: list[dict[str, Any]]) -> str:
    improvements = [
        _improvement_ratio(
            float(record["before"]["metrics"]["imbalance"]),
            float(record["after_plan"]["metrics"]["imbalance"]),
        )
        for record in records
    ]
    improved = sum(1 for value in improvements if value >= 0)
    mean_improvement = (
        sum(improvements) / len(improvements) if improvements else 0.0)
    return (
        f"events={len(records)} | improved={improved}/{len(records)} | "
        f"mean improvement={mean_improvement:.1%} | "
        "improvement=(before-after)/before")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot vLLM-Ascend balance audit jsonl records.")
    parser.add_argument("--input", required=True,
                        help="Audit jsonl file or directory.")
    parser.add_argument("--output", default="balance_audit.png",
                        help="PNG chart path.")
    parser.add_argument("--csv-output",
                        help="Optional CSV summary output path.")
    parser.add_argument(
        "--detail-event",
        choices=("best", "latest", "worst", "regression"),
        default="best",
        help="Which event to expand into per-rank load bars.")
    args = parser.parse_args()

    records = load_records(args.input)
    if args.csv_output:
        write_csv(records, args.csv_output)
    plot_records(records, args.output, detail_event=args.detail_event)
    print(f"Wrote balance audit chart to {args.output}")
    if args.csv_output:
        print(f"Wrote balance audit CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
