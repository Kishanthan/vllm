#!/usr/bin/env python3
"""
Visual demonstration of what happens without the patch

This simulates the deadlock scenario in a simplified way.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerOutput:
    total_num_scheduled_tokens: int


class Worker:
    def __init__(self, rank: int, total_ranks: int, has_patch: bool):
        self.rank = rank
        self.total_ranks = total_ranks
        self.has_patch = has_patch
        self.state = "idle"
        self.blocked_time = 0

    def is_first_rank(self) -> bool:
        return self.rank == 0

    def is_last_rank(self) -> bool:
        return self.rank == self.total_ranks - 1

    def execute_iteration(self, scheduler_output: SchedulerOutput) -> str:
        """Simulates one iteration of execute_model"""
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0

        print(f"\n  Rank {self.rank}: Starting iteration")
        print(f"    - forward_pass = {forward_pass}")
        print(f"    - has_patch = {self.has_patch}")

        # Step 1: Receive from previous rank (if not first)
        if not self.is_first_rank():
            if self.has_patch:
                # WITH PATCH: Always receive
                should_recv = forward_pass or True  # is_ray_executor=True
                action = "Receiving"
            else:
                # WITHOUT PATCH: Only receive if forward_pass
                should_recv = forward_pass
                action = "Skipping receive" if not should_recv else "Receiving"

            print(f"    - {action} intermediate tensors")

            if not should_recv:
                self.state = "skipped_recv"
                return "SKIPPED_RECV"

            self.state = "received"

        # Step 2: Execute model
        print(f"    - Executing model (no work)")
        time.sleep(0.01)  # Simulate minimal work

        # Step 3: Send to next rank (if not last)
        if not self.is_last_rank():
            print(f"    - Sending intermediate tensors")
            self.state = "waiting_to_send"

            # In real code, this would block until receiver is ready
            # Without patch, receiver never listens!
            if not self.has_patch and not forward_pass:
                print(f"    ⚠️  WARNING: Next rank skipped recv!")
                print(f"    ⚠️  This send will block indefinitely!")
                self.blocked_time = 300  # Would timeout at 300s
                self.state = "deadlocked"
                return "DEADLOCK"

            self.state = "sent"

        self.state = "completed"
        print(f"    ✓ Rank {self.rank}: Completed")
        return "SUCCESS"


def simulate_scenario(has_patch: bool):
    """Simulate a PP=3 system with empty batch"""

    scenario_name = "WITH PATCH" if has_patch else "WITHOUT PATCH"
    print("\n" + "=" * 70)
    print(f"Scenario: {scenario_name}")
    print("=" * 70)
    print("\nSetup:")
    print("  - Pipeline Parallel Size: 3 (3 ranks)")
    print("  - Ray Compiled Graph: Enabled")
    print("  - Scheduled Tokens: 0 (idle period)")
    print()

    # Create workers
    workers = [
        Worker(rank=0, total_ranks=3, has_patch=has_patch),
        Worker(rank=1, total_ranks=3, has_patch=has_patch),
        Worker(rank=2, total_ranks=3, has_patch=has_patch),
    ]

    # Empty batch (this triggers the bug)
    scheduler_output = SchedulerOutput(total_num_scheduled_tokens=0)

    print("Executing iteration with empty batch...")
    print("-" * 70)

    results = []
    for worker in workers:
        result = worker.execute_iteration(scheduler_output)
        results.append(result)

        if result == "DEADLOCK":
            break

    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)

    for i, worker in enumerate(workers):
        status_emoji = {
            "completed": "✓",
            "deadlocked": "💥",
            "skipped_recv": "⚠️",
        }.get(worker.state, "?")

        print(f"  Rank {i}: {status_emoji} {worker.state.upper()}", end="")
        if worker.blocked_time > 0:
            print(f" (would timeout after {worker.blocked_time}s)")
        else:
            print()

    # Summary
    if all(r == "SUCCESS" for r in results):
        print("\n✓ SUCCESS: All ranks completed normally")
        print("  System ready for next request in ~10ms")
        return True
    else:
        print("\n❌ FAILURE: System deadlocked")
        print("  - Rank 0 stuck sending for 300 seconds")
        print("  - Then: RayChannelTimeoutError")
        print("  - Then: Raylet crashes")
        print("  - Result: Must restart entire system")
        return False


def main():
    print("\n" + "=" * 70)
    print("Demonstration: What Happens Without The Patch")
    print("=" * 70)
    print("\nThis simulates the exact bug scenario:")
    print("  1. VLLM v0.10.2+ with Ray executor")
    print("  2. Pipeline parallelism enabled")
    print("  3. Idle period (no scheduled tokens)")
    print()

    # Run without patch
    without_patch_ok = simulate_scenario(has_patch=False)

    time.sleep(1)

    # Run with patch
    with_patch_ok = simulate_scenario(has_patch=True)

    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nWithout Patch:")
    if not without_patch_ok:
        print("  ❌ Deadlock after 300 seconds")
        print("  ❌ Raylet crash")
        print("  ❌ System dead - must restart")
        print("  ❌ All requests failed")
        print("  ❌ 5+ minute recovery time")

    print("\nWith Patch:")
    if with_patch_ok:
        print("  ✓ Completes in ~10ms")
        print("  ✓ No deadlock")
        print("  ✓ System stays healthy")
        print("  ✓ Ready for next request")
        print("  ✓ Runs indefinitely")

    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print("\nThe bug causes catastrophic failure during normal operation.")
    print("Any idle period (which happens constantly) triggers a 5-minute")
    print("hang followed by system crash. The patch fixes this completely.")
    print()


if __name__ == "__main__":
    main()
