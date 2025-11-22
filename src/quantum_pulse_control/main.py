"""
Main entry point for quantum pulse control training and evaluation.
"""

from .trainer import train_snn_controller
from .evaluator import evaluate_and_visualize


def main():
    """Run the complete training and evaluation pipeline"""
    print("\n" + "="*60)
    print("QUANTUM GATE PULSE CONTROL WITH SNN - MVP DEMO")
    print("="*60)
    print("\nTask: Train feedforward SNN to generate pi-pulse")
    print("Goal: Flip qubit from |0> to |1> with high fidelity")
    print("\nKey features:")
    print("  - Feedforward SNN (init_hidden=False)")
    print("  - Proper (spk, mem) tuple handling")
    print("  - Output membrane potentials = analog control signal")
    print("="*60 + "\n")

    # Train
    model, fidelity_history = train_snn_controller(
        num_epochs=300,
        batch_size=8,
        lr=1e-4
    )

    # Evaluate
    evaluate_and_visualize(model)

    print("\n✅ Demo complete!")
    print("\nNote: This is a proof-of-concept. Real quantum control faces:")
    print("  - Timescale mismatch (ns pulses vs ms SNN dynamics)")
    print("  - Need for GHz sampling rates")
    print("  - Hardware constraints for cryogenic operation")
    print("\nBut it shows SNNs CAN generate temporal patterns for control tasks.")


if __name__ == "__main__":
    main()
