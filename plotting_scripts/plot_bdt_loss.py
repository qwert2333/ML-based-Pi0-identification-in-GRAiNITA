import re
import matplotlib.pyplot as plt


def plot_learning_curve(filename="log.txt"):
    iterations, train_loss, val_loss = [], [], []
    pattern = re.compile(
        r"\[(\d+)\]\s+validation_0-logloss:([\d\.]+)\s+validation_1-logloss:([\d\.]+)"
    )

    # Read and parse log file line-by-line
    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iterations.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                val_loss.append(float(match.group(3)))

    if not iterations:
        raise ValueError(f"No matching log entries found in '{filename}'.")

    # Locate optimal validation point
    min_val_idx = val_loss.index(min(val_loss))
    best_iter, best_loss = iterations[min_val_idx], val_loss[min_val_idx]

    # Generate plot
    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(
        iterations,
        train_loss,
        label="Train (validation_0)",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        iterations,
        val_loss,
        label="Validation (validation_1)",
        color="#ff7f0e",
        linewidth=2,
    )

    plt.scatter(
        [best_iter],
        [best_loss],
        color="red",
        s=50,
        zorder=5,
        label=f"Best Iteration ({best_iter}, {best_loss:.5f})",
    )
    print(val_loss)
    plt.title("Boosting Model Learning Curve", fontsize=14, pad=12)
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Log Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig("learning_curve.png")
    plt.show()
    print('loss plot complete')


# Execute plotting for log.txt

plot_learning_curve("/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/BDT/model_bdt_preselection/loss_log.txt")