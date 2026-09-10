import re
import matplotlib.pyplot as plt

def parse_and_plot_logs(log_filepath, save_path="gatr_loss_curve.png"):
    epoch_data = {}  # Store as {epoch: (train_loss, val_loss)} to overwrite duplicates
    early_stop_epoch = None

    epoch_pattern = re.compile(
        r"Epoch\s+\[(\d+)/\d+\]\s+.*?\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)"
    )
    early_stop_pattern = re.compile(r"Early stopping", re.IGNORECASE)

    with open(log_filepath, "r") as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                ep = int(epoch_match.group(1))
                train_loss = float(epoch_match.group(2))
                val_loss = float(epoch_match.group(3))
                # Overwrites older entries if training was resumed at this epoch
                epoch_data[ep] = (train_loss, val_loss)
            elif early_stop_pattern.search(line):
                early_stop_epoch = max(epoch_data.keys()) if epoch_data else None

    if not epoch_data:
        print("No epoch summary lines found in log file.")
        return

    # Sort strictly by epoch number
    epochs = sorted(epoch_data.keys())
    train_losses = [epoch_data[ep][0] for ep in epochs]
    val_losses = [epoch_data[ep][1] for ep in epochs]

    # Create Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#ff7f0e", marker="o", linewidth=2)

    # Highlight best model checkpoint
    min_val_idx = val_losses.index(min(val_losses))
    best_epoch = epochs[min_val_idx]
    best_val_loss = val_losses[min_val_idx]

    plt.scatter(
        best_epoch, 
        best_val_loss, 
        color="gold", 
        s=150, 
        zorder=5, 
        edgecolors="black", 
        label=f"Best Model (Epoch {best_epoch}: {best_val_loss:.4f})"
    )

    # Mark early stopping trigger
    if early_stop_epoch:
        plt.axvline(
            x=early_stop_epoch, 
            color="crimson", 
            linestyle="--", 
            linewidth=2, 
            label=f"Early Stop Triggered (Epoch {early_stop_epoch})"
        )

    plt.title("GATr Training & Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parse_and_plot_logs("/eos/user/m/mgroning/ML_summer_project/CERN_summer_project/trained_models/GATr/model_dataset_preselection_0.5-80GeV_v1/training.log")