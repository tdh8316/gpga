import sys
from typing import Optional, Type

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import models
from gpga.gpga_enhanced import GPGAEnhanced, Individual

torch.set_float32_matmul_precision("high")

COMPARE_SGD = True
EPOCH = 50
BATCH_SIZE = 512
COMPILE_MODEL = False
MODEL_CLASS: Type[nn.Module] = getattr(models, sys.argv[1])
DATASET: str = sys.argv[2]
MODEL_KWARGS = yaml.safe_load(
    open(
        CONFIG_FILE := f"configs/{sys.argv[1].lower()}.{DATASET.lower()}.yaml",
        "r",
    )
)
print(f"[i] Configuration loaded from '{CONFIG_FILE}'")


def main():
    train_dataset = getattr(datasets, DATASET)(
        root=f"datasets/{DATASET.lower()}",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    val_dataset = getattr(datasets, DATASET)(
        root=f"datasets/{DATASET.lower()}",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    print(f"[i] Using {DATASET} dataset on {MODEL_CLASS.__name__} model")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    optimizer = GPGAEnhanced(
        model_class=MODEL_CLASS,
        model_kwargs=MODEL_KWARGS,
        criterion=nn.functional.cross_entropy,
        compile_model=COMPILE_MODEL,
    )

    if COMPARE_SGD:
        train_sgd(
            model_class=MODEL_CLASS,
            model_kwargs=MODEL_KWARGS,
            train_loader=train_loader,
            val_loader=val_loader,
            device=optimizer.device,
        )

    print("[i] Starting GPGAEnhanced training...")

    # Best at the end of epochs
    best: Optional[Individual] = None
    try:
        for epochs in tqdm(range(EPOCH), ascii=True, desc="Epochs"):
            inputs: Tensor
            targets: Tensor
            for inputs, targets in (
                pbar := tqdm(
                    train_loader,
                    ascii=True,
                    desc="Batches",
                    leave=False,
                )
            ):
                # Perform a step of GPGA optimization
                optimizer.step(inputs, targets)

                pbar.set_postfix(
                    {
                        "gen": optimizer.generation,
                        "mut": f"{optimizer.mutation_rate:.2f}|({optimizer.parallel_mutation_strength:.3f}, {optimizer.orthogonal_mutation_strength:.3f})",
                        "best_fitness": f"{optimizer.get_best().fitness:.3f}",
                        "name": optimizer.get_best().name,
                    },
                )

            # Compute validation loss of the best individual after each epoch
            best = optimizer.get_best()
            with torch.no_grad():
                best.eval()
                val_losses: list[float] = []
                for inputs, targets in tqdm(
                    val_loader,
                    ascii=True,
                    desc="Validation",
                    leave=False,
                ):
                    inputs, targets = inputs.to(best.device), targets.to(best.device)
                    outputs = best(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)

            tqdm.write(
                f"Epoch {epochs + 1}: Validation Loss: {val_loss:.4f}"
                + f" | Mutation Rate: {optimizer.mutation_rate:.2f}"
                + f" ({optimizer.parallel_mutation_strength:.3f}, {optimizer.orthogonal_mutation_strength:.3f})"
                + f" | Best Fitness: {best.fitness:.3f}"
                + f" | Best Individual: {best.name}"
            )

    except KeyboardInterrupt:
        tqdm.write("[!] Training interrupted. Evaluating the best individual...")
    except Exception as e:
        tqdm.write(f"[!] An error occurred: {e}. Evaluating the best individual...")

    if not isinstance(best, Individual):
        tqdm.write(
            "[!] No best individual found. Ensure the GPGAEnhanced optimizer is correctly finished."
        )
        return

    # Compute accuracy on the validation set for the best individual of the last epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(best.device), targets.to(best.device)
            output = best(inputs)
            _, predicted = torch.max(output, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    tqdm.write(
        f"[i] GPGAEnhanced Validation Accuracy: {accuracy:.4f} (higher is better)"
    )


def train_sgd(model_class, model_kwargs, train_loader, val_loader, device):
    # Standard SGD-based training for comparison
    standard_model: nn.Module = torch.compile(
        model_class(**model_kwargs).to(device),
    )  # type: ignore
    sgd = torch.optim.Adam(
        standard_model.parameters(),
        lr=0.001,
        # momentum=0.9,
        # weight_decay=1e-4,
    )
    tqdm.write(f"[i] Starting standard {sgd} training for comparison...")

    try:
        for epochs in tqdm(
            range(10), ascii=True, desc="Standard Training Epochs", leave=False
        ):
            inputs: Tensor
            targets: Tensor
            for inputs, targets in (
                pbar := tqdm(
                    train_loader,
                    ascii=True,
                    desc="Standard Batches",
                    leave=False,
                )
            ):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = standard_model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

                sgd.zero_grad()
                loss.backward()
                sgd.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    except KeyboardInterrupt:
        tqdm.write(
            "[!] Standard training interrupted. Evaluating the best individual..."
        )
    except Exception as e:
        tqdm.write(f"[!] An error occurred during standard training: {e}")

    # Compute accuracy on the validation set for the standard model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = standard_model(inputs)
            _, predicted = torch.max(output, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    tqdm.write(f"[i] SGD Validation Accuracy: {accuracy:.4f} (higher is better)")


if __name__ == "__main__":
    main()
