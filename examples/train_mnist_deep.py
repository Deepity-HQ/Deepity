import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pydeepity import SimplePCN
from time import perf_counter

from rich.console import Console
from rich.progress import track

console = Console()

def load_full_mnist():
    console.print("[bold cyan]Fetching full MNIST dataset...[/bold cyan]")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = (X.astype(np.float32) / 127.5) - 1.0
    y = y.astype(int)

    Y_bipolar = np.full((y.shape[0], 10), -0.9, dtype=np.float32)
    Y_bipolar[np.arange(y.shape[0]), y] = 0.9

    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
        X, Y_bipolar, y, train_size=60000, test_size=10000, stratify=y, random_state=42
    )
    return X_train, Y_train, X_test, y_test_labels, y_train_labels


def train_and_evaluate(X_train, Y_train, y_train_labels, X_test, y_test_labels,
                        train_steps, test_steps, lr, epochs, batch_size):

    net = SimplePCN(batch_size=batch_size)
    
    # 3 Hidden Layers (784 -> 512 -> 256 -> 128 -> 10) with cooled-down ir=0.02
    net.add_layer(784, 512, lr=lr, ir=0.02, act="tanh", lmbda=0.0001)
    net.add_layer(512, 256, lr=lr, ir=0.02, act="tanh", lmbda=0.0001)
    net.add_layer(256, 128, lr=lr, ir=0.02, act="tanh", lmbda=0.0001)
    net.add_layer(128,  10, lr=lr, ir=0.02, act="tanh", lmbda=0.0001)
    net.add_layer( 10,   0, lr=lr, ir=0.02, act="linear", lmbda=0.0001)

    O_: str = "ADAMW"
    net.set_optimizer(O_)
    console.print(f"[bold blue]Set optimizer to {O_}[/bold blue]")

    net.compile()
    net.randomize_weights()
    
    net.summary()

    # Drop the base learning rate significantly for deep ADAMW
    adjusted_lr = lr / 1000.0

    start_time = perf_counter()

    # Train with I_avg caching and decay_rate=0.95 to prevent late-stage explosions
    net.fit_iavg(
        X=X_train, 
        Y=Y_train, 
        labels=y_train_labels,
        epochs=epochs,
        steps=train_steps,
        per_class=batch_size // 10,
        num_classes=10,
        hidden_layer_index=1,
        hidden_size=512,
        initial_lr=adjusted_lr,
        decay_rate=0.95,
        reset_cache_per_epoch=True
    )

    train_time = perf_counter() - start_time
    console.print(f"[bold green]Training complete in {train_time:.1f}s.[/bold green]\n")

    console.print(f"[bold yellow]Evaluating on held-out test set with {test_steps} inference steps...[/bold yellow]")
    correct, total = 0, 0
    
    for i in track(range(0, len(X_test), batch_size), description="[cyan]Evaluating..."):
        X_batch = X_test[i:i + batch_size]
        y_labels_batch = y_test_labels[i:i + batch_size]
        if len(X_batch) != batch_size: continue

        # Use the longer zero-init test steps to let the network settle fully
        pred_classes = np.argmax(net.predict(X_batch, steps=test_steps), axis=1)
        correct += np.sum(pred_classes == y_labels_batch)
        total += batch_size

    console.print(f"\n[bold magenta]Test Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)[/bold magenta]")

if __name__ == "__main__":
    X_train, Y_train, X_test, y_test_labels, y_train_labels = load_full_mnist()
    
    # Decouple the steps: 6 for lightning-fast training, 60 for accurate prediction
    train_and_evaluate(
        X_train, Y_train, y_train_labels, X_test, y_test_labels,
        train_steps=60,       
        test_steps=60,       
        lr=0.06, 
        epochs=50, 
        batch_size=250
    )
