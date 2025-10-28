import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from config import config
from model.umia import UMIA
from data.causal_stories import CausalStoryDataset

def compute_loss(model_output, batch, config, step):
    """Compute all loss components"""
    # 1. Prediction loss
    pred = model_output['output']
    target = batch['target']
    loss_pred = nn.functional.mse_loss(pred, target)
    
    # 2. Causal graph losses
    adjacency = model_output['adjacency']
    loss_sparsity = adjacency.abs().mean()
    
    # Acyclicity (DAG constraint)
    N = adjacency.size(1)
    A_squared = adjacency * adjacency
    exp_A = torch.matrix_exp(A_squared.mean(0))  # Average over batch
    loss_dag = (torch.diagonal(exp_A).sum() - N) ** 2
    
    # Anneal DAG weight
    lambda_dag = config.lambda_graph_dag * (config.dag_anneal_rate ** (step / 1000))
    
    # 3. Energy loss (routing sparsity)
    gates = model_output['gates']
    loss_energy = gates.sum(dim=-1).mean()
    loss_budget = torch.relu(gates.sum(dim=-1) - config.routing_budget).pow(2).mean()
    
    # 4. Total loss
    loss_total = (
        loss_pred +
        config.lambda_graph_sparsity * loss_sparsity +
        lambda_dag * loss_dag +
        config.lambda_energy * (loss_energy + loss_budget)
    )
    
    return {
        'total': loss_total,
        'pred': loss_pred,
        'sparsity': loss_sparsity,
        'dag': loss_dag,
        'energy': loss_energy
    }

def train_step(model, batch, optimizer, scaler, config, step):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Move to device
    inputs = batch['input'].to(config.device)
    
    # Forward pass with mixed precision
    with autocast(enabled=config.use_amp):
        outputs = model(inputs)
        losses = compute_loss(outputs, batch, config, step)
    
    # Backward pass
    scaler.scale(losses['total']).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    
    # Update temperature
    model.router.update_temperature(config.gumbel_temp_decay)
    
    return {k: v.item() for k, v in losses.items()}

def main():
    # Initialize
    print("Initializing UMIA...")
    model = UMIA(config).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Data
    print("Generating dataset...")
    train_dataset = CausalStoryDataset(
        num_samples=50000,
        seq_length=64,
        vocab_size=config.input_dim,
        max_entities=config.num_slots // 4
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training loop
    print("Starting training...")
    wandb.init(project="umia", config=vars(config))
    
    step = 0
    for epoch in range(100):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            losses = train_step(model, batch, optimizer, scaler, config, step)
            
            if step % config.log_interval == 0:
                wandb.log(losses, step=step)
                pbar.set_postfix(losses)
            
            if step % config.save_interval == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step
                }, f'checkpoints/umia_step_{step}.pt')
            
            step += 1
    
    print("Training complete!")

if __name__ == '__main__':
    main()
