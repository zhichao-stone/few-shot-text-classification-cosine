import time
import logging
import torch

from config import Config
from util import build_optimizer, EMA, evaluate
from data_helper import create_dataloaders
from model import CCFModel

def train(model, train_loader, dev_loader, config):
    # build optimizer
    optimizer, scheduler = build_optimizer(model, config)
    # ema
    ema = EMA(model, 0.999)
    ema.register()
    # training
    step = 0
    best_score = config.best_score
    start_time = time.time()
    num_total_steps = len(train_loader) * config.max_epochs

    for epoch in range(config.max_epochs):
        model.train()
        for batch in train_loader:
            loss, acc, _, _ = model(batch)
            loss = loss.mean()
            acc = acc.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()
            
            step += 1
            if step % config.print_steps == 0:
                time_per_step = (time.time() - start_time) / step
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch + 1} step {step} eta {remaining_time}: loss {loss:.4f}, accuracy {acc:.4f}")

        # validation
        ema.apply_shadow()
        dev_loss, dev_f1_macro, dev_f1_micro = evaluate(model, dev_loader)
        logging.info(f'Epoch {epoch + 1} step {step}: loss {dev_loss:.4f}, macro f1 {dev_f1_macro:.4f}, micro f1 {dev_f1_micro:.4f}')
        if dev_f1_macro > best_score:
            best_score = dev_f1_macro
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'f1_macro': dev_f1_macro},
                f'{config.model_dir}/model_best.pkl'
            )
        ema.restore()

    
if __name__ == '__main__':
    config = Config()
    train_loader, dev_loader = create_dataloaders(config)
    model = CCFModel(config, config.method)
    if config.cuda >= 0:
        model = torch.nn.parallel.DataParallel(model.to(config.device))

    logging.info(f'Training Configs: \n{config}')
    train(model, train_loader, dev_loader, config)
