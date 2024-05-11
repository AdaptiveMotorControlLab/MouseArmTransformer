import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import helper
import transformer


def run_training(seq_length, cutoff, loss_weight_mse, loss_weight_continuity, loss_weight_connectivity,
                 smoothing_window):
    # Load data
    full_X_c1, full_X_c2, full_y = helper.load_data(cutoff=cutoff, reload=False)
    X_3d_test, X_2d_c1_test, X_2d_c2_test, cam_positions, likelihood_c1, likelihood_c2 = helper.load_test_data()
    all_aggregated_tensor, relative_displacements = helper.get_relative_displacements(cam_positions)

    train_dataset = transformer.PositionDataset(full_X_c1, full_X_c2, full_y, seq_length=seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = transformer.PositionDataset(X_2d_c1_test, X_2d_c2_test, X_3d_test, seq_length=seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformer.Net().to(device)
    train_criterion = transformer.masked_loss
    eval_criterion = transformer.masked_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # Train the model
    loss_weights = {
        'mse': loss_weight_mse,
        'continuity': loss_weight_continuity,
        'connectivity': loss_weight_connectivity
    }
    for epoch in range(10):
        train_loss = transformer.train(model, train_dataloader, device, train_criterion, optimizer, loss_weights,
                                       relative_displacements, all_aggregated_tensor)
        print(f"Epoch {epoch+1} training loss: {train_loss:.5f}")
        if epoch % 1 == 0:
            test_loss, test_preds = transformer.evaluate(model, test_dataloader, device, eval_criterion)
            print(f"Epoch {epoch+1} test loss: {test_loss:.5f}")

    run_name = helper.save_video(
        test_preds,
        cam_positions, {
            'epoch': epoch,
            'loss': test_loss,
            'seq_length': seq_length,
            'cutoff': cutoff,
            'smoothing': smoothing_window,
            'loss_weights': str([loss_weight_mse, loss_weight_continuity, loss_weight_connectivity])
        },
        smoothing_window=smoothing_window)

    # Also save pytorch model weights
    torch.save(model.state_dict(), f'/data/markus/mausspaun/3D/weights_{run_name}.pt')

    print('Finished Training')


if __name__ == '__main__':
    # Parse seq_length, cutoff, loss_weight_mse, loss_weight_continuity, loss_weight_connectivity
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=7)
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--cutoff', type=float, default=0.99)
    parser.add_argument('--loss_weight_mse', type=float, default=1)
    parser.add_argument('--loss_weight_continuity', type=float, default=1)
    parser.add_argument('--loss_weight_connectivity', type=float, default=1)
    args = parser.parse_args()

    run_training(args.seq_length, args.cutoff, args.loss_weight_mse, args.loss_weight_continuity,
                 args.loss_weight_connectivity, args.smoothing)
