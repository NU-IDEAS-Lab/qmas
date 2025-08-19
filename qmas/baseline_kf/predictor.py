import torch
import numpy as np

class Predictor(torch.nn.Module):
    ''' This class implements a KF-based predictor.
        Based on https://doi.org/10.1109/TAC.2004.834121 '''

    def __init__(self, obs_dim, action_dim, args, device=None):
        super(Predictor, self).__init__()

        transition_dim = obs_dim + action_dim
        self.prediction_horizon = args.diffusion_horizon


    def get_prediction(self, trajectory, visibility_mask=None, prediction_prev=None):
        ''' Get a prediction from the KF.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
                visibility_mask: A tensor of shape (T, D) indicating which parts of the observation are visible.
                prediction_prev: A tensor of shape (1, T, D) for autoregression.
            Returns:
                prediction: A tensor of shape (1, T, D) containing the predicted trajectory.
        '''

        # Set up trajectory and visibility mask.
        trajectory = trajectory.unsqueeze(0)  # Add sample dimension.
        if visibility_mask == None:
            visibility_mask = torch.ones_like(trajectory)  # Default to all visible.
        else:
            visibility_mask = visibility_mask.unsqueeze(0) # Add sample dimension.

        # Apply the visibility mask to the trajectory.
        trajectory = trajectory * visibility_mask

        raise NotImplementedError("KF prediction not implemented yet.")

        return prediction


class KalmanFilter:
    ''' This class implements a Kalman Filter for trajectory prediction.
        Implementation is based on: https://www.geeksforgeeks.org/python/kalman-filter-in-python/ '''
    
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
    

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x


    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x