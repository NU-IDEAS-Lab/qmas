import torch

@torch.no_grad
class Predictor:
    ''' This class implements a KF-based predictor.
        Based on https://doi.org/10.1109/TAC.2004.834121 '''

    def __init__(self, obs_dim, action_dim, args, device=None):
        self.args = args
        self.device = device

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.transition_dim = obs_dim + action_dim
        self.prediction_horizon = args.prediction_history_window

        if self.prediction_horizon != 1:
            raise ValueError("Prediction horizon must be 1 for the Kalman Filter predictor.")

        # Initialize the Kalman Filter parameters.
        dt = 1.0
        self.A = torch.tensor(
            data=[
                [1, 0, dt, 0, 0, 0, 0, 0, 0], #l0p0
                [0, 1, 0, dt, 0, 0, 0, 0, 0], #l0p1
                [0, 0, 1, 0, 0, 0, 0, 0, 0],  #l0v0
                [0, 0, 0, 1, 0, 0, 0, 0, 0],  #l0v1
                [0, 0, 0, 0, 1, 0, dt, 0, 0], #f1p0
                [0, 0, 0, 0, 0, 1, 0, dt, 0], #f1p1
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  #f1v0
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  #f1v1
                [0, 0, 0, 0, 0, 0, 0, 0, 1],  #id
            ],
            dtype=torch.float32,
            device=device
        )
        self.B = torch.tensor(
            data=[
                [0, 0], #l0p0
                [0, 0], #l0p1
                [0, 0], #l0v0
                [0, 0], #l0v1
                [0, 0], #f0p0
                [0, 0], #f0p1
                [1.0, 0], #f0v0
                [0, 1.0], #f0v1
                [0, 0], #id
            ],
            dtype=torch.float32,
            device=device
        )
        self.C = torch.eye(obs_dim, device=device)
        self.Q = torch.eye(obs_dim, device=device) * 0.1 #args.kf_process_noise
        self.R = torch.eye(obs_dim, device=device) * 0.0 #args.kf_measurement_noise

        print(f"Lambda critical lower bound: {self.get_lower_bound_lambda_critical()}")
        print(f"Lambda critical upper bound: {self.get_upper_bound_lambda_critical()}")

        self.kf = None

    def initialize(self, x0, P0):
        ''' Initialize the Kalman Filter. '''

        self.kf = KalmanFilterIntermittenObservations(self.A, self.B, self.C, self.Q, self.R, x0, P0)


    def get_lower_bound_lambda_critical(self):
        ''' Get the lower bound on the critical observation probability for stability.
            Based on Sinopoli et al. (2004) - https://doi.org/10.1109/TAC.2004.834121
        '''

        eigvals = torch.linalg.eigvals(self.A)
        lambda_critical = 1.0 - 1.0 / torch.max(torch.abs(eigvals)) ** 2
        return lambda_critical.item()


    def get_upper_bound_lambda_critical(self):
        ''' Get the upper bound on the critical observation probability for stability.
            Based on Sinopoli et al. (2004) - https://doi.org/10.1109/TAC.2004.834121
        '''

        special_case = False
        try:
            C_inv = torch.linalg.inv(self.C)
            special_case = True
        except RuntimeError:
            pass

        # This is the special case where C is invertible.
        # Detailed in Section IV of Sinopoli et al. (2004).
        if special_case:
            return self.get_lower_bound_lambda_critical()
        
        # General case - not implemented.
        else:
            def psi(Y, Z, lam):
                zeros = torch.zeros_like(Y)
                sqrt_lam = torch.sqrt(lam)
                sqrt_one_minus_lam = torch.sqrt(1.0 - lam)

                return torch.tensor([
                    [Y,                                             sqrt_lam * (Y @ self.A + Z @ self.C),   sqrt_one_minus_lam * (Y @ self.A)],
                    [sqrt_lam * (self.A.T @ Y + self.C.T @ Z.T),    Y,                                      zeros],
                    [sqrt_one_minus_lam * (self.A.T @ Y),            zeros,                                 Y]
                ], dtype=torch.float32, device=self.device)
            
            raise NotImplementedError("Upper bound calculation for the general case is not implemented.")


    def get_prediction(self, trajectory, visibility_mask, prediction_prev=None):
        ''' Get a prediction from the KF.
            Args:
                trajectory: A tensor of shape (T, D), where T is the trajectory length and D is the transition dimension (action + observation).
                visibility_mask: A tensor of shape (T, D) indicating which parts of the observation are visible.
                prediction_prev: A tensor of shape (1, T, D) for autoregression.
            Returns:
                prediction: A tensor of shape (1, T, D) containing the predicted trajectory.
        '''

        # Apply the visibility mask to the trajectory.
        trajectory = trajectory * visibility_mask

        # Check whether the filter is initialized.
        if self.kf is None:
            self.initialize(
                x0=trajectory[-1, self.action_dim:],
                P0=torch.eye(self.obs_dim, device=self.device)
            )

        # Get actions.
        actions = trajectory[-1, :self.action_dim]

        # Get prediction.
        prediction = torch.zeros((1, self.prediction_horizon, self.transition_dim), device=self.device)
        prediction[0, 0, self.action_dim:] = self.kf.predict(actions)

        # Set up a diagonal visibility mask for the observation.
        visibility_mask_diag = torch.diag(visibility_mask[-1, self.action_dim:])

        # Update the Kalman Filter with the observation.
        self.kf.update(trajectory[-1, self.action_dim:], visibility_mask_diag)

        return prediction


class KalmanFilter:
    ''' This class implements a Kalman Filter for trajectory prediction.
        Implementation is based on: https://www.geeksforgeeks.org/python/kalman-filter-in-python/ '''
    
    @torch.no_grad
    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A = A  # State transition matrix
        self.B = B  # Control input matrix
        self.C = C  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0 # Initial state estimate
        self.P = P0 # Initial estimate covariance
    

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x


    def update(self, z):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ torch.linalg.inv(S)
        y = z - self.C @ self.x
        self.x = self.x + K @ y
        I = torch.eye(self.P.shape[0], device=self.P.device)
        self.P = (I - K @ self.C) @ self.P
        return self.x


class KalmanFilterIntermittenObservations(KalmanFilter):
    ''' This class implements a Kalman Filter for trajectory prediction with intermittent observations.
        It inherits from the KalmanFilter class.
        It is based on work by Sinopoli et al. (2004) - https://doi.org/10.1109/TAC.2004.834121
    '''

    def update(self, z, gamma=None):
        """
        Measurement update with gamma in [0, 1].
        gamma=0 -> no update (no measurement available at t)
        gamma=1 -> standard KF update
        """

        if gamma is None:
            gamma = torch.eye(z.shape[0], dtype=z.dtype, device=z.device)

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ torch.linalg.inv(S)
        y = z - self.C @ self.x
        self.x = self.x + K @ (gamma @ y)
        self.P = self.P - K @ gamma @ self.C @ self.P
        return self.x