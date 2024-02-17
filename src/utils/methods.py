"""Compared methods.
"""
import numpy as np

from math import atan2, cos, sin
from typing import List
from sklearn.linear_model import RANSACRegressor


class KellnerMethod:

    def __init__(self, b, l, beta, trials=100, threshold=0.1) -> None:
        self.b = b
        self.l = l
        self.beta = beta
        self.RANSACRegressor = RANSACRegressor(min_samples=2, residual_threshold=threshold, max_trials=trials, random_state=42)

    def _ransac(self, vr:np.array, theta:np.array) -> List[float]:
        y = vr.reshape(-1,1)
        Ai = np.cos(theta).reshape(-1,1)
        Aj = np.sin(theta).reshape(-1,1)
        A = np.hstack((Ai,Aj))
        self.RANSACRegressor.fit(A, y)
        return None 
    
    def _lstsq(self, vr:np.array, theta:np.array) -> List[float]:
        y = vr.reshape(-1,1)
        Ai = np.cos(theta).reshape(-1,1)
        Aj = np.sin(theta).reshape(-1,1)
        A = np.hstack((Ai,Aj))
        v, *_ = np.linalg.lstsq(A, vr, rcond=None)
        return v 
    
    def _compute_vel_profile(self, vx, vy):
        vs = -(vx**2 + vy**2)**0.5
        alpha = atan2(vy, vx)
        return vs, alpha
    
    def estimate(self, vr:np.array, theta:np.array) -> List[float]:
        # Solve velocity profile using RANSAC
        self._ransac(vr, theta)
        inliers_mask = self.RANSACRegressor.inlier_mask_

        # Fine-tuning model with inliers
        vx, vy = self._lstsq(vr[inliers_mask], theta[inliers_mask])

        # Computing ego-motion
        vs, alpha = self._compute_vel_profile(vx, vy)            
        v = (cos(alpha + self.beta) - (self.b / self.l) * sin(alpha + self.beta)) * vs
        omega = (sin(alpha + self.beta) / self.l) * vs

        return v, omega