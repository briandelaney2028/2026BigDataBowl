import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os


class FeatureScaler:
    """
    Robust per-feature scaler that handles NaNs and angle encoding.

    This helper fits per-feature scalers (StandardScaler or MinMaxScaler)
    on valid (non-NaN) values and applies transformations while preserving
    NaNs. Spatial features (x/y coordinates) are forced to MinMax scaling.

    Attributes
    ----------
    feature_names : list
        List of feature names in the input arrays.
    method : str
        Scaling method for non-spatial features: 'standard' or 'minmax'.
    angle_features : list
        Features to be encoded as sin/cos after scaling.
    nan_threshold : float
        Fraction of NaNs above which a warning will be printed during fit.
    scalers : dict
        Mapping from feature name to fitted sklearn scaler.
    """

    def __init__(self, feature_names, method='standard', angle_features=None, nan_threshold=0.8):
        """
        Initialize the FeatureScaler.

        Parameters
        ----------
        feature_names : list
            Ordered list of feature names expected in input arrays.
        method : str, optional
            'standard' to use StandardScaler for non-spatial features,
            or 'minmax' to use MinMaxScaler (default: 'standard').
        angle_features : list or None, optional
            Names of angle features to encode as sine/cosine after scaling.
        nan_threshold : float, optional
            Fraction of NaN values above which a warning is emitted (default 0.8).
        """
        self.feature_names = feature_names
        self.method = method
        self.angle_features = angle_features if angle_features else []
        self.nan_threshold = nan_threshold
        self.scalers = {}
        self.fitted = False

        # Store input structure
        self.B = None
        self.N = None
        self.T = None
        self.F = None

        self.spatial_features = {'x', 'y', 'ball_land_x', 'ball_land_y'}
        self.output_feature_names = self._generate_output_features()

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _generate_output_features(self):
        """
        Generate output feature list by replacing angle features with sin/cos pairs.

        Returns
        -------
        list
            Copy of feature names with each angle feature replaced by
            '<name>_sin' and '<name>_cos'.
        """
        f_names = self.feature_names.copy()
        idx = 0
        while idx < len(f_names):
            if f_names[idx] in self.angle_features:
                base = f_names[idx]
                f_names[idx:idx+1] = [f"{base}_sin", f"{base}_cos"]
            idx += 1
        return f_names

    def _array_to_df(self, X, feature_cols=None):
        """
        Flatten a 4D array into a DataFrame for per-column scaling operations.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (B, N, T, F).
        feature_cols : list or None
            Optional list of column names corresponding to the last dimension.

        Returns
        -------
        tuple
            (df, B, N, T, F) where df is a DataFrame with shape (B*N*T, F).
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Expected ndarray input.")
        B, N, T, F = X.shape
        cols = feature_cols or self.feature_names
        if len(cols) != F:
            raise ValueError(f"Shape mismatch: array has {F} features but {len(cols)} names were provided.")
        flat = X.reshape(-1, F)
        return pd.DataFrame(flat, columns=cols), B, N, T, F

    def _df_to_array(self, df, B, N, T):
        """
        Convert a flattened DataFrame back to a 4D numpy array.

        Parameters
        ----------
        df : pd.DataFrame
            Flattened DataFrame with columns matching feature_names.
        B, N, T : int
            Batch, player, and time dimensions to reshape into.

        Returns
        -------
        np.ndarray
            Array of shape (B, N, T, F).
        """
        return df.to_numpy().reshape(B, N, T, -1)

    def _fit_column(self, values, col):
        """
        Fit a sklearn scaler on the valid (non-NaN) entries of a single column.

        Parameters
        ----------
        values : np.ndarray
            1-D array of column values (may contain NaNs).
        col : str
            Column name (used to choose spatial scaling and for warnings).

        Returns
        -------
        scaler or None
            Fitted scaler instance, or None if no valid values exist.
        """
        mask = ~np.isnan(values)
        valid_vals = values[mask].reshape(-1, 1)
        frac_nan = 1 - (len(valid_vals) / len(values))
        if frac_nan > self.nan_threshold:
            print(f"[Warning] Column '{col}' has {frac_nan*100:.1f}% NaNs — scaling may be unreliable.")

        if len(valid_vals) == 0:
            return None

        # Force spatial features to always use MinMaxScaler
        if col in self.spatial_features:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler() if self.method == 'standard' else MinMaxScaler()
        scaler.fit(valid_vals)
        return scaler

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def fit(self, X: np.ndarray, feature_cols=None):
        """
        Fit per-feature scalers using the provided 4D array.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (B, N, T, F) used to fit scalers.
        feature_cols : list or None
            Subset of columns to fit. If None, fits all known features.

        Returns
        -------
        None
        """
        df, self.B, self.N, self.T, self.F = self._array_to_df(X, self.feature_names)
        cols_to_fit = feature_cols or df.columns.tolist()

        for col in cols_to_fit:
            if col in self.angle_features:
                continue
            scaler = self._fit_column(df[col].values, col)
            if scaler is not None:
                self.scalers[col] = scaler
        self.fitted = True

    def transform(self, data, feature_cols=None):
        """
        Scale a 4D data array using previously fitted per-feature scalers.

        Parameters
        ----------
        data : np.ndarray
            4D array with shape (B, N, T, F) to transform.
        feature_cols : list or None
            Columns to scale; if None, uses all fitted scalers.

        Returns
        -------
        np.ndarray
            Scaled array with the same shape as input (NaNs preserved).
        """
        assert self.fitted, "Scaler not fitted"
        assert len(data.shape) == 4, "Expected 4D array"

        B, N, T, F = data.shape
        df_cols = self.feature_names if F == len(self.feature_names) else feature_cols
        df, B, N, T, F = self._array_to_df(data, df_cols)
        df_scaled = df.copy()

        cols_to_iterate = feature_cols or list(self.scalers.keys())
        for col in cols_to_iterate:
            if col not in df_scaled.columns or col not in self.scalers:
                continue
            scaler = self.scalers[col]
            vals = df_scaled[col].values.reshape(-1, 1)
            if np.isnan(vals).any():
                mask = ~np.isnan(vals).flatten()
                transformed = np.full_like(vals, np.nan, dtype=np.float32)
                transformed[mask] = scaler.transform(vals[mask].reshape(-1, 1))
            else:
                transformed = scaler.transform(vals)
            df_scaled[col] = transformed

        # Handle angle encoding AFTER scaling
        angle_cols = set(self.angle_features).intersection(df_scaled.columns)
        for col in angle_cols:
            rads = np.deg2rad(df_scaled[col].values)
            df_scaled[f'{col}_sin'] = np.sin(rads)
            df_scaled[f'{col}_cos'] = np.cos(rads)
            df_scaled.drop(columns=[col], inplace=True)

        return self._df_to_array(df_scaled, B, N, T)

    def fit_transform(self, X: np.ndarray, feature_cols=None):
        """
        Convenience method: fit the scalers on X and return the transformed array.

        Parameters
        ----------
        X : np.ndarray
            Input array to fit and transform.
        feature_cols : list or None
            Columns to fit/transform.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        self.fit(X, feature_cols)
        return self.transform(X, feature_cols)

    def inverse_transform(self, X_scaled: np.ndarray, feature_cols=None):
        """
        Reverse the scaling operation, restoring original value ranges.

        Parameters
        ----------
        X_scaled : np.ndarray
            Scaled data array (B, N, T, F) or a subset matching feature_cols.
        feature_cols : list or None
            Feature subset corresponding to columns present in X_scaled.

        Returns
        -------
        np.ndarray
            Inverse-transformed array with same shape as input.
        """
        assert self.fitted, "Scaler not fitted"
        df, B, N, T, F = self._array_to_df(X_scaled, feature_cols or self.feature_names)
        df_inv = df.copy()

        for col, scaler in self.scalers.items():
            if col not in df_inv.columns:
                continue
            vals = df_inv[col].values.reshape(-1, 1)
            if np.isnan(vals).any():
                mask = ~np.isnan(vals).flatten()
                restored = np.full_like(vals, np.nan, dtype=np.float32)
                restored[mask] = scaler.inverse_transform(vals[mask])
            else:
                restored = scaler.inverse_transform(vals)
            df_inv[col] = restored

        # Decode sin/cos back to angles if present
        angle_cols = set(self.angle_features).intersection(df_inv.columns)
        for col in angle_cols:
            sin_col, cos_col = f'{col}_sin', f'{col}_cos'
            if sin_col in df_inv.columns and cos_col in df_inv.columns:
                rads = np.arctan2(df_inv[sin_col], df_inv[cos_col])
                df_inv[col] = np.rad2deg(rads)
                df_inv.drop(columns=[sin_col, cos_col], inplace=True)

        return self._df_to_array(df_inv, B, N, T)
    
    def save(self, fpath):
        """
        Save the fitted FeatureScaler

        Parameters:
            fpath (str): path to save scaler
        """

        state = {
            'feature_names': self.feature_names,
            'method': self.method,
            'angle_features': self.angle_features,
            'nan_threshold': self.nan_threshold,
            'scalers': self.scalers,
            'fitted': self.fitted,
            'B': self.B,
            'N': self.N,
            'T': self.T,
            'F': self.F,
            'spatial_features': self.spatial_features,
            'output_feature_names': self.output_feature_names,
        }
        joblib.dump(state, fpath)
        print(f"[FeatureScaler] Saved to {os.path.abspath(fpath)}")

    @classmethod
    def load(cls, fpath):
        """
        Load a previously saved FeatureScaler

        Parameters:
            fpath (str): path to saved scaler

        Returns:
            Resotred FeatureScaler instance
        """
        state = joblib.load(fpath)
        scaler = cls(
            feature_names=state['feature_names'],
            method=state['method'],
            angle_features=state['angle_features'],
            nan_threshold=state['nan_threshold'],
        )
        scaler.scalers = state['scalers']
        scaler.fitted = state['fitted']
        scaler.B = state['B']
        scaler.N = state['N']
        scaler.T = state['T']
        scaler.F = state['F']
        scaler.spatial_features = state['spatial_features']
        scaler.output_feature_names = state['output_feature_names']
        print(f"[FeatureScaler] Loaded from {os.path.abspath(fpath)}")
        return scaler