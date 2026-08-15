from crystallite.backend import xp


class Grid:
    """Uniform real-space grid with matching Fourier-space wavenumbers.

    Axes with a single grid point are treated as inactive and are not
    transformed. The last active axis uses the real-FFT convention.

    Parameters
    ----------
    shape : tuple of int, optional
        Grid points along each of the 3 axes.
    lengths : tuple of float, optional
        Physical size along each axis.

    Attributes
    ----------
    real_dtype, complex_dtype : dtype
        Floating-point types used for real- and Fourier-space arrays.
    fft_axes : tuple of int
        Negative-indexed axes with more than one grid point.
    fft_shape : tuple of int
        Grid shape along `fft_axes`.
    spacing : tuple of float
        Real-space spacing along each axis.
    x : tuple of ndarray
        Real-space coordinates, broadcastable against the grid shape.
    k : tuple of ndarray
        Angular wavenumbers (rfftfreq on the last active axis, fftfreq
        elsewhere), broadcastable against the grid shape.
    k2 : ndarray
        Squared wavenumber magnitude, ``k[0]**2 + k[1]**2 + k[2]**2``.
    safe_k2 : ndarray
        `k2` with the k=0 entry replaced by 1, to divide by safely.
    inv_k2 : ndarray
        Elementwise reciprocal of `k2`, with the k=0 entry set to 0.
    """

    def __init__(self, shape=(256, 256, 1), lengths=(1.0, 1.0, 1.0)):

        self.real_dtype = xp.float32
        self.complex_dtype = xp.complex64

        assert len(shape) == 3, "shape: {}".format(shape)
        assert len(lengths) == 3, "lengths: {}".format(lengths)

        assert all(n > 0 for n in shape), "shape: {}".format(shape)
        assert all(l > 0 for l in lengths), "lengths: {}".format(lengths)

        self.fft_axes = tuple(
            axis - 3 for axis, n in enumerate(shape) if n > 1
        )
        self.fft_shape = tuple(shape[i] for i in self.fft_axes)

        rfft_axis = self.fft_axes[-1]

        self.spacing = tuple(l / n for n, l in zip(shape, lengths))

        xs, ks = [], []

        for axis, (n, l, dx) in enumerate(zip(shape, lengths, self.spacing)):
            dims = [1] * len(shape)
            dims[axis] = -1

            # Real-space coordinates
            x = xp.arange(n, dtype=self.real_dtype) * dx
            xs.append(x.reshape(dims))

            # Reciprocal-space coordinates
            if n == 1:
                k = xp.zeros(1, dtype=self.real_dtype)
            elif axis - len(shape) == rfft_axis:
                k = 2 * xp.pi * xp.fft.rfftfreq(n, d=dx)
            else:
                k = 2 * xp.pi * xp.fft.fftfreq(n, d=dx)

            ks.append(k.astype(self.real_dtype).reshape(dims))

        self.x = tuple(xs)
        self.k = tuple(ks)

        self.k2 = self.k[0] ** 2 + self.k[1] ** 2 + self.k[2] ** 2

        self.safe_k2 = xp.where(self.k2 == 0, 1.0, self.k2)
        self.inv_k2 = xp.where(self.k2 == 0, 0.0, 1.0 / self.safe_k2)

    def fft(self, x):
        """Forward real FFT over the active axes.

        Parameters
        ----------
        x : ndarray
            Real-space field.

        Returns
        -------
        ndarray
            Fourier coefficients over `fft_axes`.
        """
        return xp.fft.rfftn(x, axes=self.fft_axes)

    def ifft(self, x):
        """Inverse real FFT over the active axes.

        Parameters
        ----------
        x : ndarray
            Fourier coefficients.

        Returns
        -------
        ndarray
            Real-space field of shape `fft_shape` over `fft_axes`.
        """
        return xp.fft.irfftn(x, s=self.fft_shape, axes=self.fft_axes)
