import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import lstsq

# Hilbert transform of Bethe lattice DOS
def bethe_gf(wn, sigma, mu, D):
    zeta = 1.j * wn + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag * wn)
    return 2. / (zeta + sig * sq)

# Fourier transform
def ft(wn, g_tau, tau):
    exp = np.exp(1.j * np.outer(wn, tau))
    return np.dot(exp, g_tau)

# Inverse Fourier transform
def ift(wn, g_wn, tau, beta):
    exp = np.exp(-1.j * np.outer(tau, wn))
    return 1/beta * np.dot(exp, g_wn)

def freq_tail_fourier(tail_coef, beta, tau, wn):
    # The Fourier transforms use as tail expansion of the atomic limit self-enegy
    # \Sigma(i\omega_n\rightarrow \infty) = \frac{U^2}{4(i\omega_n)} 
    
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]_

    +------------------------+-----------------------------------------+
    | :math:`g(iw)`          | :math:`g(t)`                            |
    +========================+=========================================+
    | :math:`(i\omega)^{-1}` | :math:`-\frac{1}{2}`                    |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-2}` | :math:`\frac{1}{2}(\tau-\beta/2)`       |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-3}` | :math:`-\frac{1}{4}(\tau^2 -\beta\tau)` |
    +------------------------+-----------------------------------------+

    See also
    --------
    gw_invfouriertrans
    gt_fouriertrans

    """

    freq_tail = tail_coef[0] / (1.j * wn)\
        + tail_coef[1] / (1.j * wn)**2\
        + tail_coef[2] / (1.j * wn)**3

    time_tail = - tail_coef[0] / 2 \
        + tail_coef[1] / 2 * (tau - beta / 2) \
                - tail_coef[2] / 4 * (tau**2 - beta * tau)

    return freq_tail, time_tail
    
def gt_fouriertrans(g_tau, tau, wn, tail_coef=(1., 0., 0.)):
    r"""Performs a forward fourier transform for the interacting Green function
    in which only the interval :math:`[0,\beta)` is required and output given
    into positive fermionic matsubara frequencies up to the given cutoff.
    Time array is twice as dense as frequency array

    .. math:: g(i\omega_n) = \int_0^\beta g(\tau)
       e^{i\omega_n \tau} d\tau

    Parameters
    ----------
    g_tau : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    wn : real float array
            fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    freq_tail_fourier
    gt_fouriertrans"""

    beta = tau[1] + tau[-1] # Un-comment if tau doesnt include beta 
    #beta = tau[-1]
        
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, wn)
    gtau = g_tau.real - time_tail  
    gt = beta * ifft(gtau * np.exp(1j * np.pi * tau / beta))[..., :len(wn)]   
    return np.append(gt, np.zeros(len(freq_tail)-len(gt))) + freq_tail    


def gw_invfouriertrans(g_wn, tau, wn, tail_coef=(1., 0., 0.)):
    r"""Performs an inverse fourier transform of the green Function in which
    only the imaginary positive matsubara frequencies
    :math:`\omega_n= \pi(2n+1)/\beta` with :math:`n \in \mathbb{N}` are used.
    The high frequency tails are transformed analytically up to the third moment.

    Output is the real valued positivite imaginary time green function.
    For the positive time output :math:`\tau \in [0;\beta)`.
    Array sizes need not match between frequencies and times, but a time array
    twice as dense is recommended for best performance of the Fast Fourrier
    transform.

    .. math::
       g(\tau) &= \frac{1}{\beta} \sum_{\omega_n}
                   g(i\omega_n)e^{-i\omega_n \tau} \\
       &= \frac{1}{\beta} \sum_{\omega_n}\left( g(i\omega_n)
          -\frac{1}{i\omega_n}\right) e^{-i\omega_n \tau} +
          \frac{1}{\beta} \sum_{\omega_n}\frac{1}{i\omega_n}e^{-i\omega_n \tau} \\

    Parameters
    ----------
    g_wn : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    wn : real float array
            fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails


    Returns
    -------
    complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    gt_fouriertrans
    freq_tail_fourier
    """

    beta = tau[1] + tau[-1] # Un-comment if tau doesnt include beta 
    #beta = tau[-1]
        
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, wn)
    giwn = g_wn - freq_tail
    g_tau = fft(giwn, len(tau)) * np.exp(-1j * np.pi * tau / beta)   
    return (g_tau * 2 / beta).real + time_tail
    
    '''
    gt = np.array([])
    for t in tau:
        gt = np.append(gt, np.dot(g_wn, np.exp(-1.j*wn*t)))
    return 1/beta * gt
    '''

def tail(wn, coef, powers):
    return np.sum([c / wn**p for c, p in zip(coef, powers)], 0)

def lin_tail_fit(wn, inp_gf, x, span=30, negative_freq=False):
    """Perform a Least squares fit to the tail of Green function

    the fit is done in inp_gf[-x:-x + span]

    Parameters:
        wn (real 1D ndarray) : Matsubara frequencies
        inp_gf (complex 1D ndarray) : Green function to fit
        x (int) : counting from last element from where to do the fit
        span (int) : amount of frequencies to do the fit over
        negative_freq (bool) : Array has negative Matsubara frequencies
    Returns:
        complex 1D ndarray : Tail patched Green function (copy of origal)
        real 1D ndarray : The First 3 moments
"""
    def tail_coef(wn, data, powers):
        A = np.array([1 / wn**p for p in powers]).T
        return lstsq(A, data)[0]

    tw_n = wn[-x:-x + span].copy()
    datar = inp_gf[-x:-x + span].real.copy()
    datai = inp_gf[-x:-x + span].imag.copy()

    re_c = tail_coef(tw_n, datar, [2])
    re_tail = tail(wn, re_c, [2])
    im_c = tail_coef(tw_n, datai, [1, 3])
    im_tail = tail(wn, im_c, [1, 3])

    f_tail = re_tail + 1j * im_tail

    patgf = inp_gf.copy()
    patgf[-x:] = f_tail[-x:]
    if negative_freq:
        patgf[:x] = f_tail[:x]

    return patgf, np.array([im_c[0], re_c[0], im_c[1]])


def fit_gf(wn, giw, p=2):
    """Performs a quadratic fit of the *first's* matsubara frequencies
    to estimate the value at zero energy.

    Parameters
    ----------
    wn : real float array
            First's matsubara frequencies to fit
    giw : real array
            Function to fit

    Returns
    -------
    Callable for inter - extrapolate function
    """
    gfit = np.squeeze(giw)[:len(wn)]
    pf = np.polyfit(wn, gfit, p)
    return np.poly1d(pf)
