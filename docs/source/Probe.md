# Probe
## Focal Plane Wavefront Sensing: Pairwise Probing (PWP)

### Theoretical Framework

In the code, the focal plane wavefront sensing uses Pairwise Probing (PWP) to estimate the complex electric field of the coherent starlight speckles. By applying known phase perturbations ( = probes) to the DM, we modulate the coherent stellar flux to extract the speckles electric field.

Let $E_0$ be the unperturbed electric field in the pupil plane, and $C$ the linear coronagraphic propagation operator. Because the host star and the exoplanet are incoherent, the total intensity measured on the detector is the sum of coherent and incoherent intensities:
$$I_0 = |C[E_0]|^2 + I_{incoh}$$

When applying a pair of symmetric probes $\{+, -\}$, we induce a phase shift $\Delta\Psi$ on the DM. Assuming the phase modulation primarily affects the coherent stellar flux (and under the small aberration approximation where the optical path difference is small compared to the wavelength), the measured intensities on the detector are:
$$I_{+} \approx |C[E_0] + i\Delta p|^2 + I_{incoh}$$
$$I_{-} \approx |C[E_0] - i\Delta p|^2 + I_{incoh}$$
where $\Delta p = \frac{C[E_0e^{i\Delta\Psi}]-C[E_0]}{i}=\frac{C[E_0]-C[E_0e^{-i\Delta \Psi}]}{i}$.
In fact, the $\Delta p$ we use is the mean value using both definition above.

Using the theoretical model, we obtain the phase of the probe's electric field by taking the argument of the mean $\Delta_p$.
To obtain the field amplitude, we do not use the model but an empirical measurement: we show that 
$$|\Delta p| \approx \sqrt{\frac{I_{+} + I_{-}}{2} - I_0}$$

To estimate the stellar speckles, we take for the probe number $n$ the difference of the paired images to isolate the interference cross-terms. This fundamental step removes the static unmodulated terms, including the incoherent planetary signal:
$$\delta_n = \frac{I_{+,n} - I_{-,n}}{2} \approx -2\Re(C[E_0])\Im(\Delta p_n) + 2\Im(C[E_0])\Re(\Delta p_n)$$

By applying multiple independent probes (in the code it's 3 pairs $\Delta p_1$, $\Delta p_2$ and $\Delta p_3$ to increase measurement robustness so 6 probes), we can construct an invertible linear system for every pixel:
$$
\begin{bmatrix}
-2\Im(\Delta p_1) & 2\Re(\Delta p_1) \\
-2\Im(\Delta p_2) & 2\Re(\Delta p_2) \\
-2\Im(\Delta p_3) & 2\Re(\Delta p_3)
\end{bmatrix}
\begin{bmatrix}
\Re(C[E_0]) \\
\Im(C[E_0])
\end{bmatrix}
=
\begin{bmatrix}
\delta_1 \\
\delta_2 \\
\delta_3
\end{bmatrix}
$$
### 


### References
* CADY, Eric, BOWMAN, Nicholas, GREENBAUM, Alexandra Z., et al. High-order wavefront sensing and control for the Roman Coronagraph Instrument (CGI): architecture and measured performance. Journal of Astronomical Telescopes, Instruments, and Systems, 2025, vol. 11, no 2, p. 021408-021408.


* KRIST, John E., STEEVES, John B., DUBE, Brandon D., et al. End-to-end numerical modeling of the Roman Space Telescope coronagraph. Journal of Astronomical Telescopes, Instruments, and Systems, 2023, vol. 9, no 4, p. 045002-045002.