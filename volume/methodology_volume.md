# Methodology — volume estimation and uncertainty (LaTeX source)

Draft text for the report's methodology chapter, covering only what is needed
for the volume computation and its error estimate. The longitudinal (length)
analysis and the coverage analysis are deliberately left out; they belong to the
geometrical-analysis chapter and are referenced here as `\ref{chap:geom}`.

**Packages required:** `amsmath`, `amssymb`, `booktabs`. Units are written out,
so `siunitx` is not needed.

**Placeholders to replace:** `\ref{chap:targets}` (the chapter introducing the
surveyed targets) and `\ref{chap:geom}` (the geometrical-analysis chapter).
Sectioning levels (`\section`/`\subsection`/…) may need shifting to match the
surrounding document.

**Parameter values** in `tab:params` are those actually used for the reported
runs. **No measured volumes appear in this text** — it is methodology only. For
reference, the worked example behind the statistics is April, Livox, profiles:
`V̄ = 1001.5 m³`, `s = 13.6 m³`, `CV = 1.35 %`, 95 % CI of the mean
`[984.7, 1018.4] m³`, 95 % CI of σ `[8.1, 39.0] m³`, Grubbs `G = 1.49 < 1.715`.

---

```latex
\section{Methodology}
\label{sec:method}

\subsection{Curvilinear tunnel coordinates}
\label{sec:coords}

\subsubsection{Motivation}

The quantity to be determined is the air volume enclosed between two
cross-sections of an inclined, gently curved tunnel. Expressed in the global
Cartesian frame of the reference scan, this volume has no convenient
description: the tunnel axis is neither aligned with a coordinate axis nor
straight, so a plane of constant $x$, $y$ or $z$ intersects the conduit
obliquely. The area of such an oblique intersection is not the cross-sectional
area, and the error grows with the local inclination and curvature. Slicing
along a global axis would therefore introduce a bias that depends on the
geometry of the tunnel rather than on the measurement.

A volume of this shape is naturally described in a curvilinear coordinate
system attached to the conduit itself. Let $s$ denote the arclength along a
fitted tunnel centreline, $r$ the distance from that centreline measured
perpendicular to it, and $\theta$ the azimuth around it. In these coordinates
the volume decomposes into a one-dimensional integral over cross-sections that
are perpendicular to the local axis by construction,
%
\begin{equation}
  V \;=\; \int_{s_0}^{s_1} A(s)\,\mathrm{d}s ,
  \qquad
  A(s) \;=\; \frac{1}{2}\int_{-\pi}^{\pi} r^2(s,\theta)\,\mathrm{d}\theta ,
  \label{eq:volume-general}
\end{equation}
%
which holds for a curved axis as well as a straight one. Four further
properties motivated the change of coordinates:

\begin{enumerate}
  \item \textbf{The tunnel wall becomes a scalar field.} After the
    transformation the measured surface is described by a single function
    $r(s,\theta)$ on a two-dimensional, cylindrical parameter domain. Regions
    that were not observed are then identifiable as empty cells of that domain
    and can be filled by interpolation along physically meaningful directions
    (around the cross-section, and along the tunnel). An unobserved region in
    the original three-dimensional point cloud has no comparable
    parameterisation over which to interpolate.
  \item \textbf{Robust estimation.} Each cell of the $(s,\theta)$ domain
    typically contains many points, so the wall position can be estimated by a
    median rather than by individual points. Measurement noise and isolated
    spurious returns are thereby suppressed before any integration is performed.
  \item \textbf{A common domain for both instruments.} Both point clouds are
    mapped into the same coordinate system, defined by one centreline and one
    pair of bounding planes. Volumes, and the radii from which they are formed,
    are therefore compared over an identical domain rather than over two
    independently defined ones.
  \item \textbf{Physical interpretability of the azimuth.} With $\theta$
    referenced to gravity (Sect.~\ref{sec:frame}), $\theta = 0$ denotes the
    ceiling, $\theta = \pm 180^\circ$ the floor and $\theta = \pm 90^\circ$ the
    side walls, so that any azimuthal structure can be stated in physical terms
    instead of as an arbitrary angle.
\end{enumerate}

Equation~(\ref{eq:volume-general}) additionally separates the volume into a
mean cross-sectional area and a length, $V = \bar{A}\,L$, which is used in the
geometrical analysis of Chapter~\ref{chap:geom} and is not pursued here.

\subsubsection{Centreline}
\label{sec:centreline}

The centreline is derived from the trajectory of the mobile scanner, i.e.\ from
the sequence of sensor poses estimated by the SLAM system, transformed into the
datum of the reference scan by the rigid registration of the corresponding
survey. The trajectory is a record of where the operator walked and is not
itself a centreline; it is used only as the support of a smooth curve that
follows the conduit.

Two properties of the raw trajectory must be addressed before fitting. First,
the poses are sampled in time, so every pause of the operator produces a dense
cluster of nearly stationary, jittering positions. Because a spline
parameterised by chord length advances only marginally across such a cluster
while the position continues to fluctuate, each pause is reproduced as a
near-cusp of the fitted curve, with local curvatures several orders of
magnitude above any physical bend. Such a cusp corrupts the tangent and hence
the entire local frame. The trajectory is therefore resampled to uniform
arclength spacing $\Delta s_\mathrm{traj}$ before fitting, which removes the
non-uniform weighting. Second, the survey is walked out and back, so the
trajectory is first reduced to a single monotonic outbound leg.

A cubic smoothing B-spline $\mathbf{c}(u)$ is then fitted to the resampled
positions $\mathbf{q}_k$ under the smoothing condition
%
\begin{equation}
  \sum_{k} \left\lVert \mathbf{q}_k - \mathbf{c}(u_k) \right\rVert^2 \;\le\; S ,
  \qquad S = \lambda N_q ,
\end{equation}
%
with $N_q$ the number of support points and $\lambda$ the smoothing factor of
Table~\ref{tab:params}. The fitted curve is resampled at uniform arclength
$\Delta s_\mathrm{cl}$, yielding centreline samples $\mathbf{c}(s)$ with unit
tangent
%
\begin{equation}
  \mathbf{T}(s) \;=\; \frac{\mathrm{d}\mathbf{c}}{\mathrm{d}s} .
\end{equation}
%
Two quantities are monitored as fit diagnostics: the root-mean-square distance
between the raw outbound trajectory and the fitted curve, and the maximum
curvature $\kappa_{\max} = \lVert \mathbf{c}' \times \mathbf{c}'' \rVert /
\lVert \mathbf{c}' \rVert^3$. The latter is the direct test for residual cusps;
after resampling and smoothing it corresponds to a minimum bend radius of tens
of metres, consistent with the geometry of the excavation.

\subsubsection{Azimuthal reference frame}
\label{sec:frame}

A cross-sectional frame requires two unit vectors spanning the plane
perpendicular to $\mathbf{T}(s)$. These are obtained by projecting the world
vertical $\hat{\mathbf{z}}$ into that plane,
%
\begin{equation}
  \mathbf{e}_1(s) \;=\;
  \frac{\hat{\mathbf{z}} - \bigl(\hat{\mathbf{z}}\cdot\mathbf{T}(s)\bigr)\mathbf{T}(s)}
       {\bigl\lVert \hat{\mathbf{z}} - \bigl(\hat{\mathbf{z}}\cdot\mathbf{T}(s)\bigr)\mathbf{T}(s) \bigr\rVert} ,
  \qquad
  \mathbf{e}_2(s) \;=\; \mathbf{T}(s) \times \mathbf{e}_1(s) ,
  \label{eq:frame}
\end{equation}
%
so that $\theta = 0$ points upwards. Definition~(\ref{eq:frame}) is local: each
sample is constructed independently of its neighbours, and the frame therefore
cannot accumulate drift or twist along the tunnel. A Frenet frame was not used,
because its normal is undefined at vanishing curvature and rotates abruptly on
nearly straight sections, which would scatter one physical direction across
different values of $\theta$. The construction degenerates only as the tangent
approaches the vertical, where the projection of $\hat{\mathbf{z}}$ vanishes; a
rotation-minimising frame is retained as a fallback for that case and is not
required here, the maximum tangent inclination being approximately
$7^\circ$.

The frame is verified against the raw point coordinates rather than against the
definition that produced it, by binning the vertical offset
$(\mathbf{p}-\mathbf{c})\cdot\hat{\mathbf{z}}$ of the points by $\theta$ and
fitting its first harmonic; the phase of that harmonic is the direction the
data identify as upward, and it agrees with $\theta = 0$ to approximately
$2^\circ$.

\subsubsection{Transformation of the point clouds}

Each point $\mathbf{p}_i$ of a cloud is assigned coordinates
$(s_i, r_i, \theta_i)$ by locating the nearest centreline sample, refining the
chainage by projection onto the local tangent, and decomposing the residual
vector in the frame of Eq.~(\ref{eq:frame}):
%
\begin{align}
  s_i &= \arg\min_{s} \left\lVert \mathbf{p}_i - \mathbf{c}(s) \right\rVert , \\
  \mathbf{d}_i &= \mathbf{p}_i - \mathbf{c}(s_i) , \qquad
  \mathbf{d}_i^{\perp} = \mathbf{d}_i - \bigl(\mathbf{d}_i \cdot \mathbf{T}(s_i)\bigr)\,\mathbf{T}(s_i) , \\
  r_i &= \bigl\lVert \mathbf{d}_i^{\perp} \bigr\rVert , \qquad
  \theta_i = \operatorname{atan2}\!\left(
      \mathbf{d}_i^{\perp}\cdot\mathbf{e}_2(s_i),\;
      \mathbf{d}_i^{\perp}\cdot\mathbf{e}_1(s_i)\right) .
\end{align}
%
The nearest-sample search is performed with a $k$-d tree over the centreline
samples.

The representation assumes that the cross-sections are star-shaped with respect
to the centreline, i.e.\ that $r$ is single-valued for a given $(s,\theta)$. A
ray that intersects the wall more than once — at an overhang, a niche, or an
object standing against the wall — violates this assumption. Such cases are
detected per cell as a multimodal radius distribution and are reported, so that
the assumption is monitored rather than presumed (Sect.~\ref{sec:profiles}).

\subsubsection{Measurement domain}
\label{sec:domain}

The volume is not defined for the tunnel as a whole but for a bounded section
of it, and the bounds must be identical for every cloud and every method if the
results are to be comparable. They are provided by the surveyed targets
introduced in Chapter~\ref{chap:targets}. The first and the last target define
the two end-cap planes that close the volume: each is projected onto the fitted
centreline, giving the chainages $s_0$ and $s_1$, and the measurement domain is
%
\begin{equation}
  \mathcal{D} = [\,s_0,\, s_1\,] , \qquad L = s_1 - s_0 .
\end{equation}
%
All estimators integrate over this same domain and both end faces are closed at
these planes, so the reported volumes differ only in how the enclosed space is
reconstructed, not in what space is enclosed. The remaining targets are not
used in the volume computation; they enter the longitudinal analysis of
Chapter~\ref{chap:geom}.

\begin{table}[htbp]
  \centering
  \caption{Parameters of the coordinate construction and of the estimators.}
  \label{tab:params}
  \begin{tabular}{llr}
    \toprule
    Symbol & Description & Value \\
    \midrule
    $\Delta s_\mathrm{traj}$ & trajectory resampling before spline fit & $0.50$ m \\
    $\lambda$                & spline smoothing factor                & $0.05$ \\
    $\Delta s_\mathrm{cl}$   & centreline sampling interval           & $0.10$ m \\
    $\Delta s$               & slab thickness (cross-sections)        & $0.25$ m \\
    $\Delta\theta$           & azimuthal bin width                    & $1^\circ$ \\
    $\delta_r$               & radius gap flagged as multimodal       & $0.30$ m \\
    $c_{\min}$               & minimum azimuthal coverage of a slab   & $0.10$ \\
    $h$                      & voxel sizes (marching cubes)           & $0.05$, $0.10$, $0.15$ m \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Volume estimation}
\label{sec:volume}

Three estimators are evaluated for every cloud, all over the domain
$\mathcal{D}$ of Sect.~\ref{sec:domain}. A fourth, independent estimator is
described in Sect.~\ref{sec:mc} and is not applicable to the present data.

\subsubsection{Cross-sectional profiles}
\label{sec:profiles}

The domain is partitioned into slabs of thickness $\Delta s$ centred at
$s_i = s_0 + (i-\tfrac{1}{2})\Delta s$, and the azimuth is divided into
$n_\theta = 360^\circ/\Delta\theta$ bins. For every cell $(i,j)$ the wall radius
is estimated as the median of the radii of all points falling into it,
%
\begin{equation}
  r_{ij} \;=\; \operatorname{median}\bigl\{\, r_k \;:\;
    \lvert s_k - s_i \rvert < \tfrac{\Delta s}{2},\;
    \theta_k \in \text{bin } j \,\bigr\} ,
\end{equation}
%
the median being preferred to the mean because it is insensitive to the small
fraction of points returned from beyond or in front of the wall. If the sorted
radii within a cell exhibit a gap larger than $\delta_r$, the cell contains two
distinct surfaces along the same ray; the outermost cluster is then used, and
the cell is counted as multimodal. The fraction of such cells is reported per
slab, and a slab in which it exceeds $20\,\%$ is flagged, since the star-shape
assumption is then locally questionable.

\paragraph{Interpolation of unobserved cells.}
Cells that contain no point are filled in two sequential passes.

\emph{(i) Azimuthal pass.} Within each slab, missing radii are obtained by
linear interpolation over $\theta$ between the nearest observed bins on either
side. The interpolation is periodic: the bin index axis is extended by one
period in each direction before interpolating, so that a gap crossing the
$\pm 180^\circ$ seam is filled from both of its neighbours rather than
extrapolated from one side. This pass is the appropriate one for the majority
of gaps, which are azimuthal bands caused by the field of view of the mobile
sensor or by occlusion, and it reconstructs the missing part of a cross-section
from the observed remainder of the same cross-section.

\emph{(ii) Longitudinal pass.} A slab in which fewer than a fraction $c_{\min}$
of the azimuthal bins are occupied carries too little information for the first
pass to be meaningful and is excluded from it. Such slabs are instead filled
column-wise: for each azimuth $j$, the radius is interpolated linearly in $s$
between the nearest slabs that possess a value at the same azimuth, with
constant continuation beyond the outermost observed slab. The gap is thus
bridged along the tunnel at fixed azimuth, which preserves the azimuthal shape
of the profile.

The fraction of interpolated cells is recorded for every slab and reported with
the result, so that the extent to which a volume relies on reconstruction
rather than on measurement remains explicit.

\paragraph{Integration.}
The filled radii of a slab define a closed polygon in the cross-sectional plane
with vertices at radius $r_{ij}$ and angle $\theta_j$. Its area follows from the
shoelace formula, which for constant angular spacing reduces to
%
\begin{equation}
  A_i \;=\; \frac{\sin \Delta\theta}{2} \sum_{j=1}^{n_\theta} r_{ij}\, r_{i,j+1} ,
  \qquad r_{i,n_\theta+1} \equiv r_{i1} ,
  \label{eq:shoelace}
\end{equation}
%
the discrete counterpart of the polar area integral in
Eq.~(\ref{eq:volume-general}). The volume is then obtained by integrating the
area profile $A(s)$ using the composite trapezoidal rule; Simpson's rule is
evaluated in parallel as a check on the integration scheme. Because the slab
areas are located at slab \emph{centres}, a quadrature over these abscissae
spans only $[s_1, s_{n}]$ and omits a half-slab at each end of the domain, a
systematic deficit of order $\Delta s / L$. The two end pieces are therefore
added explicitly, with the area held constant across each:
%
\begin{equation}
  V \;=\; \underbrace{\int_{s_1}^{s_n} A(s)\,\mathrm{d}s}_{\text{trapezoidal}}
      \;+\; A_1\,(s_1 - s_0) \;+\; A_n\,(s_1^{\mathrm{end}} - s_n) .
\end{equation}
%
Holding the area constant rather than extrapolating it is deliberate: the
pieces are only $\Delta s/2$ wide, so the difference between the two treatments
is of second order, whereas a linear extrapolation could be destabilised by a
sparsely sampled end slab.

\subsubsection{Closed surface mesh}
\label{sec:mesh}

The second estimator reconstructs the tunnel wall as a closed triangulated
surface and computes the volume it encloses. It uses the same median radius
grid $r_{ij}$ as Sect.~\ref{sec:profiles}, but differs from it in both the
interpolation and the integration.

Missing cells are filled by two-dimensional interpolation over the $(s,\theta)$
domain, i.e.\ from all surrounding observed cells simultaneously rather than
first along $\theta$ and then along $s$. Periodicity in $\theta$ is enforced by
tiling the grid three times in the azimuthal direction and retaining the central
copy, which prevents a discontinuity along the $\pm 180^\circ$ seam. Linear
interpolation is applied first, and nearest-neighbour interpolation is used for
the few cells lying outside the convex hull of the observed data, since an
unfilled cell would leave a hole in the surface and thereby invalidate the
volume.

Each grid node is mapped back into three dimensions by
%
\begin{equation}
  \mathbf{X}(s_i,\theta_j) \;=\; \mathbf{c}(s_i)
    \;+\; r_{ij}\bigl[\cos\theta_j\,\mathbf{e}_1(s_i) + \sin\theta_j\,\mathbf{e}_2(s_i)\bigr] ,
\end{equation}
%
and the resulting quadrilateral grid is triangulated. The two ends are closed by
triangle fans at the end-cap planes of Sect.~\ref{sec:domain}, producing a
closed, consistently oriented surface. Its volume follows from the divergence
theorem, $V = \tfrac{1}{3}\oint_{\partial\Omega} \mathbf{x}\cdot\mathbf{n}\,
\mathrm{d}A$, which for a triangulation with vertices
$\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2$ per face becomes
%
\begin{equation}
  V \;=\; \frac{1}{6}\left\lvert \sum_{f} \mathbf{v}_0^{(f)} \cdot
    \bigl(\mathbf{v}_1^{(f)} \times \mathbf{v}_2^{(f)}\bigr) \right\rvert .
  \label{eq:divergence}
\end{equation}
%
Closure of the mesh is verified before the volume is accepted.

It should be emphasised that this estimator shares the radius extraction with
the profile method. Agreement between the two therefore validates the
integration scheme and the hole-filling strategy, which genuinely differ, but
not the extraction itself; an error in the estimation of $r_{ij}$ would affect
both identically.

\subsubsection{Convex-hull bound}
\label{sec:hull}

The third estimator replaces the polygon of Eq.~(\ref{eq:shoelace}) by the
convex hull of the measured points. Within each slab the points are projected
onto the cross-sectional plane using the frame coordinates
%
\begin{equation}
  (x,y) = \bigl(r\cos\theta,\; r\sin\theta\bigr) ,
\end{equation}
%
and the area of their two-dimensional convex hull is taken as $A_i$, which is
then integrated as before. No interpolation is applied.

By construction the convex hull spans any concavity of the true cross-section,
so this estimator is an upper bound for a fully observed section. The
complementary property must be kept in mind: where data are missing, the hull
also spans the unobserved gap and consequently under-estimates the area. The
bound is thus strict only where coverage is complete, and it is reported as a
bound rather than as an independent estimate of the volume.

\subsubsection{Marching cubes}
\label{sec:mc}

The three estimators above all depend on the centreline and on the
representation $r(s,\theta)$. A fourth estimator was implemented that depends
on neither, in order to provide a genuinely independent check. The point cloud
is voxelised at edge length $h$ into an occupancy grid $O$; the wall shell is
dilated by one voxel so that it forms a closed barrier at least two voxels
thick; the exterior is identified by a connected-component labelling of the
free space, every component reaching the grid boundary being exterior; and the
largest enclosed component is taken as the air volume. The dilation is
subsequently reversed on the air mask to compensate its inward bias. The volume
is evaluated both as the voxel count times $h^3$ and from a marching-cubes
triangulation of the air field via Eq.~(\ref{eq:divergence}), and the sequence
over several $h$ is extrapolated to $h \to 0$. In the tube configuration the
centreline enters only through the placement of the two end caps, on the same
planes used by the other estimators.

This estimator did not yield a volume for the present data sets, for a reason
that is intrinsic to the method rather than incidental. The flood fill requires
the wall to be closed; an opening wider than the dilation connects the interior
to the exterior and the enclosed cavity disappears. Sealing an opening of
radius $\rho$ requires a dilation of at least $k h \gtrsim \rho$, but the same
dilation grows the shell inward everywhere by $kh$, so that the interior is
strongly inflated and the cavity is consumed from within. Since the unobserved
patches of these clouds are of metre scale, comparable to the tunnel radius
itself, there is no dilation that simultaneously seals the wall and preserves
the interior; coarsening the voxel size instead merely replaces the leak by an
over-filled, incoherent air region. The condition is reported explicitly as a
leak, so that the method returns no value rather than a plausible but incorrect
one. The consequence is that the independent cross-check is available only on
synthetic data, and that the reported volumes rest on the shared representation
$r(s,\theta)$.

\subsubsection{Verification on a synthetic geometry}

Before application to the measured data, the estimators were verified against a
cylinder of known radius and length, sampled as a synthetic point cloud and
accompanied by a synthetic hand-held trajectory. Profiles and surface mesh
recover the analytical volume to $-0.008\,\%$, the convex-hull bound to
$-0.001\,\%$, and marching cubes to $+0.26\,\%$, the latter being consistent
with the discretisation of a voxel method. The recovered axis deviates from the
true axis by less than $0.01^\circ$. The verification therefore covers the
coordinate construction and all four estimators; it does not cover incomplete
coverage, which is addressed in Chapter~\ref{chap:geom}.

\subsection{Uncertainty quantification}
\label{sec:uncertainty}

\subsubsection{Approach}

The uncertainty of the reported volume is determined empirically, from the
dispersion of repeated independent measurements, rather than from a model of
the contributing error sources. Each campaign comprises $N = 5$ repetitions of
the same survey. Every repetition is processed through the complete pipeline
independently: it carries its own trajectory, its own registration, its own
fitted centreline and therefore its own coordinate system, its own
$r(s,\theta)$ representation and its own interpolation. The scatter of the
resulting volumes consequently contains the propagated effect of every step of
the procedure, and not only of those steps that could be modelled explicitly.
The reference scan is a single acquisition and is processed once per campaign.

The alternative, a bottom-up budget in which individual error sources are
modelled and combined in quadrature, was rejected for two reasons. It can only
contain those contributions that are enumerated in advance, and it is
structurally insensitive to errors shared by the estimators — an error in the
common extraction of $r(s,\theta)$, for instance, would propagate identically
into the profile and the surface-mesh volumes and would cancel from any
comparison between them. Repetition makes no such assumption; its limitation is
the complementary one and is stated in Sect.~\ref{sec:unc-limits}.

\subsubsection{Statistical estimators}

Let $V_{m,i}$ denote the volume obtained by method $m$ in repetition
$i = 1,\dots,N$. The repetitions are treated as independent realisations of one
random variable, whose mean and standard deviation are estimated by
%
\begin{equation}
  \bar{V}_m = \frac{1}{N}\sum_{i=1}^{N} V_{m,i} ,
  \qquad
  s_m = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\bigl(V_{m,i} - \bar{V}_m\bigr)^{2}} .
  \label{eq:mean-sd}
\end{equation}
%
The divisor $N-1$ in Eq.~(\ref{eq:mean-sd}) is Bessel's correction: the
deviations are taken with respect to the sample mean, which is itself
determined by the data, leaving $N-1$ degrees of freedom; without the
correction $s_m^2$ would be a biased estimate of the population variance, by
$20\,\%$ at $N = 5$.

The quantity $s_m$ is the dispersion of a \emph{single} measurement and is the
value quoted as the $1\sigma$ uncertainty of one survey. It is also expressed
in relative form as the coefficient of variation,
%
\begin{equation}
  \mathrm{CV}_m = \frac{s_m}{\bar{V}_m} ,
\end{equation}
%
which permits comparison between methods and between campaigns of differing
volume. The uncertainty of the campaign mean is smaller by $\sqrt{N}$,
%
\begin{equation}
  \mathrm{SE}(\bar{V}_m) = \frac{s_m}{\sqrt{N}} ,
\end{equation}
%
and the corresponding two-sided confidence interval at the $95\,\%$ level is
%
\begin{equation}
  \bar{V}_m \;\pm\; t_{0.975,\,N-1}\,\frac{s_m}{\sqrt{N}} ,
  \label{eq:ci-mean}
\end{equation}
%
where Student's $t$ distribution is used, rather than the normal distribution,
because the standard deviation is estimated from the same small sample; at
$N = 5$, $t_{0.975,4} = 2.776$ against $z_{0.975} = 1.96$, so the interval is
$42\,\%$ wider than a normal approximation would suggest.

Since $s_m$ is itself estimated from five values, its own precision is limited.
With $(N-1)s_m^2/\sigma_m^2$ distributed as $\chi^2$ with $N-1$ degrees of
freedom, a $95\,\%$ confidence interval for the true standard deviation is
%
\begin{equation}
  \left[\;
    s_m\sqrt{\frac{N-1}{\chi^{2}_{0.975,\,N-1}}}
    ,\;\;
    s_m\sqrt{\frac{N-1}{\chi^{2}_{0.025,\,N-1}}}
  \;\right] ,
  \label{eq:ci-sd}
\end{equation}
%
which at $N = 5$ spans approximately $0.60\,s_m$ to $2.87\,s_m$; equivalently,
the relative uncertainty of $s_m$ is $1/\sqrt{2(N-1)} \approx 35\,\%$. The
standard deviation is therefore reported to two significant figures, and
Eq.~(\ref{eq:ci-sd}) is quoted alongside it so that the precision of the
uncertainty itself is not overstated. The observed range
$\max_i V_{m,i} - \min_i V_{m,i}$ is reported in addition, as it requires no
distributional assumption at all.

Each sample is finally screened for a single outlier by Grubbs' test, in which
the largest deviation is normalised by the sample standard deviation,
%
\begin{equation}
  G_m = \frac{\max_i \lvert V_{m,i} - \bar{V}_m \rvert}{s_m} ,
\end{equation}
%
and compared with the critical value
%
\begin{equation}
  G_\mathrm{crit} = \frac{N-1}{\sqrt{N}}
    \sqrt{\frac{t^{2}_{\alpha/(2N),\,N-2}}{N-2+t^{2}_{\alpha/(2N),\,N-2}}} ,
\end{equation}
%
the significance level being divided by $N$ because the test is applied to the
most extreme of $N$ values rather than to one selected in advance. At $N = 5$
and $\alpha = 0.05$ the critical value is $G_\mathrm{crit} = 1.715$, while $G$
cannot algebraically exceed $(N-1)/\sqrt{N} = 1.789$; the test can therefore
identify only a gross outlier, such as a repetition processed from an incorrect
input, and it is used as a diagnostic flag. No repetition is removed from the
sample on its basis.

\subsubsection{Systematic deviation from the reference}

The dispersion of Eq.~(\ref{eq:mean-sd}) characterises what varies between
repetitions. It does not characterise a deviation common to all of them. The
latter is quantified separately, as the difference between the mean volume of
the repeated surveys and the volume obtained by the same method from the
reference scan,
%
\begin{equation}
  b_m = \bar{V}_m - V_m^{\mathrm{ref}} .
\end{equation}
%
This quantity is systematic: it does not decrease with $N$, since repeating a
survey cannot recover a surface that the instrument does not observe, nor
remove a bias of the ranging itself. It is therefore reported next to $s_m$ and
is deliberately not combined with it in quadrature, which would misrepresent a
fixed offset as random scatter. Because $V_m^{\mathrm{ref}}$ is a single value,
the dispersion of $b_m$ over the repetitions equals $s_m$ and carries no
additional information.

\subsubsection{Auxiliary quantities and attribution}

To establish which part of the processing the dispersion originates from, a set
of auxiliary quantities is recorded for every repetition and analysed with the
same statistics: the domain length $L$, the root-mean-square residual and the
maximum curvature of the centreline fit, the longitudinal scale factor derived
from the targets, and the fraction of interpolated cells. Since $V$ scales with
$L$, a comparison of the coefficients of variation of the two indicates
directly whether the scatter is longitudinal or cross-sectional in origin.

A further, separate quantity is the difference between the profile and the
surface-mesh volumes within one repetition. As noted in Sect.~\ref{sec:mesh},
these estimators share the radius extraction and differ only in interpolation
and integration; their difference is therefore a sensitivity to the
reconstruction of unobserved regions, not an independent estimate of accuracy,
and it is reported as such.

\subsubsection{Scope and limitations}
\label{sec:unc-limits}

The standard deviation of Eq.~(\ref{eq:mean-sd}) comprises every contribution
that differs between repetitions: the SLAM solution and hence the trajectory,
the fitted centreline and the measurement domain it defines, which parts of the
wall are observed in a given pass, the point density, and all interpolation
that follows from these. It excludes, by construction, every contribution that
is identical in all repetitions: the coordinates of the targets, the definition
of the domain end planes, any bias of an estimator, and any systematic offset
of the ranging. These are addressed by the comparison with the reference
instrument described above, by the verification against a known synthetic
geometry, and by the coverage analysis of Chapter~\ref{chap:geom}.

The statistical statements of Eqs.~(\ref{eq:ci-mean}) and (\ref{eq:ci-sd}) and
Grubbs' test assume normally distributed volumes. With $N = 5$ this assumption
cannot be tested with meaningful power and is adopted as a working hypothesis;
the reported range is the corresponding distribution-free statement.
```
