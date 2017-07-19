import numpy as np, numpy.linalg as nlg
import scipy.sparse as ss, scipy.sparse.linalg as ssl

_PLOTTING = True
try:
  import matplotlib.pyplot as plt
except ImportError:
  _PLOTTING = False


# Utility function.
def chop(v):
  w = v[1:]


def TVRegDiff(
    data, num_iter, alpha, u0=None, scale="large", ep=1e-6, dx=None,
    plotting=True, verbose=True):
  # u = TVRegDiff( data, num_iter, alpha, u0, scale, ep, dx, plotting, verbose)
  # u = tvdiff(e, dx, num_iter, ep, alpha)
  # Rick Chartrand (rickc@lanl.gov), Apr. 10, 2011
  # Please cite Rick Chartrand, "Numerical differentiation of noisy,
  # nonsmooth data," ISRN Applied Mathematics, Vol. 2011, Article ID 164564, 
  # 2011. 
  
  # Inputs:  (First three required; omitting the final N parameters for N < 7
  #           or passing in [] results in default values being used.) 
  #       data        Vector of data to be differentiated.
  #
  #       num_iter    Number of iterations to run the main loop.  A stopping
  #                   condition based on the norm of the gradient vector g
  #                   below would be an easy modification.  No default value.
  #
  #       alpha       Regularization parameter.  This is the main parameter
  #                   to fiddle with.  Start by varying by orders of
  #                   magnitude until reasonable results are obtained.  A
  #                   value to the nearest power of 10 is usally adequate.
  #                   No default value.  Higher values increase
  #                   regularization strenght and improve conditioning.
  #
  #       u0          Initialization of the iteration.  Default value is the
  #                   naive derivative (without scaling), of appropriate
  #                   length (this being different for the two methods).
  #                   Although the solution is theoretically independent of
  #                   the intialization, a poor choice can exacerbate
  #                   conditioning issues when the linear system is solved.
  #
  #       scale       'large' or 'small' (case insensitive).  Default is
  #                   'small'.  'small' has somewhat better boundary
  #                   behavior, but becomes unwieldly for data larger than
  #                   1000 entries or so.  'large' has simpler numerics but
  #                   is more efficient for large-scale problems.  'large' is
  #                   more readily modified for higher-order derivatives,
  #                   since the implicit differentiation matrix is square.
  #
  #       ep          Parameter for avoiding division by zero.  Default value
  #                   is 1e-6.  Results should not be very sensitive to the
  #                   value.  Larger values improve conditioning and
  #                   therefore speed, while smaller values give more
  #                   accurate results with sharper jumps.
  #
  #       dx          Grid spacing, used in the definition of the derivative
  #                   operators.  Default is the reciprocal of the data size.
  #
  #       plotting    Flag whether to display plot at each iteration.
  #                   Default is 1 (yes).  Useful, but adds significant
  #                   running time.
  #
  #       verbose     Flag whether to display diagnostics at each
  #                   iteration.  Default is 1 (yes).  Useful for diagnosing
  #                   preconditioning problems.  When tolerance is not met,
  #                   an early iterate being best is more worrying than a
  #                   large relative residual.
                    
  # Output:
  
  #       u           Estimate of the regularized derivative of data.  Due to
  #                   different grid assumptions, length( u ) = 
  #                   length( data ) + 1 if scale = 'small', otherwise
  #                   length( u ) = length( data ).

  # % Copyright notice:
  # Copyright 2010. Los Alamos National Security, LLC. This material
  # was produced under U.S. Government contract DE-AC52-06NA25396 for
  # Los Alamos National Laboratory, which is operated by Los Alamos
  # National Security, LLC, for the U.S. Department of Energy. The
  # Government is granted for, itself and others acting on its
  # behalf, a paid-up, nonexclusive, irrevocable worldwide license in
  # this material to reproduce, prepare derivative works, and perform
  # publicly and display publicly. Beginning five (5) years after
  # (March 31, 2011) permission to assert copyright was obtained,
  # subject to additional five-year worldwide renewals, the
  # Government is granted for itself and others acting on its behalf
  # a paid-up, nonexclusive, irrevocable worldwide license in this
  # material to reproduce, prepare derivative works, distribute
  # copies to the public, perform publicly and display publicly, and
  # to permit others to do so. NEITHER THE UNITED STATES NOR THE
  # UNITED STATES DEPARTMENT OF ENERGY, NOR LOS ALAMOS NATIONAL
  # SECURITY, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY,
  # EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
  # RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF
  # ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR
  # REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
  # RIGHTS. 

  # % BSD License notice:
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions
  # are met: 
  
  #      Redistributions of source code must retain the above
  #      copyright notice, this list of conditions and the following
  #      disclaimer.  
  #      Redistributions in binary form must reproduce the above
  #      copyright notice, this list of conditions and the following
  #      disclaimer in the documentation and/or other materials
  #      provided with the distribution. 
  #      Neither the name of Los Alamos National Security nor the names of its
  #      contributors may be used to endorse or promote products
  #      derived from this software without specific prior written
  #      permission. 
   
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
  # CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
  # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  # MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
  # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
  # USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  # AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  # LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  # ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  # POSSIBILITY OF SUCH DAMAGE. 

  ## Code starts here
  # Make sure we have a column vector.
  data = np.squeeze(data)
  if len(data.shape) > 1:
    raise ValueError("data must be a single dimensional array.")

  if not _PLOTTING and plotting:
    print("Warning: Plotting is disabled since matplotlib was not imported.")
    plotting = False

  # Get the data size.
  n = data.shape[0]

  # Default checking. (u0 is done separately within each method.)
  if dx is None:
    dx = 1. / n

  # Different methods for small- and large-scale problems.
  if scale == "small":
    # Construct differentiation matrix.
    c = np.ones((n + 1, 1)) / dx
    D = ss.spdiags([-c, c], [0, 1], n, n + 1)
    del c

    # Construct antidifferentiation operator and its adjoint.
    A = lambda x: chop((np.cumsum(x) - 0.5 * (x + x[0]))) * dx
    AT = lambda w: (
        np.sum(w) * np.ones((n + 1, 1)) -
        np.c_[np.sum(w) / 2., np.cumsum(w) - w / 2.]) * dx

    # Default initialization is naive derivative.
    if u0 is None:
      diff_data = data[1:] - data[:-1]
      u0 = np.r_[0, diff_data, 0]

    u = u0
    # Since Au(0) = 0, we need to adjust.
    ofst = data[0]

    # Precompute.
    ATb = AT(ofst - data)
        # Main loop.
    for itr in range(num_iter):
      # Diagonal matrix of weights, for linearizing E-L equation.
      Q = ss.spdiags(1. / (np.sqrt((D.dot(u)) ** 2 + ep)), 0, n, n)
      # Linearized diffusion matrix, also approximation of Hessian.
      L = dx * D.T.dot(Q.dot(D))
      # Gradient of functional.
      g = AT(A(u)) + ATb + alpha * L.dot(u)
      # Prepare to solve linear equation.
      tol = 1e-4
      maxit = 100
      # Simple preconditioner.
      P = alpha * ss.spdiags([L.diagonal() + 1], 0, n + 1, n + 1)

      matvec = lambda v: alpha * L.dot(v) + AT(A(v))
      Lop = ssl.LinearOperator(L.shape, matvec=matvec)
      s = ssl.cg(Lop, g, tol=tol, maxiter=maxit, M=P)
      if verbose:
        print("Iteration %4d: Relative change = %.3f, Gradient norm = %.3f\n"%(
              itr, nlg.norm(s) / nlg.norm(u), nlg.norm(g)))

      # Update solution.
      u -= s

      # Display plot.
      if plotting:
        plt.plot(u)
        plt.show(block=False)
          
  elif scale == "large":
    # Construct antidifferentiation operator and its adjoint.
    A = lambda v: np.cumsum(v)
    AT = lambda w: (
        np.sum(w) * np.ones((w.shape[0], 1)) - np.r_[0, np.cumsum(w[:-1])])

    # Construct differentiation matrix.
    c = np.ones((n, 1))
    D = ss.spdiags([-c, c], [0, 1], n, n) / dx
    D[n-1, n-1] = 0
    del c

    # Since Au(0) = 0, we need to adjust.
    data -= data[0]
    # Default initialization is naive derivative.
    if u0 is None:
      diff_data = data[1:] - data[:-1]
      u0 = np.r_[0, diff_data]

    u = u0
    # Precompute.
    ATd = AT(data)

    # Main loop.
    for itr in range(num_iter):
      # Diagonal matrix of weights, for linearizing E-L equation.
      Q = ss.spdiags(1. / np.sqrt((D.dot(u)**2 + ep)), 0, n, n)
      # Linearized diffusion matrix, also approximation of Hessian.
      L = D.T.dot(Q.dot(D))
      # Gradient of functional.
      g = AT(A(u)) - ATd
      g += alpha * L.dot(u)
      # Build preconditioner.
      c = np.cumsum(np.arange(n, 0, -1)).reshape(-1, 1)
      B = alpha * L + ss.spdiags(c[::-1], 0, n, n)
      droptol = 1e-2

      R = cholinc( B, droptol );
      % Prepare to solve linear equation.
      tol = 1.0e-4;
      maxit = 100;
      if verbose
        s = pcg( @(x) ( alpha * L * x + AT( A( x ) ) ), -g, tol, maxit, R', R );
        fprintf( 'iteration %2d: relative change = %.3e, gradient norm = %.3e\n', ii, norm( s ) / norm( u ), norm( g ) );
      else
        [ s, ~ ] = pcg( @(x) ( alpha * L * x + AT( A( x ) ) ), -g, tol, maxit, R', R );

      % Update current solution
      u = u + s;
      % Display plot.
      if plotting
          plot( u, 'ok' ), drawnow;
