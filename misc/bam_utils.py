"""
Utility functions for computing the BAM metric for a given object
"""

import numpy as np
import cv2
from skimage import draw
from skimage.measure import regionprops
import matplotlib.pyplot as plt 
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline



def get_enclosing_ellipse(cnt):
    """Generates a discretised contour for the smallest enclosing ellipse

    Args:
        cnt: a 2-column matrix for discritised contour
    
    Returns: 
        x: a 2-column matrix for discritised smallest enclosing ellipse

    """
    hull = cv2.convexHull(cnt, clockwise=True).squeeze() # get the convex hull
    A, centre, radii, rotation = min_2d_ellipse(hull, 0.01)

    return ellipse_coordinates(A,centre,radii,rotation) 


def min_2d_ellipse(P, tol):
    """Find the minimum covering 2D ellipse which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html

    Args:
        P: (d x N) dimnesional matrix containing N points in R^d.
        tol: error in the solution with respect to the optimal value.
    
    Returns:
        A: coordinates of the ellipse
        centre: centre of ellipse
        radii: radii of ellipse
        rotation: rotation of ellipse

    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initialisations
    err = 1.0 + tol
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tol:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(np.linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # centre of the ellipse 
    centre = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = np.linalg.inv(
                    np.dot(P.T, np.dot(np.diag(u), P)) - 
                    np.array([[a * b for b in centre] for a in centre])
                    ) / d
                    
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    return A, centre, radii, rotation


def ellipse_coordinates(A, centre, radii, rotation, N=20.0):
    """Generates the coordinates of the minimum enclosing ellipse

    Args: 
        A: a 2x2 matrix.
        centre: a 2D vector which represents the centre of the ellipsoid.
        radii: radii of ellipsoid generated from SVD
        rotation: rotation output by SVD
        N: the number of grid points for plotting the ellipse; Default: N = 20
    
    Returns:
        coordinates of ellipse

    """
    steps = int(round((2.0 * np.pi + 1/N)*N))
    theta = np.linspace(0.0, 2.0 * np.pi + 1/N, steps)
    
    # parametric equation of the ellipse
    x = radii[0] * np.cos(theta)
    y = radii[1] * np.sin(theta)

    # coordinate transform
    X = np.matmul(rotation,np.array([x,y]))
    X += np.expand_dims(centre.T, -1)
    return X.T


def ellipse_to_circle(coords):
    """Transform ellipse to circle

    Args:
        coords: input coordinates of ellipse
    
    Returns:
        circle_coords: coordinates of circle (transformed ellipse)
        alpha: orientation in radians
        a: major axis length of input ellipse
        b: minor axis length of input ellipse

    """
    # get the minimum x and y coordinates
    min_x = min(coords[:,1])
    min_y = min(coords[:,0])

    coords2 = np.round(coords - np.array([min_y, min_x]))

    # generate binary mask from coordinates
    shape = (int(max(coords2[:,1]) - min(coords2[:,1])), int(max(coords2[:,0]) - min(coords2[:,0])))
    mask = coords2mask(coords2[:,1], coords2[:,0], shape)
    mask = mask.T
    # get properties from binary mask
    props = regionprops(mask.astype('int'))
    orientation = props[0].orientation
    a = props[0].major_axis_length
    b = props[0].minor_axis_length
    centroid = props[0].centroid

    centre = (centroid + np.array([min_y, min_x]))
    centre = np.array([centre[1], centre[0]])
    alpha = -orientation 
    
    circle_coords = apply_transform(coords, centre, alpha, a, b)

    return circle_coords, alpha, a, b


def apply_transform(coords, centre, alpha, a, b):
    """Apply transformation to set of input coordinates

    Args:
        coords: input coordinates
        centre: centre of input coordinates
        alpha: orientation in radians
        a: major axis length of input
        b: major axis length of input
    
    Returns:
        transformed coordinates

    """
    T1 = [[1, 0, -centre[1]],
          [0, 1, -centre[0]],
          [0, 0, 1         ]]
    
    T2 = [[1, 0, centre[1]],
          [0, 1, centre[0]],
          [0, 0, 1         ]]

    R  = [[np.cos(alpha), -np.sin(alpha), 0],
          [np.sin(alpha), np.cos(alpha) , 0],
          [0            , 0             , 1]]

    S  = [[1, 0,   0],
          [0, a/b, 0],
          [0, 0  , 1]]

    trans_matrix = np.matmul(T2,np.matmul(S,np.matmul(R,T1)))

    coords_ones = np.hstack([coords[:,1:2], coords[:,:1], np.ones([coords.shape[0],1])])
    transformed_coords = np.matmul(trans_matrix, coords_ones[:, [1,0,2]].T).T
    return transformed_coords[:, [0,1]]


def coords2mask(row_coords, col_coords, shape):
    """Convert coordinates to a binary mask

    Args:
        row_coords: coordinates in y direction
        col_coords: coordinates in x direction
        shape: shape of binary mask to output
    
    Returns:
        mask: binary mask

    """
    fill_row_coords, fill_col_coords = draw.polygon(row_coords, col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

    
def best_alignment_metric(circle_coords, trans_coords, show_plots):
    """Compute BAM between two sets of coordinates as described in:

    Awan, Ruqayya, et al. "Glandular morphometrics for objective grading 
    of colorectal adenocarcinoma histology images." Scientific reports 7.1 (2017): 1-12.

    Args:
        circle_coords: circle coordinates
        trans_coords: transformed object coordinates
        show_plots: show the coordinate plots at each stage of the implementation
    
    Returns:
        d: best alignment distance
        R: best index shift
        phi: best planar rotation

    """
    # normalise the contour
    circle_coords_rs = resample_curve(circle_coords, 300)
    trans_coords_rs = resample_curve(trans_coords, 300)

    # convert to complex representation
    circle_complex = get_complex(circle_coords_rs)
    trans_complex = get_complex(trans_coords_rs)

    d, R, phi = bam_distance(circle_complex, trans_complex)
    
    if show_plots:   
        plt.figure(figsize=(5, 5))
        rotated_trans_coords = np.exp(1j*phi)*trans_complex
        rotated_and_cycled_trans_coords = np.roll(rotated_trans_coords, R)

        plt.subplot(2,2,1) 
        plt.plot(trans_coords[:,0],trans_coords[:,1]) # plot cartesian transformed object coordinates

        plt.plot(trans_coords[:,0],trans_coords[:,1],'x', markersize=5) # plot x at ticks
        plt.plot(trans_coords[0,0],trans_coords[0,1],'.', markersize=30) # mark the first point

        plt.subplot(2,2,2)
        plt.plot(circle_coords[:,0],circle_coords[:,1],'r') # plot cartesian circle coordinates
     
        plt.plot(circle_coords[:,0],circle_coords[:,1],'rx', markersize=5) # plot x at ticks
        plt.plot(circle_coords[0,0],circle_coords[0,1],'r.', markersize=30) # mark the first point
      
        plt.subplot(2,2,3)
        plt.plot(circle_complex.real, circle_complex.imag,'r') # plot complex circle coordinates
        
        plt.plot(circle_complex.real, circle_complex.imag,'rx', markersize=5)
        plt.plot(circle_complex.real[0],circle_complex.imag[0],'r.', markersize=30)
        plt.plot(rotated_trans_coords.real, rotated_trans_coords.imag) # plot optimally rotated complex transformed object
        plt.plot(rotated_trans_coords.real, rotated_trans_coords.imag,'x', markersize=5)
        plt.plot(rotated_trans_coords.real[0], rotated_trans_coords.imag[0],'.', markersize=30)
   
        plt.subplot(2,2,4)
        plt.plot(circle_complex.real, circle_complex.imag,'r') # plot complex circle coordinates
     
        plt.plot(circle_complex.real, circle_complex.imag,'rx', markersize=5)
        plt.plot(circle_complex.real[0], circle_complex.imag[0],'r.', markersize=30)
        plt.plot(rotated_and_cycled_trans_coords.real, rotated_and_cycled_trans_coords.imag) # plot optimally rotated and cyclically reordered complex transformed object
        plt.plot(rotated_and_cycled_trans_coords.real, rotated_and_cycled_trans_coords.imag,'x', markersize=5)
        plt.plot(rotated_and_cycled_trans_coords.real[0],rotated_and_cycled_trans_coords.imag[1],'.', markersize=30)

        plt.show()

    return d, R, phi


def bam_distance(u, v):
    """Rapidly computes the distance between curves u & v in the plane
    
    Args: 
        u: complex vector with shape 1xN
        v: complex vector with shape 1xN 
    
    Returns:
        d: best alignment distance
        R: best index shift
        phi: best planar rotation

    """
    sum_u = np.sum(u.real**2 + u.imag**2)
    sum_v = np.sum(v.real**2 + v.imag**2)
    v_tmp = np.flipud(v) # ! why do this?
    Xcorr = np.fft.ifft(np.fft.fft(np.conj(u)) * np.fft.fft(v_tmp))

    Xcorr2 = abs(Xcorr)
    A = np.max(Xcorr2)
    idx = np.argmax(Xcorr2)
    phi = np.arctan2(Xcorr[idx].imag, Xcorr[idx].real)
    R = idx 

    summand = sum_u + sum_v - 2*A
    if summand < 0:
        summand = 0
    d = np.sqrt(summand)

    return d, R, phi


def resample_curve(coords, N):
    """Resample the points on the input curve by interpolation.

    Args:
        coords: input coordinates
        N: number of sample points

    Returns:
        Xn: resampled coordinates
        
    """
    coords = coords.T
    diff = coords[:,1:] - coords[:,:-1] # check same size (measure diff between points)
    dist = np.sqrt(diff[0,:]**2 + diff[1,:]**2) # calculate the magnitude between neighbouring coordinates
    dist = np.concatenate((np.array([0]),dist), axis=-1) 
    cum_dist = np.cumsum(dist)/np.sum(dist) # cumulative sum of distances between neighbouring coordinates 
   
    sample_points = np.linspace(1/N,1,N)
    interp = np.zeros([2,N])
    for i in range(2):
        # generate interpolated points
        spline = InterpolatedUnivariateSpline(cum_dist,coords[i,:])
        interp[i,:] = spline(sample_points)

    q = curve_to_q(interp) # include description
    qn = ProjectC(q) # include description
    Xn = q_to_curve(qn)

    return Xn


def get_complex(curve):
    """Convert input curve to complex form and translate
    it such that it is centred at (0,0)

    Args:
        curve: input cartesian coordinates
    
    Returns:
        comp_curve_trans: translated complex curve

    """
    comp_curve = curve[0,:]+1j*curve[1,:]
    cf = np.mean(comp_curve)
    comp_curve_trans = comp_curve-cf

    return comp_curve_trans


def curve_to_q(p):
    """Include docstring

    Args:
        p: 

    """
    #! NEED TO UNDERSTAND WHAT IS GOING ON IN THIS FUNCTION and give appropriate comments
    N = p.shape[1]
    v = np.zeros([2,N])
    for i in range(2):
        v[i,:] = np.gradient(p[i,:], 1/N)

    # unit velocity
    L = np.sqrt(np.sqrt(v[0,:]**2 + v[1,:]**2)) # check whether this is the right thing to do

    okPos = L > 1e-5
    q = v[:,okPos] / L[okPos]
    q[:,~okPos] = np.zeros([2, np.sum(~okPos)])  

    T = q.shape[1]
    s = np.linspace(0,1,T)
    val = np.trapz(np.sum(q*q, axis=0), s)
    return q / np.sqrt(val)


def q_to_curve(q):
    """Include docstring

    Args:
        q:

    """
    #! NEED TO UNDERSTAND WHAT IS GOING ON IN THIS FUNCTION and give appropriate comments
    T = q.shape[1]
    qnorm = np.sqrt(q[0,:]**2 + q[1,:]**2)
 
    p = np.zeros([2,T])
    for i in range(2):
        p[i,:] = scipy.integrate.cumtrapz(q[i,:]*qnorm, initial=0)/T 
    return p


def ProjectC(q):
    """Include description

    Args:
        q:

    """
    #! NEED TO UNDERSTAND WHAT IS GOING ON IN THIS FUNCTION and give appropriate comments
    T = q.shape[1]
    dt = 0.35 # what is this? - provide description

    epsilon = 1e-6
    count = 1
    res = np.ones([1,2])

    s = np.linspace(0,1,T)
    tmp = np.trapz(np.sum(q*q, axis=0), s) 
    qnew = q / np.sqrt(tmp)

    while np.linalg.norm(res, ord=2) > epsilon:
        if count > 300:
            break

        J = np.zeros([2,2])
        for i in range(2):
            for j in range(1,2):
                J[i,j] = 3 * np.trapz(qnew[i,:]*qnew[j,:], s) 
        
        J = J + J.T
        for i in range(2):
            J[i,i] = 3 * np.trapz(qnew[i,:]*qnew[i,:], s) 

        J = J + np.identity(J.shape[0])
        qnorm = np.sqrt(qnew[0,:]**2 + qnew[1,:]**2)

        G = np.zeros([2])
        for i in range(2):
            G[i] = np.trapz(qnew[i,:]*qnorm, s)
        res = -G

        if np.linalg.norm(res, ord=2) < epsilon:
            break
        
        x = np.linalg.lstsq(J, res.T, rcond=None)[0] # solve system by least squares

        delG = form_basis_normal_A(qnew)

        tmp = x[0]*delG[0]*dt + x[1]*delG[1]*dt
        qnew = qnew + tmp

        count += 1

    tmp = np.trapz(np.sum(qnew*qnew, axis=0), s) 
    return qnew / np.sqrt(tmp)


def form_basis_normal_A(q):
    """Include docstring

    Args:
        q:

    """
    #! NEED TO UNDERSTAND WHAT IS GOING ON IN THIS FUNCTION and give appropriate comments
    T = q.shape[1]
    
    e = np.identity(2) 
    Ev = np.zeros([2,T,2])
    for i in range(2):
        Ev[:,:,i] = np.tile(np.expand_dims(e[:,i],-1),(1,T))
    
    qnorm = np.sqrt(q[0,:]**2 + q[1,:]**2)

    delG = []
    for i in range(2):
        tmp1 = np.tile(q[i,:]/qnorm,(2,1))
        tmp2 = np.tile(qnorm,(2,1))
        delG.append(tmp1*q + tmp2*Ev[:,:,i])

    return delG 
