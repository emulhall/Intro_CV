import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import random

def find_match(img1, img2):
    # Extract SIFT features
    sift=cv2.xfeatures2d.SIFT_create()
    kp1, des1=sift.detectAndCompute(img1,None)
    kp2, des2=sift.detectAndCompute(img2,None)

    #Convert keypoints to their locations
    loc1=[kp1[index].pt for index in range(len(kp1))]
    loc2=[kp2[index].pt for index in range(len(kp2))]

    x1=[]
    x2=[]
    image_2_to_1=[]
    #Ratio test using Nearest Neighbors training on image 2 and querying image 1
    nbrs=NearestNeighbors(n_neighbors=2).fit(des2)
    distances, indices=nbrs.kneighbors(des1)
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctice Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_2_to_1.append((i,matches[d1_index]))


    #To ensure bi-directional consistency we have to do it the other way around as well
    #Might've been good to put this small part in its own method if methods weren't already pre-defined
    nbrs=NearestNeighbors(n_neighbors=2).fit(des1)
    distances, indices=nbrs.kneighbors(des2)
    image_1_to_2=[]
    for i in range(indices.shape[0]):
        #Get our matches for this index and their corresponding distances
        matches=indices[i]
        dists=distances[i]

        #According to documentation for NearestNeighbors, result points are not necessarily in order of distance
        #Let's order these as d1 and d2
        d1=0
        d2=0
        d1_index=0

        if(dists[0]<dists[1]):
            d1=dists[0]
            d2=dists[1]
            d1_index=0
        else:
            d1=dists[1]
            d2=dists[0]
            d1_index=1

        #Lowe's ratio test
        #Ensure that the two points are discriminant enough according to Lowe's paper Distinctice Image Features from Scale-Invariant Keypoints
        if(d1<d2*0.7):
            image_1_to_2.append((matches[d1_index],i))

    #Let's compare and only keep those that are in both to ensure bi-directional consistency
    for i in range(len(image_2_to_1)):
        if(image_2_to_1[i] in image_1_to_2):
            x1.append(loc1[image_2_to_1[i][0]])
            x2.append(loc2[image_2_to_1[i][1]])

    pts1=np.array(x1)
    pts2=np.array(x2)
    return pts1, pts2


def compute_F(pts1, pts2):
    F=None
    inlier=[]

    #Iterate through ransac algorithm given number of times
    for n in range(500):

        #Sample 8 random points
        indices=random.sample(range(0,len(pts1)), 8)
        sample_x1=pts1[indices]
        sample_x2=pts2[indices]

        #Get an F estimate with the 8 point algorithm
        A=[]
        for i in range(len(sample_x1)):
            A.append([sample_x1[i,0]*sample_x2[i,0], sample_x1[i,1]*sample_x2[i,0], 
                sample_x2[i,0], sample_x1[i,0]*sample_x2[i,1], sample_x1[i,1]*sample_x2[i,1], sample_x2[i,1], sample_x1[i,0], sample_x1[i,1], 1])

        #Get the null space of A, which is equal to the last row of V^T
        u_initial, s_initial, v_t_initial = np.linalg.svd(np.array(A))
        F_messy=np.reshape(v_t_initial[-1], (3,3))

        #Clean up to ensure rank 2
        u_F, s_messy, v_t_F = np.linalg.svd(F_messy)
        s_clean = [1,1,0]
        F_est=u_F@np.diag(s_clean*s_messy)@v_t_F

        #Find inliers given the estimated F using v^TFu=0
        #Rearrange to add a column of ones
        u=np.vstack((pts1.T, np.ones(pts1.shape[0])))
        v=np.hstack((pts2, np.ones((len(pts2), 1))))

        #Multiply Fu
        Fu=F_est@u
        Fu=Fu.T

        #Multiply by v^T
        vtFu=v*Fu
        vtFu=np.sum(vtFu, axis=1)

        v_norm=np.linalg.norm(v, axis=1)

        Fu_norm=np.linalg.norm(Fu, axis=1)

        normalization=v_norm*Fu_norm

        score=np.abs(vtFu/normalization)

        #Find indices below threshold
        inlier_est=np.argwhere(score<1e-5)

        #Check to see if this E is a better estiamte based on the number of inliers
        if len(inlier_est)>len(inlier):
            inlier=np.reshape(inlier_est, (len(inlier_est)))
            F=F_est

    return F


def triangulation(P1, P2, pts1, pts2):
    pts3D=np.zeros((len(pts1),3))
    for i in range(len(pts3D)):
        A=np.vstack((([[0, -1, pts1[i,1]], [1, 0, -pts1[i,0]], [-pts1[i,1], pts1[i,0], 0]]@P1),
            ([[0, -1, pts2[i,1]], [1, 0, -pts2[i,0]], [-pts2[i,1], pts2[i,0], 0]]@P2)))
        #Get null space
        u,s,v_t=np.linalg.svd(A)
        #Normalize by last entry to make it [X 1]
        if(abs(v_t[-1,3])>0):
            pts3D[i]=v_t[-1,:3]/v_t[-1,3]
        #Prevent divide by zero
        else:
            pts3D[i]=v_t[-1,:3]/(v_t[-1,3]+1e-8)
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    valid_indices=[]
    R=None
    C=None
    pts3D=None
    for i in range(len(Rs)):
        #Evaluate cheirality
        x_c1=pts3Ds[i]
        x_c2=pts3Ds[i]-np.reshape(Cs[i],(3))

        cheir1=np.eye(3)[2]@x_c1.T
        cheir2=Rs[i][2]@x_c2.T
        valid=np.logical_and(cheir1>0, cheir2>0)
        valid_ind=np.argwhere(valid)

        if(len(valid_ind)>len(valid_indices)):
            R=Rs[i]
            C=Cs[i]
            pts3D=pts3Ds[i][valid_ind.flatten(),:]
            valid_indices=valid_ind
    return R, C, pts3D


def compute_rectification(K, R, C):
    #Calculate r_x
    r_x=C/np.linalg.norm(C)

    #Calculate r_z
    r_z_tilde=np.reshape(np.array([0,0,1]), (3,1))
    r_z=(r_z_tilde-np.dot(r_z_tilde.flatten(), r_x.flatten())*r_x)/np.linalg.norm((r_z_tilde-np.dot(r_z_tilde.flatten(), r_x.flatten())*r_x))

    #Calculate r_y
    r_y=np.reshape(np.cross(r_z, r_x, axisa=0, axisb=0), (3,1))

    #Build rectification rotation
    R_rect=np.vstack((r_x.T, r_y.T, r_z.T))

    #Ensure determinant of R_rect is one
    if(np.linalg.det(R_rect)<0):
        R_rect=-R_rect

    #Build homographies
    H1=K@R_rect@np.linalg.inv(K)
    H2=K@R_rect@R.T@np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2):
    disparity=np.zeros(img1.shape)
    sift=cv2.xfeatures2d.SIFT_create()
    #Build up the list of keypoints
    kps = []
    coords= []
    size=5
    for v in range(0,img2.shape[0]):
        for u in range(0,img2.shape[1]):
            #Use given size to build a keypoint
            kp = cv2.KeyPoint(u, v, size)
            kps.append(kp)
            coords.append([u,v])
    #Use sift.compute with our collected keypoints to get descriptors
    kps2,des2=sift.compute(img2, kps)
    kps1,des1=sift.compute(img1, kps)
    coords=np.array(coords)
    for i in range(len(des2)):
        #Sweep along the epipolar line
        pos_mask=np.logical_and(coords[:,1]==coords[i,1], coords[:,0]>=coords[i,0])
        pos_coords=np.argwhere(pos_mask)
        pos_features=des1[pos_coords.flatten(),:]
        #Get the disparity
        disparity[coords[i,1],coords[i,0]]=np.argmin(np.linalg.norm(des2[i]-pos_features,axis=1)**2)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    #save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
