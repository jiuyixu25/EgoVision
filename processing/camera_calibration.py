import cv2
import numpy as np
import glob

def calibrate_camera(calib_folder, chessboard_dim=(9, 6), criteria=None):
    """
    Calibrate the camera using chessboard images from a specified folder.

    Parameters:
        calib_folder (str): Path to the folder containing chessboard images (*.jpg).
        chessboard_dim (tuple): Number of inner corners per a chessboard row and column (w, h).
                                Default is (9, 6), where 9 corresponds to the number of inner corners along width.
        criteria (tuple): Termination criteria for the iterative algorithm to refine corner locations.
                          Default is (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001).

    Returns:
        mtx (ndarray): Camera matrix.
        dist (ndarray): Distortion coefficients.
        rvecs (list): Rotation vectors estimated for each calibration image.
        tvecs (list): Translation vectors estimated for each calibration image.
    """
    # Set default termination criteria if not provided.
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Unpack chessboard dimensions.
    w, h = chessboard_dim

    # Prepare object points: (0,0,0), (1,0,0), ...,(w-1, h-1, 0)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    objpoints = []  # 3D point in real world space.
    imgpoints = []  # 2D points in image plane.

    # Read all images from the calibration folder.
    images = glob.glob(f'{calib_folder}/*.jpg')
    for fname in images:
        print(f"Processing file: {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Unable to read image {fname}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        print(f"Chessboard found: {ret}")

        if ret:
            # Refine corner positions.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    # Perform camera calibration.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist, rvecs, tvecs

def undistort_image(image_file, mtx, dist, save_result=False, out_filename=None):
    """
    Undistort an image using the provided camera matrix and distortion coefficients.

    Parameters:
        image_file (str): Path to the image file to be undistorted.
        mtx (ndarray): Camera matrix obtained from calibration.
        dist (ndarray): Distortion coefficients obtained from calibration.
        save_result (bool): If True, the undistorted image will be saved to disk.
        out_filename (str): Output file name/path. If None and save_result is True, the input file will be overwritten.

    Returns:
        undistorted (ndarray): The undistorted image.
    """
    # Load the image.
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError(f"Unable to read image at {image_file}")

    h, w = img.shape[:2]
    # Get optimal new camera matrix.
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # Undistort the image.
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Optionally save the undistorted image.
    if save_result:
        if out_filename is None:
            out_filename = image_file
        cv2.imwrite(out_filename, undistorted)
        print(f"Undistorted image saved as: {out_filename}")

    return undistorted

# Example usage:
if __name__ == '__main__':
    # Calibrate the camera using images in the "chessboard" folder.
    calib_folder = 'chessboard'
    mtx, dist, rvecs, tvecs = calibrate_camera(calib_folder, chessboard_dim=(9, 6))
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Undistort a sample image "2_0000.jpg".
    test_image = '2_0000.jpg'
    undistorted_img = undistort_image(test_image, mtx, dist, save_result=True)