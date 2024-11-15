import numpy as np
from numpy import pad
from scipy.ndimage import uniform_filter, gaussian_filter, convolve
import cv2
import os

def calculate_cci(image, tolerance):
    """
    Calculate the contrast code image (CCI) as proposed in the paper
    "A Contrast-Guided Approach for the Enhancement of Low-Lighting Underwater Images"
    by Tunai P. Marques, Alexandra Branzan-Albu and Maia Hoeberechts.
    
    Args:
        image: RGB image as numpy array (height, width, 3)
        tolerance: parameter that defines the priority that bigger patch sizes will have
        
    Returns:
        CCI: 2D array with the same dimensions as input image, where each value
             indicates the patch size that generated the smallest local standard deviation
    """
    
    # Convert image to double precision for calculations
    A = image.astype(np.float64)
    x, y, z = A.shape
    
    # Define patch sizes to be tested (usual range is [15 13 ... 5 3])
    psize2 = np.array([15, 13, 11, 9, 7, 5, 3])
    
    # Calculate tolerance factor (weight decay)
    tol = 1 - (tolerance/100)
    
    # Create tolerance array for prioritizing bigger patch sizes
    tolerance_array = np.array([1*(tol**6), 1*(tol**5), 1*(tol**4), 
                              1*(tol**3), 1*(tol**2), 1*(tol**1), 1])
    
    # Reshape and repeat tolerance array to match dimensions
    tolerance_array = tolerance_array.reshape(1, 1, -1)
    tolerance_matrix = np.tile(tolerance_array, (x, y, 1))
    
    # Create score image for storing standard deviations
    score_temp = np.zeros((x, y, len(psize2)))
    
    # Convert to grayscale using ITU-R BT.601-7 weights
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def local_std(img, window_size):
        """Calculate local standard deviation using uniform filter"""
        c1 = uniform_filter(img, window_size, mode='reflect')
        c2 = uniform_filter(img*img, window_size, mode='reflect')
        return np.sqrt(abs(c2 - c1*c1))
    
    # Calculate standard deviation for each patch size
    for i, patch_size in enumerate(psize2):
        score_temp[:,:,i] = local_std(gray, patch_size)
    
    # Apply tolerance weights to prioritize larger patch sizes
    score_temp = score_temp * tolerance_matrix
    
    # Find patch size with minimum standard deviation
    CCI = np.argmin(score_temp, axis=2)
    
    print("Contrast code image calculated.")
    return CCI


def contrastGuidedAL(dc, CCI, mult=3):
    prange_biggest = round(15/2)-1
    extended = pad(dc, ((prange_biggest*mult, prange_biggest*mult), 
                   (prange_biggest*mult, prange_biggest*mult)), 
              mode='reflect')
    
    x, y = dc.shape
    result = np.zeros((x,y))
    
    m = mult
    upsilon = np.array([3*m - ((m/3)*(i-1)) for i in range(1, 8)])
    # print("\nUpsilon", upsilon)

    upsilon = np.floor(upsilon/2)

    for i in range(x):
        for j in range(y):
            cpx = i + (prange_biggest*mult)
            cpy = j + (prange_biggest*mult)
            
            c = CCI[i, j]
            prange = int(upsilon[6-c])

            patch = extended[cpx-prange:cpx+prange+1, cpy-prange:cpy+prange+1]
            # print(*list(patch), sep=" ")
            result[i, j] = np.max(patch[:])

    return result


def luwe_input(image, CCI, w, t0, filter, atmLightPatchMul, name, save):
    if not os.path.exists(name):
        os.mkdir(name)

    A = image.astype(np.float64)
    image[image < 10] = 0
    x, y, z = A.shape

    biggest_psize = 15
    prange_biggest = round(biggest_psize/2)-1
    extended = pad(A, ((prange_biggest, prange_biggest),
                  (prange_biggest, prange_biggest),
                  (0, 0)),  # no padding on color channels
              mode='reflect')

    dcs = np.zeros((x, y, 3))

    for i in range(x):
        for j in range(y):
            cpx = i + (prange_biggest)
            cpy = j + (prange_biggest)
            prange = CCI[i, j]
            prange = 6-prange

            patch = extended[cpx-prange:cpx+prange+1, cpy-prange:cpy+prange+1]

            dcs[i,j,0] = np.min(patch[:,:,0])
            dcs[i,j,1] = np.min(patch[:,:,1])
            dcs[i,j,2] = np.min(patch[:,:,2])

    atmLightImage = np.zeros((x,y,3))

    atmLightImage[:, :, 0] = contrastGuidedAL(dcs[:,:,0], CCI, atmLightPatchMul)
    atmLightImage[:, :, 1] = contrastGuidedAL(dcs[:,:,1], CCI, atmLightPatchMul)
    atmLightImage[:, :, 2] = contrastGuidedAL(dcs[:,:,2], CCI, atmLightPatchMul)

    if (filter == 1):
        atmLightImageFilt = gaussian_filter(atmLightImage,10)
    else:
        atmLightImageFilt = atmLightImage;

    transmm = np.zeros((x, y))
    
    normalized = np.ones_like(A)
    normalized[:,:,0] = A[:,:,0] / atmLightImageFilt[:,:,0]
    normalized[:,:,1] = A[:,:,1] / atmLightImageFilt[:,:,1]
    normalized[:,:,2] = A[:,:,2] / atmLightImageFilt[:,:,2]
    normalized[np.isnan(normalized)] = 0

    for i in range(x):
        for j in range(y):
            cpx = i + (prange_biggest)
            cpy = j + (prange_biggest)
            
            prange = CCI[i, j]
            prange = 7-prange
            # print("cpx: ", cpx)
            # print("cpy: ", cpy)
            # print("prange: ", prange)
            # print("NORM SHAPE", normalized.shape)

            patch = normalized[cpx-prange:cpx+prange+1, cpy-prange:cpy+prange+1]
            # print("PATCH SHAPE", patch.shape)
            
            if not 0 in patch.shape:
                transmm[i, j] = 1 - w*np.min(patch)
                # if transmm[i,j] == 1:
                #     print("wtf")

    tmr = transmm

    if (filter==1):
        p = transmm.astype(np.float32)
        r = 16
        s = 4
        eps = 0.45
        transmm = cv2.ximgproc.guidedFilter(guide=p, src=p, radius=r, eps=eps, dDepth=-1)

    tmd = transmm

    J = np.zeros_like(A, dtype=np.float32)

    # Rescale intensities of each pixel in the three channels of the original image
    a_r = cv2.normalize(A[:, :, 0].astype(np.uint8), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    a_g = cv2.normalize(A[:, :, 1].astype(np.uint8), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    a_b = cv2.normalize(A[:, :, 2].astype(np.uint8), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Normalize atmospheric lighting to the [0, 1] range
    atm_lighting = atmLightImageFilt / 255.0

    x, y, _ = A.shape
    for i in range(x):
        for j in range(y):
            # Radiance calculation for the red channel
            diff_r = a_r[i, j] - atm_lighting[i, j, 0]
            J[i, j, 0] = diff_r / max(transmm[i, j], t0) + atm_lighting[i, j, 0]
            
            # Radiance calculation for the green channel
            diff_g = a_g[i, j] - atm_lighting[i, j, 1]
            J[i, j, 1] = diff_g / max(transmm[i, j], t0) + atm_lighting[i, j, 1]
            
            # Radiance calculation for the blue channel
            diff_b = a_b[i, j] - atm_lighting[i, j, 2]
            J[i, j, 2] = diff_b / max(transmm[i, j], t0) + atm_lighting[i, j, 2]
    
    if save == 1:
        prefix = f"{name}{atmLightPatchMul}"

        # Save the filtered transmission map
        filtered_transmission_map = np.nan_to_num(tmd * 255, nan=0, posinf=255, neginf=0).astype(np.uint8)
        cv2.imwrite(f"{prefix}tmap_filtered.png", filtered_transmission_map)

        # Save the raw transmission map
        raw_transmission_map = np.nan_to_num(tmr * 255, nan=0, posinf=255, neginf=0).astype(np.uint8)
        cv2.imwrite(f"{prefix}tmap_raw.png", raw_transmission_map)

        # Save the filtered atmospheric lighting image, scaled to [0, 1] then to [0, 255]
        filtered_atm_light = np.nan_to_num(atmLightImageFilt * 255, nan=0, posinf=255, neginf=0).astype(np.uint8)
        cv2.imwrite(f"{prefix}atml_filtered.png", filtered_atm_light)

        # Save the raw atmospheric lighting image, scaled to [0, 1] then to [0, 255]
        raw_atm_light = np.nan_to_num(atmLightImage * 255, nan=0, posinf=255, neginf=0).astype(np.uint8)
        cv2.imwrite(f"{prefix}atml_raw.png", raw_atm_light)

        # Convert the dark channel (dcs) to grayscale and save
        dcs_gray = 0.2989 * dcs[:, :, 2] + 0.5870 * dcs[:, :, 1] + 0.1140 * dcs[:, :, 0]
        print(dcs_gray.shape)
        cv2.imwrite(f"{name}dc.png", dcs_gray)


    J = np.clip(J, 0, 1)

    return J


def contrastWM(input, laplacianKernel):
    luminance = np.mean(input, axis=2)
    return np.abs(convolve(luminance, laplacianKernel, mode='reflect'))

def lumincanceWM(input):
    luminance = np.mean(input, axis=2)
    Bl = input[:,:,0]-luminance
    Gl = input[:,:,1]-luminance
    Rl = input[:,:,2]-luminance

    mos = (Rl**2 + Gl**2 + Bl**2) / 3
    return np.sqrt(mos)

def saliencyWM(input):
    # Create a 5x5 separable binomial kernel (equivalent to high frequency cut-off of pi/2.75)
    a = np.array([1, 4, 6, 4, 1]) / 16
    Gkernel = np.outer(a, a)

    if input.dtype != np.float32:
        input = input.astype(np.float32) / 255.0

    print("input.shape", input.shape)


    oneC = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    meank = np.mean(oneC)
    gaussian_smoothed = convolve(oneC, Gkernel, mode='reflect')
    out = np.abs(gaussian_smoothed - meank)

    return out



def multiresolutionPyramid(img, num_levels=5):
    """
    Create a multiresolution pyramid from input image.
    
    Parameters:
    -----------
    img : ndarray
        Input image (2D or 3D array)
    num_levels : int, optional
        Number of pyramid levels. If None, computed automatically to keep
        smallest level at least 32x32
        
    Returns:
    --------
    list
        List of images forming the multiresolution pyramid
    """
    # Convert image to float and ensure range 0-1
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
        
    M, N = img.shape[:2]
    
    # # Calculate number of levels if not specified
    # if num_levels is None:
    #     lower_limit = 32
    #     num_levels = min(int(np.log2([M, N]) - np.log2(lower_limit))) + 1
    # else:
    #     num_levels = min(num_levels, min(int(np.log2([M, N]))) + 2)
    
    # Calculate padding
    smallest_size = np.ceil(np.array([M, N]) / (2 ** (num_levels - 1)))
    padded_size = smallest_size * (2 ** (num_levels - 1))
    pad_height = int(padded_size[0] - M)
    pad_width = int(padded_size[1] - N)
    
    # Pad image
    if len(img.shape) == 3:
        padded_img = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')
    else:
        padded_img = np.pad(img, ((0, pad_height), (0, pad_width)), mode='edge')
    
    # Create pyramid
    mrp = [None] * num_levels
    mrp[0] = padded_img
    
    for k in range(1, num_levels):
        mrp[k] = cv2.resize(mrp[k-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    
    # Replace first level with original unpadded image
    mrp[0] = img
    
    return mrp

def laplacianPyramid(mrp):
    """
    Create a Laplacian pyramid from a multiresolution pyramid.
    
    Parameters:
    -----------
    mrp : list
        Multiresolution pyramid (list of images)
        
    Returns:
    --------
    list
        Laplacian pyramid
    """
    num_levels = len(mrp)
    lapp = [None] * num_levels
    lapp[-1] = mrp[-1]  # Copy the smallest level
    
    for k in range(num_levels - 1):
        A = mrp[k]
        B = cv2.resize(mrp[k+1], (A.shape[1], A.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        lapp[k] = A - B
        
    return lapp

def reconstructFromLaplacianPyramid(lapp):
    """
    Reconstruct original image from a Laplacian pyramid.
    
    Parameters:
    -----------
    lapp : list
        Laplacian pyramid
        
    Returns:
    --------
    ndarray
        Reconstructed image
    """
    num_levels = len(lapp)
    out = lapp[-1]
    
    for k in range(num_levels - 2, -1, -1):
        out = cv2.resize(out, (lapp[k].shape[1], lapp[k].shape[0]), interpolation=cv2.INTER_LANCZOS4)
        out = out + lapp[k]
        
    return out


def multiscalefusion(input1, input2, 
                     regTerm, pyramidLevels, laplacianKernel, 
                     display, outPath, save):
    
    saliencyWM1 = saliencyWM(input1)
    saliencyWM2 = saliencyWM(input2)

    luminanceWM1 = lumincanceWM(input1)
    luminanceWM2 = lumincanceWM(input2)

    contrastWM1 = contrastWM(input1, laplacianKernel)
    contrastWM2 = contrastWM(input2, laplacianKernel)

    aggregatedWM1 = (saliencyWM1 + luminanceWM1 + contrastWM1) + regTerm
    aggregatedWM2 = (saliencyWM2 + luminanceWM2 + contrastWM2) + regTerm

    total = (aggregatedWM1 + aggregatedWM1) + (2*regTerm)

    normalizedWM1 = aggregatedWM1/total
    normalizedWM2 = aggregatedWM2/total

    if save == 1:
        # Rescale each image by multiplying by 255 and converting to uint8
        cv2.imwrite(os.path.join(outPath, 'ContrastWM1.png'), np.nan_to_num(contrastWM1 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'ContrastWM2.png'), np.nan_to_num(contrastWM2 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'SaliencyWM1.png'), np.nan_to_num(saliencyWM1 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'SaliencyWM2.png'), np.nan_to_num(saliencyWM2 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'LuminanceWM1.png'), np.nan_to_num(luminanceWM1 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'LuminanceWM2.png'), np.nan_to_num(luminanceWM2 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'input1.png'), np.nan_to_num(input1 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'input2.png'), np.nan_to_num(input2 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'NormalizedWM1.png'), np.nan_to_num(normalizedWM1 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))
        cv2.imwrite(os.path.join(outPath, 'NormalizedWM2.png'), np.nan_to_num(normalizedWM2 * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))


    # Construct the Gaussian pyramids for the normalized weight maps
    gaussPyramid1 = multiresolutionPyramid(normalizedWM1, pyramidLevels)
    gaussPyramid2 = multiresolutionPyramid(normalizedWM2, pyramidLevels)

    # Calculate Laplacian pyramids of the inputs
    mrp1 = multiresolutionPyramid(input1, pyramidLevels)
    mrp2 = multiresolutionPyramid(input2, pyramidLevels)
    lapp1 = laplacianPyramid(mrp1)
    lapp2 = laplacianPyramid(mrp2)

    print("\nINPUT SHAPES", gaussPyramid1[0].shape, gaussPyramid2[0].shape,
          lapp1[0].shape, lapp2[0].shape, "\n")

    # Initialize the fused pyramid list
    fusedPyramid = []

    # Fuse the pyramids
    for i in range(pyramidLevels):
        if gaussPyramid1[i].shape[:2] != lapp1[i].shape[:2]:
            print(f'Different sizes in level {i+1} of the pyramid. Check its construction.')

        # Eq. (13) from [1]
        fused_layer = np.stack((gaussPyramid1[i],)*3, axis = -1) * lapp1[i] + np.stack((gaussPyramid2[i],)*3, axis = -1) * lapp2[i]
        fusedPyramid.append(fused_layer)

    # Reconstruct the final result from the fused Laplacian pyramid
    result = reconstructFromLaplacianPyramid(fusedPyramid)

    return result

def main():
    img = cv2.imread("data/181.jpg")
    tol = 3
    filter = 0
    multiplier = [5, 30]
    w = 0.9
    t0 = 0.02

    # Fusion-related parameters
    laplacian_kernel = np.array([[-1, -1, -1], 
                                [-1,  8, -1], 
                                [-1, -1, -1]]) / 8  # Laplacian kernel used on the local contrast weight map

    reg_term = 0.001  # A term that guarantees that each input contributes to the multi-scale fusion output  
    pyramid_levels = 5
    display_multi_fusion = 0

    savepath = "out/"
    save = 1
    
    inverted_image = cv2.bitwise_not(img)
    CCI = calculate_cci(inverted_image, tol)
    
    # Scale and convert to uint8 before applying bitwise_not
    luwe_output1 = luwe_input(inverted_image, CCI, w, t0, filter, multiplier[0], savepath, save)
    input1 = cv2.bitwise_not((luwe_output1 * 255).astype(np.uint8)).astype(np.float32) / 255.0

    luwe_output2 = luwe_input(inverted_image, CCI, w, t0, filter, multiplier[1], savepath, save)
    input2 = cv2.bitwise_not((luwe_output2 * 255).astype(np.uint8)).astype(np.float32) / 255.0

    

    result = multiscalefusion(input1, input2, reg_term, pyramid_levels,
                              laplacian_kernel, display_multi_fusion,
                              savepath, save)

    cv2.imwrite(os.path.join(savepath, "inverted.jpg"), inverted_image.astype(np.uint8))
    print(result.shape)
    cv2.imwrite(os.path.join(savepath, "final.jpg"), np.nan_to_num(result * 255, nan=0, posinf=255, neginf=0).astype(np.uint8))

if __name__ == "__main__":
    main()