import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mat_img
import cv2
cap=cv2.VideoCapture("C:/Users/user/PycharmProjects/Computer_Vision_Projects/challenge/project_video.mp4")
for i in range(1000):

    ret,frame=cap.read()

    img=frame
    read_dictionary = np.load('my_file1.npy', allow_pickle='TRUE').item()
    mat=read_dictionary['matrix']  # displays "world"

    read_dictionary = np.load('my_file2.npy', allow_pickle='TRUE').item()
    dist = read_dictionary['dist']  # displays "world"

    img = cv2.undistort(img, mat, dist, None, mat)

    def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
        hls=cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        gray = hls[:,:,2]
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    mag_binary=abs_sobel_thresh(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()

    src=np.float32([[565,429],
                    [310,688],
                    [928,596],
                    [674,446]])

    dst=np.float32([[300,400],
                    [300,701],
                    [980,701],
                    [980,400]])

    M=cv2.getPerspectiveTransform(src,dst)
    M2=cv2.getPerspectiveTransform(dst,src)



    warped=cv2.warpPerspective(mag_binary,M,(mag_binary.shape[1],mag_binary.shape[0]),flags=cv2.INTER_LINEAR)
    ww = warped.shape[1]
    hh = warped.shape[0]


    #plt.imshow(warped,cmap='gray')
    #plt.show()

    def hist(img):
        bottom_img= img[500:700,:]
        summation=np.sum(bottom_img,axis=0)
        return summation

    histogram=hist(warped)
    midpoint = np.int(histogram.shape[0]//2)


    left_max= np.argmax(histogram[:midpoint])
    right_max=np.argmax(histogram[midpoint:])+midpoint

    warped = warped[450:]
    #plt.imshow(warped)
    #plt.show()



    nwindows=100
    window_height= np.int(warped.shape[0]//nwindows)
    margin=10
    minpix = 20

    def extract_left_and_rigth_fit(warped):



        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current=left_max
        rightx_current=right_max
        left_lane_inds=[]
        right_lane_inds=[]
        new_img=warped[400:].copy()
        #plt.imshow(new_img)
        #plt.show()


        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(new_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (140, 255, 0), 2)
            cv2.rectangle(new_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (140, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]


            left_lane_inds= (np.append(left_lane_inds,good_left_inds))
            right_lane_inds= (np.append(right_lane_inds,good_right_inds))
            left_lane_inds=left_lane_inds.astype(int)
            right_lane_inds=right_lane_inds.astype(int)



            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))





        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit,right_fit


    left_fit, right_fit = extract_left_and_rigth_fit(warped)




    def fit_poly(img_shape, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty
    def search_around_poly(binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 40

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = window_img


        # Plot the polynomial lines onto the image


        ## End visualization steps ##

        return result


    # Run image through the pipeline
    # Note that in your project, you'll also want to feed in the previous fits

    result=search_around_poly(warped)

    ht, wd, cc = result.shape

    color = (0, 0, 0)
    new_result = np.full((hh, ww, cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    # copy img image into center of result image
    new_result[450:] = result

    #plt.imshow(new_result)
    #plt.show()

    hehe=cv2.warpPerspective(new_result,M2,(new_result.shape[1],new_result.shape[0]),flags=cv2.INTER_LINEAR)
    #plt.imshow(hehe)
    #plt.show()



    pp=cv2.addWeighted(img,1,hehe,0.3,0.0)
    # View your output

    cv2.imshow('frame', pp)
    if cv2.waitKey(1) == ord('q'):
        break






