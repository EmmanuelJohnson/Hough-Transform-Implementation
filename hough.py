import numpy as np
import cv2

config = {
    'vertical':{
        'color': (0, 0, 255),
        'R': 20,
        'TH': 5,
        'thresh': 90
    },
    'diagonal':{
        'color': (255, 255, 0),
        'R': 25,
        'TH': 8,
        'thresh': 45
    },
    'circle':{
        'color': (0, 255, 255),
        'radius': 22,
        'R': 25,
        'TH': 25,
        'thresh': 138
    }
}
global CURRENT

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name, image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name, image):
    cv2.imwrite(name,image) 

#Perform normalization on the image
def norm(image):
    return (image/255).astype(np.uint8)

#Copy the image
def copy(image):
    return image.copy()

#Convert degree to radian
def to_radian(degree):
    return np.deg2rad(degree)

#Get all theta values in radian within a range
def get_thetas(start, end, step):
    thetas = np.arange(start, end, step)
    return to_radian(thetas), len(thetas)

#Convert an array of thetas to sin or cos
def convert_thetas(thetas, operation):
    if operation == 'cos':
        return np.cos(thetas)
    elif operation == 'sin':
        return np.sin(thetas)
    else:
        print('Unkown Operation')
        return None

#Vertical and Horizontal Sobel Matrix defined
def get_sobel(value):
    #Flipped Sobel X and Sobel Y
    sobels = {
        "sobelX": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "sobelY": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    }
    return sobels[value]

#Create a Matrix with all elements 0
def createZeroMatrix(r, c, t):
    return np.zeros((r,c), dtype=t)

#Add padding around the edge of the image
def add_padding(image,padding):
    return np.pad(image, (padding,padding), 'edge')

#Perform convolution between image and sobel
def convolution(image,sobelx,sobely):
    #Get image height and width
    ih, iw = image.shape[:2]
    
    #Create empty matrix to store the result
    op_mx_g = createZeroMatrix(ih, iw, 'float32')
    
    #Padding width
    pad = sobelx.shape[0]//2

    #Adding zero padding to the image
    img_matrix = add_padding(image, pad)
    
    #Sobel height and width
    sh, sw = sobelx.shape[0],sobelx.shape[1]
    
    for i in range(pad, ih + pad):
        for j in range(pad, iw + pad):

            sx,sy,sg = 0,0,0
            #The submatrix from the image on which convolution is done
            #Dimension is based on the dimension of the kernel being applied
            submatrix = img_matrix[i-pad:i-pad+sh,j-pad:j-pad+sw]
            
            for si in range(sh):
                for sj in range(sw):
                    sx = sx + (sobelx[si][sj] * submatrix[si][sj])
                    sy = sy + (sobely[si][sj] * submatrix[si][sj])
            #Finding the gradient of the two sobels
            sg = np.sqrt((sx * sx) + (sy * sy))
            op_mx_g[i-1][j-1] = sg
    return op_mx_g

#rho = x cos(theta) * y sin(theta)
def normal_line_eq(x,y,sin,cos,diag):
    xcos = x*cos
    ysin = y*sin
    return int(xcos+ysin+diag)

#a = x - radius * cos(theta)
#b = y - radius * sin(theta)
def normal_circle_eq(x,y,sin,cos,r):
    rcos = r * cos
    rsin = r * sin
    rh = int(x - rcos)
    th = int(y - rsin)
    return rh, th

#Compute the accumulator matrix 
def get_accumulator(image, sines, coses, diag, n_range, T):
    h,w = image.shape[:2]
    if diag is not None:
        acc_shape = tuple([int(2*diag),n_range])
    else:
        acc_shape = (h,w)
    res_acc = np.zeros(acc_shape, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if image[i,j] >= T:
                for theta in range(n_range):
                    if diag is not None:
                        rho = normal_line_eq(i,j,sines[theta],coses[theta],diag)
                        #Voting
                        res_acc[rho,theta]+=1
                    else:
                        rh, th = normal_circle_eq(i,j,sines[theta],coses[theta],CURRENT['radius'])
                        if 0<=rh<h and 0<=th<w:
                            #Voting
                            res_acc[rh,th]+=1
    return res_acc

#Compute x1 and x2 value
def get_x(sin,cos,rho,diag,mx):
    rmind = rho - diag
    x1 = int((1-cos/rmind)*(rmind/sin))
    x2 = int((1-mx*cos/rmind)*(rmind/sin))
    return x1, x2

#Draw lines for a given set of points
def draw_lines(image,lines,color):
    for l in lines:
        cv2.line(image,l[0],l[1],color,2)
    return image

#Draw Circles around given points
def draw_circles(image, circles, r, color):
    for c in circles:
        cv2.circle(image, c, r, color, 2)
    return image

#Check if the vote is the highest in a given range
def check_peak(vote, acc, rho, theta, diag):
    R, TH = CURRENT['R'], CURRENT['TH']
    r,t = acc.shape[:2]
    r1,r2 = 0 if rho-R<0 else rho-R, r if rho+R>r else rho+R
    t1,t2 = 0 if theta-TH<0 else theta-TH, t if theta+TH>t else theta+TH
    if diag is not None:
        rmind = rho - diag
    else:
        rmind = 1
    if rmind == 0:
        return False
    for i in range(r1,r2):
        for j in range(t1,t2):
            if acc[i,j]>vote:
                return False
    return True

#Detect vertical lines in the given image
def get_vertical_lines(image, ih, iw, T):
    pre_process_kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(cv2.morphologyEx(image, cv2.MORPH_CLOSE, pre_process_kernel),pre_process_kernel,1)
    len_diag = int(np.sqrt(ih**2+iw**2))+1
    thetas, n_thetas = get_thetas(-100, -80, 1)
    cos_thetas = convert_thetas(thetas, 'cos')
    sin_thetas = convert_thetas(thetas, 'sin')
    line_accumulator = get_accumulator(img, sin_thetas, cos_thetas, len_diag, n_thetas, 180)
    ah,aw = line_accumulator.shape[:2]
    points = list()
    for rho in range(ah):
        for theta in range(aw):
            if line_accumulator[rho,theta] > T:
                #Check if the vote is the highest by comparing it with its neighbors
                #If it is the highest, then consider it else ignore
                if check_peak(line_accumulator[rho,theta],line_accumulator,rho,theta,len_diag):
                    x1, x2 = get_x(sin_thetas[theta], cos_thetas[theta],rho,len_diag,ih)
                    points.append([tuple([x1,0]),tuple([x2,ih])])
    return points

#Detect diagonal lines in the given image
def get_diagonal_lines(image, ih, iw, T):
    len_diag = int(np.sqrt(ih**2+iw**2))+1
    thetas, n_thetas = get_thetas(-65, -45, 1)
    cos_thetas = convert_thetas(thetas, 'cos')
    sin_thetas = convert_thetas(thetas, 'sin')
    diag_accumulator = get_accumulator(image, sin_thetas, cos_thetas, len_diag, n_thetas, 150)
    ah,aw = diag_accumulator.shape[:2]
    points = list()
    for rho in range(ah):
        for theta in range(aw):
            if diag_accumulator[rho,theta] > T:
                #Check if the vote is the highest by comparing it with its neighbors
                #If it is the highest, then consider it else ignore
                if check_peak(diag_accumulator[rho,theta],diag_accumulator,rho,theta,len_diag):
                    x1, x2 = get_x(sin_thetas[theta], cos_thetas[theta],rho,len_diag,ih)
                    points.append([tuple([x1,0]),tuple([x2,ih])])
    return points

#Detect circles in the given image
def get_circles(image, ih, iw, T):
    thetas, n_thetas = get_thetas(0, 360, 1)
    cos_thetas = convert_thetas(thetas, 'cos')
    sin_thetas = convert_thetas(thetas, 'sin')
    circle_accumulator = get_accumulator(image, sin_thetas, cos_thetas, None, n_thetas, 170)
    save_image('CircleAccumulator.jpg', circle_accumulator)
    points = list()
    for rho in range(ih):
        for theta in range(iw):
            if circle_accumulator[rho,theta] >= T:
                #Check if the vote is the highest by comparing it with its neighbors
                #If it is the highest, then consider it else ignore
                if check_peak(circle_accumulator[rho,theta],circle_accumulator,rho,theta,None):
                    points.append(tuple([theta,rho]))
    return points

if __name__ == '__main__':
    
    print('__Reading the given image : ../original_imgs/hough.jpg__\n')
    img = get_image_gray('../original_imgs/hough.jpg')
    cimg = get_image('../original_imgs/hough.jpg')
    simg = get_image_gray('hough_edges.jpg')
    ih, iw = img.shape[:2]

    print('__Detecting Edges Using Sobel__\n')
    if simg is None:
        sobelX = np.asarray(get_sobel('sobelX'),dtype='float32')
        sobelY = np.asarray(get_sobel('sobelY'),dtype='float32')
        sobeled_img = convolution(img,sobelX,sobelY)
        save_image('hough_edges.jpg', sobeled_img)
    else:
        sobeled_img = simg
    
    print('__Task 3 (a)__\n')
    print('__Detecting Vertical Lines__\n')
    CURRENT = config['vertical']
    vertical_lines = get_vertical_lines(sobeled_img, ih, iw, CURRENT['thresh'])
    print('__No of vertical lines detected__  : ', len(vertical_lines), '\n')
    vertical_lines_image = draw_lines(copy(cimg),vertical_lines, CURRENT['color'])
    save_image('red_line.jpg', vertical_lines_image)
    
    print('__Task 3 (b)__\n')
    print('__Detecting Diagonal Lines__\n')
    CURRENT = config['diagonal']
    diagonal_lines = get_diagonal_lines(sobeled_img, ih, iw, CURRENT['thresh'])
    print('__No of diagonal lines detected__  : ', len(diagonal_lines), '\n')
    diagonal_lines_image = draw_lines(copy(cimg),diagonal_lines, CURRENT['color'])
    save_image('blue_line.jpg', diagonal_lines_image)
    
    print('__Task 3 (c)__\n')
    print('__Detecting Circles__\n')
    print('##### this might take a while #####\n')
    CURRENT = config['circle']
    circles = get_circles(sobeled_img, ih, iw, CURRENT['thresh'])
    print('__No of circles detected__  : ', len(circles), '\n')
    circles_image = draw_circles(copy(cimg), circles, CURRENT['radius'], CURRENT['color'])
    save_image('coin.jpg', circles_image)