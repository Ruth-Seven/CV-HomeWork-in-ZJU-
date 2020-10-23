



# CV Assigement1 10.17

> 胡竣杰 人工智能 2001班 22051236

这次的课程作业完成内容咋多，涉及基础知识多，理解数学公式多，实现细节复杂，查阅资料更多，希望老师不要太苛求我们。

## 0. 数据准备

从[Affine数据源](http://www.robots.ox.ac.uk/~vgg/data/affine/)下载了另一个特征数据图像库的8组图像，每组图像都有一个唯一的`label`。按照PPT要求我们从48张图片中尽量均匀的选择出10张图片，作为`test dataset`，剩下的图片作为`train dataet`以便提取特征。

## 1. Extract SIFT  features from image dataset

特征部分要求实现一个`SIFT`函数。SIFT函数从每张图像中抽取`keypoints`和对应的`descroptors`，其数据结构应包含`opencv-python`提供的`SIFT`实现的函数返回的结构功能。

使用opencv观察图像可知，每组的6个图像是对同一景物的不同角度，不同远近和不同拍摄地点从而获取。因此6张图像中存在相同的景物主体，但是主体的形态可能发生了`rotation`，`translation`和`scale`。`G.lowe`大佬提出的SIFT特征对尺度扩缩、明亮变化和旋转都是保持稳定的不变性。在局部图像上能够有效提取特征点，并生成描述符。

![image-20201023150821589](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023150821589.png)

比如最后的在测试集查询的结果中，可以看到两个匹配的图片虽然在同一`label`之下，但是具有相同但不完全相同的主体。

### 1.1 The four step of Scala Invariant Feature Transform 

SIFT算法的提取和计算过程如下，其部分代码也摘录在其中。

```python
def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """
    计算 keypoints and descriptors
    """
    image = image.astype('float32')
    # blur and double the original picture
    base_image = generateBaseImage(image, sigma, assumed_blur)
    # 通过图像大小计算机octaves的层数
    num_octaves = computeNumberOfOctaves(base_image.shape)
    # 创建不同 sigma 的高斯核函数
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    # 在baseimage上使用不同的高斯核函数生成 不同程度 blured图像
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    # 由 octave 图像计算 DoG图像金字塔
    dog_images = generateDoGImages(gaussian_images)
    # 在DoG金字塔中通过相邻三层的图像寻找极值点
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    # 移除无效的 keypoints
    keypoints = removeDuplicateKeypoints(keypoints)
    # 把keypoint转化为原图像所对应的 keypoints
    keypoints = convertKeypointsToInputImageSize(keypoints)
    # 为每个keypoints 生成 Descriptors
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints, descriptors

```

#### 1.1.1 Detection of scale-space extrema  

##### `scale-space`

为了让计算机识别出不同`scale-space`的图像特色，我们需要构建不同`scale`的图像组喂给计算机，由此`scale pyramid`概念便可以提出来，各个`scale`的图像经过降采样组成`pyramid`各层的`baseimage`。从视觉角度来说。不同`scale`的图像应该有不同程度的模糊，自然的`blured`或者说对图像进行平滑的概念就合理的提出来了。对`base image`进行`blur`，依次生成同一组`octave`。

`Gaussian scale-sapce`就是在`scale-space`的基础上进行高斯模糊，其中，$\sigma$尺度空间因子 代表了模糊范围的大小，$L(x,y,\sigma)$代表了高斯模糊尺度空间。
$$
L(x,y,\sigma)=G(x,y,\sigma)\ast I(x,y)
$$ {\tag{1.3}


$G(x,y,\sigma)$就是指高斯卷积核：
$$
G(x,y,\sigma)=\frac{1}{2\pi \sigma^2} e^{\frac{x^2+y^2}{2\sigma ^2}}
$$
我们可以使用$\Delta^2G$ 高斯拉普拉斯（LoG）算子来检测计算点：
$$
\Delta^2 = \frac{\partial ^2}{\partial x^2} + \frac{\partial ^2}{\partial y^2}
$$
##### `DoG`

但是由于过大的计算量，可以优化为：
$$
\begin{split}
D(x,y,\sigma) &= [G(x,y,k\sigma) - G(x,y,\sigma)] \ast I(x,y) \\
 &= L(x,y,k\sigma) - L(x,y,\sigma)
\end{split}
$$

由此我们可以直接将两个相邻高斯空间相减就直接得到了`DoG`的响应图像。详细的来说，`image pyramid`中每个`octave`的`scale`不同，每组的图像的$\sigma$不同。上述公式的$k=2^{\frac{1}{k}}$，因为按照每相邻`octave`之间的降采样比例为2，$S$为每个`octave`的个数，$S$张图片刚刚够分。那么按照公式所述，将同一`octave`之间的图像相减就可以获得`DoG`。

![image-20201023160258356](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023160258356.png)



为了寻找`scale-space`的`extrema`，每个像素点都要和同一`scale-space`和的尺度域。当一个像素点比其周围的26个像素点都大或者都小的时，它就是极大值点或者极小值点。为了每个`octave`的第一组和最后一组都有数值可比，我们又多放缩出两个层面，故每个`octave of DoG`有$S+2$面图像。

```python
def generateDoGImages(gaussian_images):
    """
    从高斯 image pyramid 获取 两个图像的差分，计算出GOD pyramid
    """
    logger.info('Generating Difference-of-Gaussian images...')
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images)
```

##### 去除特征点

因为之前的特征点是在离散空间中近似求取到的，所以其极值点不一定是真正的极值点。主要有两种不符合条件的极值点：低对比度和不稳定的边缘相应点。我们可以通过尺度空间`DoG`函数进行曲线拟合寻找极值点。

设特点点$x$，偏移量$\Delta$，对比度为$D(x)$：
$$
D(x) =D + \frac{\partial D^T}{\partial x}\Delta x + \frac{1}{2}\Delta x^T\frac{\partial D}{\partial x^2}\Delta x
$$
在带入$\Delta x$有：
$$
D(\hat{x}) = D + \frac{1}{2} \frac{\partial D^T}{\partial x} \hat{x}
$$


若获得的近似结果$|D(\hat{x})| \geq T$ ，则特征点保留，否则舍去。

对于不稳定的边缘响应点。在边缘梯度的方向上主曲率值比较大，而沿着边缘方向则主曲率值较小。候选特征点的`DoG`函数`D(x)`的主曲率**Hessian**矩阵H的特征值成正比。
$$
H = \begin{bmatrix}
	D_{xx} & D_{xy} \\
	D_{xy} & D_{yy} 
\end{bmatrix}
$$
其中$D$是对相邻像素进行差分求得的。又由于$H$矩阵的特征值与曲率成正比，设置一个一个阈值为特征值的下限。为了避免计算特征值，设$r=\frac{\alpha}{\beta}$, 为最大特征值和最小特征值的比值。需要检测：
$$
\frac{f(r+1)^2}{r } > \frac{(T_r+1)^2}{T_r}
$$
成立的特征点保留。

```python
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  
    keypoints = []

    # i, j 是极值点？
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints



def computeHessianAtCenterPixel(pixel_array):
    """
    近似计算机 Hessian matrix
    """
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

```

#### 1.1.2 Orientation assignment

在不同尺度下都存在的特征点，给特征点的方向进行赋值。利用特征点邻域像素的梯度分布特性来确定其方向参数，再利用图像的梯度直方图求取关键点局部结构的稳定方向。
$$
\begin{split}
m(x,y) &= \sqrt{[L(x+1,y)-L(x-1,y)]^2+[L(x,y+1)-L(x,y-1)]^2}\\
\theta(x,y)&=tan^{-1}((L(x,y+1)-L(x,y-1))/(L(x+1,y)-L(x-1,y)))
\end{split}
$$
计算得到梯度的方向之后，利用直方图统计各个方向的梯度和幅值。横轴就是角度，360度就可以分为36个方向，纵轴就是各个梯度方向的幅度累计。当存在一个柱子的峰值可以认为该方向就是主方向。相当于主峰值80%能量的柱值时，则可以将这个方向认为是该特征点辅助方向。所以，一个特征点可能检测到多个方向

于是，可以对每个特征点得到三个信息$(x,y,\sigma,\theta)$，即位置、尺度和方向。由此可以确定一个SIFT特征区域，一个SIFT特征区域由三个值表示，中心表示特征点位置，半径表示关键点的尺度，箭头表示主方向。具有多个方向的关键点可以被复制成多份，然后将方向值分别赋给复制后的特征点，一个特征点就产生了多个坐标、尺度相等，但是方向不同的特征点

```python
def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    计算每个一个keypoints的方向 orientations
    """
    logger.info('Computing keypoint orientations...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations
```

##### 1.1.3 The local image descriptor

`keypoints`的关键值已经计算出来，但是如何以方便的数学方式描述需要`descriptor`来表达。`descriptor`不仅仅应该包括特征点的信息，也应该包括周围像素的信息。其生成步骤为：

1. 校正旋转主方向，确保旋转不变性。
2. 生成描述子，最终形成一个128维的特征向量
3. 归一化处理，将特征向量长度进行归一化处理，进一步去除光照的影响。

为了保证特征矢量的旋转不变性，要以特征点为中心，在附近邻域内将坐标轴旋转θθ（特征点的主方向）角度，即将坐标轴旋转为特征点的主方向。旋转后邻域内像素的新坐标为：
$$
\begin{bmatrix}
x'\\ y'
\end{bmatrix}
=
\begin{bmatrix}
cos \theta & -sin \theta 
\\ 
sin \theta & cos \theta
\end{bmatrix}
(x,y \in [-radius,radius])
$$



旋转后采用以方向的$8\ast 8$的窗口。代表为关键点邻域所在尺度空间的一个像素，求取每个像素的梯度幅值与梯度方向，箭头方向代表该像素的梯度方向，长度代表梯度幅值，然后利用高斯窗口对其进行加权运算。最后在每个$4\ast 4$的小块上绘制8个方向的梯度直方图，计算每个梯度方向的累加值，即可形成一个种子点，如右图所示。每个特征点由4个种子点组成，每个种子点有8个方向的向量信息。这样就能生成128维的SIFT特征向量。



![描述子生成](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\20151001110709020)

```python
def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint
    """
    logger.info('Generating descriptors...')
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # 通过差值进行平滑
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')
```



## 2. Clustering on the extracted data

这一步相对简单，需要构建一个Kmeans算法完成聚类过程。

K-Means算法的思想很简单，对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。

　　　　如果用数据表达式表示，假设簇划分为$(C_1,C_2,...C_k)$，则我们的目标是最小化平方误差E：
$$
E = \sum_{i=1}^{k}{\sum_{x\in C_i}{||x-u_i||_2^2}}
$$
其中$μ_i$是簇$C_i$的均值向量，有时也称为质心，表达式为：
$$
u_i = \frac{1}{|C_i|}\sum_{x\in C_i}{x}
$$

如果我们想直接求上式的最小值并不容易，这是一个NP难的问题，因此只能采用启发式的迭代方法。`K-Means`采用的启发式方式很简单，用下面一组图就可以形象的描述。

现在描述下传统的K-Means算法流程。　

　　　　输入是样本集$D={x_1,x_2,...x_m}$,聚类的簇树$k$,最大迭代次数$N$

　　　　输出是簇划分$C={C_1,C_2,...C_k}$

　　　　1) 从数据集$D$中随机选择$k$个样本作为初始的$k$个质心向量：$ {μ_1,μ_2,...,μ_k}$

　　　　2）对于$n=1,2,...,N$

　　　　　　a) 将簇划分$C$初始化为$Ct=∅,\ t=1,2...k$

　　　　　　b) 对于$i=1,2...m$,计算样本$x_i$和各个质心向量$μ_j(j=1,2,...k)$的距离：$d_{ij}=||xi−μj||^2$，将xixi标记最小的为$d_{ij}$所对应的类别$λ_i$。此时更新$ C_{λ_i}=C_{λ_i} \cup x_i$

　　　　　　c) 对于$j=1,2,...,k$,对$C_j$中所有的样本点重新计算新的质心$μ_j=\frac{1}{|C_j|}\sum_{x\in C_j}{x}$

　　　　　　e) 如果所有的k个质心向量都没有发生变化，则转到步骤3）

　　　　3） 输出簇划分$C={C_1,C_2,...C_k}$



```python
class KMeans:
    def __init__(self, n_clusters, device, tol=1e-4):
        self.n_clusters = n_clusters
        self.device = device
        self.tol = tol
        self._labels = None
        self._cluster_centers = None

    def _initial_state(self, data):
        n, c = data.shape
        dis = torch.zeros((n, self.n_clusters), device=self.device)
        initial_state = torch.zeros((self.n_clusters, c), device=self.device)
        idx = np.random.randint(0, n)
        initial_state[0, :] = data[idx]

        for k in range(1, self.n_clusters):
            for center_idx in range(self.n_clusters):
                dis[:, center_idx] = torch.sum((data - initial_state[center_idx, :]) ** 2, dim=1)
            min_dist, _ = torch.min(dis, dim=1)
            p = min_dist / torch.sum(min_dist)
            initial_state[k, :] = data[np.random.choice(np.arange(n), 1, p=p.to('cpu').numpy())]

        return initial_state

     def fit(self, data):
        data = data.to(self.device)
        cluster_centers = self._initial_state(data)
        dis = torch.zeros((len(data), self.n_clusters))

        while True:
            for i in range(self.n_clusters):
                dis[:, i] = torch.norm(data - cluster_centers[i], dim=1)
            labels = torch.argmin(dis, dim=1)
            cluster_centers_pre = cluster_centers.clone()
            for i in range(self.n_clusters):
                cluster_centers[i, :] = torch.mean(data[labels == i], dim=0)
            center_shift = torch.sum(torch.sqrt(torch.sum((cluster_centers - cluster_centers_pre) ** 2, dim=1)))
            if center_shift ** 2 < self.tol:
                break

        self._cluster_centers = cluster_centers
        self._labels = labels

    # 按照分簇结果对x进行分类
    def predict(self, x):
        x = x.to(self.device)
        dis = torch.zeros([x.shape[0], self.n_clusters]).to(self.device)

        for i in range(self.n_clusters):
            dis[:, i] = torch.sum((x-self._cluster_centers[i, :])**2, dim=1)

        pred = torch.argmin(dis, dim=1)
        return pred

```

### 2.1 轮廓系数 Silhouette Score

轮廓系数（Silhouette Coefficient）结合了聚类的凝聚度（Cohesion）和分离度（Separation），用于评估聚类的效果。该值处于-1~1之间，值越大，表示聚类效果越好。具体计算方法如下：

对于第$i$个元素$x_i$，计算$x_i$与其同一个簇内的所有其他元素距离的平均值，记作$a_i$，用于量化簇内的凝聚度。

选取$x_i$外的一个簇$b$，计算$x_i$与$b$中所有点的平均距离，遍历所有其他簇，找到最近的这个平均距离,记作$b_i$，用于量化簇之间分离度。

对于元素$x_i$，轮廓系数
$$
s_i = \frac{(b_i – a_i)}{max(a_i,b_i)}
$$
计算所有x的轮廓系数，求出平均值即为当前聚类的整体轮廓系数 从上面的公式，不难发现若$s_i$小于0，说明$x_i$与其簇内元素的平均距离小于最近的其他簇，表示聚类效果不好。如果$a_i$趋于0，或者$b_i$足够大，那么$s_i$趋近与1，说明聚类效果比较好。

```python
def silhouette_score(self, x, labels):
    x = x.to(self.device)
    labels = labels.to(self.device)
    length = len(x)
    indexes = torch.arange(length)
    sampled_indexes = np.random.choice(indexes, 10000)
    ones_vector = torch.ones(length).to(self.device)
    total = torch.Tensor([0]).to(self.device)
    for index, i in enumerate(x[sampled_indexes]):
        if index % 100 == 0:
            print(index)
        matched = labels == labels[index]
        dis = torch.sqrt(torch.sum((x-i)**2, dim=1))
        cnt = torch.sum(ones_vector[matched])
        a = torch.sum(dis[matched]) / cnt
        matched = labels != labels[index]
        cnt = torch.sum(ones_vector[matched])
        b = torch.sum(dis[matched]) / cnt
        total += (b-a) / (torch.max(a, b))
    return total / 10000
```



## 3. get visual word representation

对获取的`descroptors`进行聚类之后，在设定`K=70`的情况下可以，获得70个簇类的分类结果。由此每个`keypoints`进行分类，由此可以得到每个`image`的特征值在不同簇的分类结果。每一个`label`的分类结果可以代表`visual word representation`。但是由于特征在不同图片中都可能重复出现，有必要使用`tf-idf`来修正分类结果。



#### `tf-idf`

`tf-idf`全名为：**term frequency–inverse document frequency**。正如其名字所称， 其计算结结果由`tf`和`idf`相减二来：
$$
\begin{split}
tf(t,d)&=log(1 + freq(t,d))\\
idf(t,D)&=log(\frac{N}{count\ d \in D: t \in d })
\end{split}
$$
正如公式所描述的：`tf`给出了某个特征点在图像中的重要性，而`idf`计算出了某个特征点在整个图像库的普遍性的负值。当然`tf-idf`广泛应用于关键词抽取等文本应用中。

> 代码实现见下节

### 4. Calculate the inverted file index

这一步实现的需求不是很大，这里我采用了直接使用`image`自带的`filename`来获取文件。



```python
import torch
import math


# 计算tf-idf化的各个图像histogram, 同时也负责计算 test image的histogram, 以及计算选取the test image 和 train images 距离最小的图像
class ImageRetriever:
    def __init__(self, bag_of_visual_words):
        self.kmeans = bag_of_visual_words.kmeans
        self.images = bag_of_visual_words.images
        self.inverted_file_table = bag_of_visual_words.inverted_file_table
        self.generate_image_histogram = bag_of_visual_words.generate_image_histogram
        self._generate_tf_idf_weighted_histogram()

        self.total_tf_idf_weighted_histogram = self._generate_total_tf_idf_weighted_histogram()

    def _generate_tf_idf_weighted_histogram(self):
        for image in self.images:
            image.set_tf_idf_weighted_histogram(self._tf_idf(image))

    # 用 tf-idf 将histogram of image修正
    def _tf_idf(self, image):
        k = self.kmeans.n_clusters
        tf_idf_weighted_histogram = torch.zeros([1, k])
        visual_words_num = torch.sum(image.histogram)
        # 计算 tf-idf
        for i in range(k):
            tf = image.histogram[:, i] / visual_words_num
            idf = math.log(len(self.images) / (len(self.inverted_file_table[i]) + 1))
            tf_idf_weighted_histogram[:, i] = tf * idf
        return tf_idf_weighted_histogram

    # 将tf-idf histogram cat 起来
    def _generate_total_tf_idf_weighted_histogram(self):
        total_tf_idf_weighted_histogram = None
        for image in self.images:
            if total_tf_idf_weighted_histogram is None:
                total_tf_idf_weighted_histogram = image.tf_idf_weighted_histogram
            else:
                total_tf_idf_weighted_histogram = torch.cat(
                    (total_tf_idf_weighted_histogram, image.tf_idf_weighted_histogram), dim=0)
        return total_tf_idf_weighted_histogram



    def retrieve(self, image):
        labels = self.kmeans.predict(torch.Tensor(image.descriptors))
        self.generate_image_histogram(image, labels)
        image.set_tf_idf_weighted_histogram(self._tf_idf(image))
        m = self.total_tf_idf_weighted_histogram.shape[0]
        l = torch.zeros([m, 1])

        # 取出距离最小的值， 如果用矩阵并行计算更好一点？
        for i in range(m):
            l[i, :] = torch.sum((image.tf_idf_weighted_histogram-self.total_tf_idf_weighted_histogram[i])**2, dim=1)
        min_loss_image_index = torch.argmin(l, dim=0)
        return self.images[min_loss_image_index]
```

### 5. Evaluation

从上述讲述可知，`test dataset`是10张图像，依次判断其`label`的过程如下：

1. 计算`keypoints`和`destriptors`
2. 对`keypoints`和`destriptors`使用之前的KMeans类进行聚类，即进行分类
3. 进行`tf-idf`修正得到`visual word representation: p`
4. 采用`L2 norm`为指标评估`train set P`和`p`，并选出值最小的`label`作为结果

最后根据各个分结果计算`Recall`和`Precision`



```python

def test(train_path, test_path):
    test_acc, test_report, test_confusion = evaluate(train_path, test_path)
    print("Test Acc: " + str(test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(train_path, test_path):
    images = load_images(test_path)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    classes = ['bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall']
    classes2idx = {key: val for val, key in enumerate(classes)}
    kmeans = KMeans(n_clusters=70, device=device)
    image_retriever = ImageRetriever(BagOfVisualWords(images=load_images(train_path), kmeans=kmeans))

    for image in images:
        labels_all = np.append(labels_all, classes2idx[image.label])
        predict_all = np.append(predict_all, classes2idx[image_retriever.retrieve(image).label])

    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=classes, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, report, confusion

```



## 6. 实验结果

按照上面的算法和步骤对代码进行实现和调用，其实验结果如下：

![image-20201023220815192](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023220815192.png)

![image-20201023220801277](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023220801277.png)

可以看到，分类效果还是相当不错的，`90%`的成功率，只有一张图片出现了错误。具体图片如下：

+ 分类错误：

![image-20201023221039482](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023221039482.png)

+ 其他几个分类成功的图片：

  

![image-20201023221121450](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023221121450.png)

![image-20201023221153762](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023221153762.png)


![image-20201023221205199](D:\个人文件\重要文件\#2研究生在校\学习内容\计算机视觉导论\作业\experiment1\docs\image-20201023221205199.png)



实验过程中很痛苦，需要学习编写很多有些陌生的领域，尤其是`SIFT`。其实`SIFT`实现的部分不多。也是“虽败犹荣”，希望老师手下留情，还想拿个好点的奖学金。

## 参考资料

[数据源下载](http://www.robots.ox.ac.uk/~vgg/data/affine/)

[Bow词袋知识点](https://blog.csdn.net/u012328159/article/details/84719494)

[SIFT详解1](https://blog.csdn.net/zddblog/article/details/7521424?utm_medium=distribute.pc_relevant.none-task-blog-title-2)

[SIFT解析2](https://blog.csdn.net/jiaoyangwm/article/details/79986729)

[pysift解析](https://github.com/rmislam/PythonSIFT)

[pysift项目](https://github.com/rmislam/PythonSIFT) [作者解析系列](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b)

[Canopy聚类算法详解](https://blog.csdn.net/jameshadoop/article/details/27242039)

[SIFT中文博客解析](https://www.jianshu.com/p/95c4890c486b)

[轮廓`score`详解](http://studio.galaxystatistics.com/report/cluster_analysis/article4/)

[tf-idf wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#:~:text=In%20information%20retrieval%2C%20tf%E2%80%93idf,in%20a%20collection%20or%20corpus.)

opencv-python

[Unstanding the feature](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_meaning/py_features_meaning.html#features-meaning)

[What is Image Pyramids?](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html#pyramids)







