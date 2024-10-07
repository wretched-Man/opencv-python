def constant_creator():
    """
    Creates the constants used for the smile detector algorithm.

    This method does just that, in getting the necessary thresholds.    
    """
    # This is the benchmark image set to get the best values for smile
    import glob

    image_set = glob.glob("imageset/*.jpg")

    # hold the summary values
    points0 = [] # 61 - 67
    points1 = [] # 62 - 66
    points2 = [] # 63 - 65

    for read_image in image_set:
        image = cv2.imread(read_image)

        # Get predictions
        faces = detectfaces(net, image)
        number_of_faces = faces.shape[0]

        if number_of_faces > 0:
            retval, face_landmarks = facemark.fit(image, faces)

            if retval is True:
                for landmark in face_landmarks:
                    # each landmark is for a single face, hence get the corresponding
                    # points, find their distances and store them as averages
                    landmark = np.squeeze(landmark)
                    # first calculate (60 - 64), the denominator of the ratio
                    denom = calculate_distance(landmark[60], landmark[64])
                    # 61 - 67
                    points0.append(calculate_distance(landmark[61], landmark[67]) / denom)
                    # 62 - 66
                    points1.append(calculate_distance(landmark[62], landmark[66]) / denom)
                    # 63 - 65
                    points2.append(calculate_distance(landmark[63], landmark[65]) / denom)
        else:
            print(f"Image {read_image} has no identifiable face.")

    points = [points0, points1, points2]

    # kmeans to sort outliers
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    flags = cv2.KMEANS_PP_CENTERS

    for point in points:
        point = np.squeeze(np.array(point))
        #clustering
        ret, label, center = cv2.kmeans(point,K,None,criteria,10,flags)
        label = label.ravel()

        small = point[np.where(label == 0)[0]]
        big = point[np.where(label == 1)[0]]
        
        if len(small) > len(big):
            main_points = small[:]
        else:
            main_points = big[:]
    
    return min(main_points), max(main_points), sum(main_points)/len(main_points)
