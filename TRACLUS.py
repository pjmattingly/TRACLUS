class TRACLUS:
    '''
    An implementation of the TRACLUS algorithm
    
    see:
    Lee, J. G., Han, J., & Whang, K. Y. (2007, June)
    '''

    def __init__(self):
        pass

    def run(self, trajectories, _epsilon=None, _minlns=None):
        '''
        A function to begin running the TRACLUS algorithm on a set of trajectories

        ----

        Given a set of trajectories (self._trajectories)
        create approximate, simpler, trajectories via _approximate_trajectory_partitioning()
        then, given these tarjectories, estimate optimal values for epsilon and minlns with _parameter_value_selection()
        then, given these values and the set of approximate trajectories, form clusters with _line_segment_clustering()
        finally, generate a set of representative trajectories for each cluster with _representative_trajectory_generation()
        '''

        assert len(trajectories) > 0

        if not _epsilon is None:
            assert epsilon >= 0

        if not _minlns is None:
            assert minlins >= 0

        #----
        #Pre-processing the input trajectories before handing off to the algorithm
        #----

        traj_tmp = self._pre_process(trajectories)

        #if there aren't any usable trajectories left after pre-processing
        #then panic out

        if len( traj_tmp ) == 0:
            err = ""
            err += "No usable trajectorise in input after pre-processing. " + "\n"
            raise RuntimeError(err)

        self._trajectories = traj_tmp

        #----
        #Begin algorithm
        #----

        #NOTE
        #we took out the step of the algorithm to approximate trajectories here
        #this is because, for this project, such approximations aren't necessary
        #  since we're using contrived trajectories
        #indeed an approximation might hide information needed for analysis

        #approximate_trajectories = list()
        #for t in self._trajectories:
        #    approximate_trajectories.append( self._approximate_trajectory_partitioning(t) )

        #NOTE, here we skip the approximation step
        approximate_trajectories = self._trajectories

        epsilon, minlns_range = self._parameter_value_selection(approximate_trajectories)

        #use the epsilon as is, but take the average of the minlns range and round to the nearest whole number
        #the motivation is find a usable value for the minimum number of line segments to include in various methods called later in the algorithm
        minlns = round( float( sum(minlns_range)/len(minlns_range) ) )

        #use passed in values if they are set
        if not _epsilon is None:
            epsilon = _epsilon

        if not _minlns is None:
            minlns = _minlns

        #convert the approximate trajectories into line segment object for the next stage of the algorithm
        line_segment_objects = list()
        for t in approximate_trajectories:
            lss = self._convert_trajectory_to_line_segments(t)
            line_segment_objects.extend( lss )

        #the clusters are returned in a dictionary that maps their cluser id to the set of line segments in that cluster
        #this also includes the set of line segments not part of any cluster labeled as: 'noise'
        d_clusters = self._line_segment_clustering(line_segment_objects, epsilon, minlns)

        #NOTE, there is no guidance on what value of gamma to use, so it is set to zero here
        d_representative_trajectories = dict()
        for k in d_clusters:
            cluster = d_clusters[k]

            _t = self._representative_trajectory_generation(cluster, minlns)

            d_representative_trajectories[k] = _t

        #----
        #End algorithm
        #----

        #on completion return a single result object
        class _result:
            def __init__(self):
                self.approximate_trajectories = None
                self.epsilon = None
                self.minlns = None
                self.clusters = None
                self.representative_trajectories = None

        ret = _result()

        #convert output to standard objects
        ret.approximate_trajectories = list()
        for t in approximate_trajectories:
            new_trajectory = list()
            for p in t:
                import numpy
                if type(p) is numpy.ndarray:
                    new_trajectory.append( p.tolist() )
                else:
                    new_trajectory.append( p )

            ret.approximate_trajectories.append( new_trajectory )

        ret.epsilon = float(epsilon)
        ret.minlns = float(minlns)

        ret.clusters = dict()
        for c in d_clusters:
            new_list = list()
            for ls in d_clusters[c]:
                raw_ls = None

                if hasattr(ls, 'get_line_segment'):
                    raw_ls = ls.get_line_segment()
                else:
                    raw_ls = ls

                if type(raw_ls) is numpy.ndarray:
                    new_list.append( raw_ls.tolist() )
                else:
                    new_list.append( raw_ls )

            ret.clusters[c] = new_list

        #ret.representative_trajectories = d_representative_trajectories

        ret.representative_trajectories = dict()
        #d_representative_trajectories
        for k in d_representative_trajectories:
            new_trajectory = list()
            for p in d_representative_trajectories[k]:
                import numpy
                if type(p) is numpy.ndarray:
                    new_trajectory.append( p.tolist() )
                else:
                    new_trajectory.append( p )

            ret.representative_trajectories = new_trajectory

        return ret

    def _pre_process(self, trajectories):
        '''
        this method services as a central call for any pre-processing that
        needs to be done to the input before handing them over
        to the TRACLUS algorithm
        '''

        ret = self._remove_zero_length_line_segments(trajectories)

        ret = self._remove_zero_length_trajectories(ret)

        return ret

    def _remove_zero_length_line_segments(self, trajectories):
        '''
        pre-processing

        this method removes zero length line segments from the trajectory
        as these line segments are zero length, they do not contain information relevant to TRACLUS
        also the behaviour of TRACLUS is undefined when working with zero length line segments
        '''

        ret = list()

        for traj in trajectories:
            new_trajectory = list()
            last_seen = None

            for coordinate in traj:
                if last_seen is None:
                    last_seen = coordinate
                    import numpy as np
                    new_trajectory.append( np.array(coordinate) )
                    continue

                import numpy as np
                coordinate = np.array( coordinate )
                
                if np.all(coordinate == last_seen):
                    continue

                new_trajectory.append(coordinate)
                last_seen = coordinate

            ret.append( new_trajectory )

        return ret

    def _remove_zero_length_trajectories(self, trajectories):
        '''
        given an iterable of trajectories
        remove those that only have a single point and thus are zero length
        '''

        ret = list()

        for t in trajectories:
            if len(t) == 1:
                continue
            ret.append(t)

        return ret

    #TODO, add preprocessing to transform trajectories into ordered sets of _line_segment objects

    '''         
    methods for vector manipulation:
        used in the distance methods
    '''
    def _vectorize(self, p1, p2):
        '''
        Returns a 2d vector given a pair of 2d points
        '''
        #only works for 2d coordinates
        assert len(p1) == 2
        assert len(p2) == 2

        import numpy as np
        vec = np.array( [(p2[0] - p1[0]), (p2[1] - p1[1])] )

        return vec
    
    def _vector_magnitude(self, v1):
        '''
        returns the magnitude of a vector
        '''
        
        assert len(v1) == 2
        
        import numpy as np
        v_tmp = np.array(v1)
        
        import numpy as np
        #return np.linalg.norm(v1)
        #return np.sqrt( np.vdot(v_tmp, v_tmp) )
        return np.sqrt( (v_tmp*v_tmp).sum() )
    
    def _scalar_projection(self, v1, v2):
        '''
        Returns the scalar projection of v1 onto v2

        source: https://stackoverflow.com/questions/55226011/using-python-to-calculate-vector-projection
        '''
        
        #zero vector case
        
        #in this case we're projecting onto or projecting a zero length vector
        #so the magnitude of the projection is zero
        if (self._vector_magnitude(v1) == 0) or (self._vector_magnitude(v2) == 0):
            return float(0)
        
        #otherwise find the scalar projection as normal
        import numpy as np
        return float(np.vdot(v1, v2) / np.linalg.norm(v2))
    
    def _vector_projection(self, v1, v2):
        '''
        A function which returns the vector projection of v1 onto v2

        source: https://stackoverflow.com/questions/55226011/using-python-to-calculate-vector-projection
        '''
        import numpy as np

        #coerce inputs into numpy arrays/vectors
        v1_tmp = np.array(v1)
        v2_tmp = np.array(v2)
        
        #zero length v2 case
        v2_mag = self._vector_magnitude(v2)
        if v2_mag == 0:
            #if this is the case, then the projection would also be zero
            return np.array([0, 0])
        
        #unit vector in the direction of v2
        v2_unit_vector = (v2_tmp/v2_mag)
        
        v1_scalar_projection = self._scalar_projection(v1, v2)
        
        _ret = (v1_scalar_projection * v2_unit_vector)
        
        #avoid -0
        #source: https://stackoverflow.com/questions/26782038/how-to-eliminate-the-extra-minus-sign-when-rounding-negative-numbers-towards-zer
        _ret += float(0)
        
        return _ret
    
    def _vector_rejection(self, v1, v2):
        '''
        This returns the vector rejection between v1 and v2
        '''

        import numpy as np

        #coerce inputs into numpy arrays/vectors
        v1_tmp = np.array(v1)
        v2_tmp = np.array(v2)
        
        #zero case
        if (self._vector_magnitude(v1_tmp) == 0) or (self._vector_magnitude(v2_tmp) == 0):
            #if either of the vectors is the zero vector, then the rejection is also the zero vector
            return np.array([0, 0])

        #get the vector projection first
        proj = self._vector_projection(v1_tmp, v2_tmp)
        
        #then the rejection
        return (v1_tmp - proj)

    def _vector_angle(self, v1, v2):
        '''
        Given a pair of vectors (v1 and v2)
        compute the angle (in radians) between them
        '''

        import numpy as np
        cos_theta = np.vdot(v1, v2) / (self._vector_magnitude(v1)*self._vector_magnitude(v2))
        
        #there are some rounding error when calculating cosine of theta
        #so round to 10 decimal points to avoid edge cases
        cos_theta = round(cos_theta, 10)
        
        #TODO, testing?
        if not (-1 <= cos_theta <= 1):
            err = ""
            err += "_vector_angle(), cosine theta outside of [-1, 1]" + "\n"
            err += "Cosine theta: "  + "\n"
            err += str(cos_theta) + "\n"
            err += "vectors: "+ "\n"
            err += str([v1, v2]) + "\n"
            raise RuntimeError(err)
        
        return np.arccos( cos_theta )

    '''
    distance methods:
        _perpendicular_distance(), _parallel_distance, and angle_distance()
        are components of _distance()
    '''
    
    def _perpendicular_distance(self, l1, l2):
        '''
        Given two line segments, get the perpendicular distance between them
        which is defined as:

        "Deﬁnition 1. The perpendicular distance between Li and
        Lj is deﬁned as Formula (1), which is the Lehmer mean 2 of
        order 2. Suppose the projection points of the points sj and
        ej onto Li are ps and pe, respectively. l⊥1 is the Euclidean
        distance between sj and ps; l⊥2 is that between ej and pe."

        Lee, J. G., Han, J., & Whang, K. Y. (2007, June)
        '''
        
        assert len(l1) == 2
        assert len(l2) == 2
        
        import numpy as np
        l1_tmp = np.array(l1)
        l2_tmp = np.array(l2)
        
        #The TRACLUS algorithm does not handle the case of colinear line segments in the case of the perpendicular distance
        #thus the special handling here of specifying such a perpendicular distance as 0
        #the approach is to use various approaches depending on the number of unique coordinates
        #an additional motivation for using this approach, is that finding zero length vector rejections
        #is rife with calculation errors

        points = list()
        points.extend( l1 )
        points.extend( l2 )

        d_tmp = dict()
        for p in points:
            d_tmp[tuple(p)] = None

        l_points = list(d_tmp.keys())

        if len(l_points) == 1:
            #all points the same case, perp distance is 0
            return float(0)

        if len(l_points) == 2:
            #two unique points, forms a single line, perp distance = 0
            return float(0)

        if len(l_points) == 3:
            #three unique points, the points of the line segments form a triangle
            
            #compute the area of the triangle to check if the points are co-linear
            #https://en.wikipedia.org/wiki/Triangle#Computing_the_area_of_a_triangle
            x, y = zip(*l_points)
            
            m = [x, y, [1, 1, 1]]
            matrix = np.array(m)
            
            import numpy as np
            area = 1/2*abs(np.linalg.det(matrix))

            #FIX, floating point errors, rounding to 10 decimal places to resolve
            area = round(area, 10)

            #if the area of the triangle is zero, then all the points lie on the same line
            if area == 0:
                return float(0)

        if len(l_points) == 4:
            #four inque points, the points of the line segments form a quadrilateral
            
            #compute the area of the quadrilateral to check for colinearlity
            #source: https://en.wikipedia.org/wiki/Quadrilateral
            
            #make vectors that cross between the line segments
            v1 = self._vectorize(l2[0], l1[1])
            v2 = self._vectorize(l1[0], l2[1])
            
            import numpy as np
            area = (1/2)*abs( np.cross(v1, v2) )

            #FIX, floating point errors, rounding to 10 decimal places to resolve
            area = round(area, 10)
            
            #if the area of the quadrilateral is zero, then all the points lie on the same line
            if area == 0:
                return float(0)
        
        #then, if the points are not co-linear, we can calculate their rejections to find the perpendicular distances
        #between the line segments

        #first, create a vector that goes from the initial point of l1 to the initial point of l2
        a = self._vectorize(l1_tmp[0], l2_tmp[0])
        
        #then a vector from the end point of l1 to l2
        b = self._vectorize(l2_tmp[1], l1_tmp[1])

        #then a vector of l2
        v2 = self._vectorize(l2_tmp[0], l2_tmp[1])

        #then find the magnitude of the vector rejection between a and l2
        a2_mag = self._vector_magnitude( self._vector_rejection(a, v2) )
        
        #similarly, find the magnitued of the vector rejection of the vector b onto l2
        b2_mag = self._vector_magnitude( self._vector_rejection(b, v2) )

        if (b2_mag + a2_mag) == 0:
            err = ""
            err += "Divide by zero"
            raise RuntimeError(err)
        
        #then return the lehmer mean of the two magnitudes
        return float((a2_mag**2 + b2_mag**2) / (a2_mag + b2_mag))
            
    def _parallel_distance(self, l1, l2):
        '''
        Given two line segments, get the parallel distance between them
        which is defined as:

        "Deﬁnition 2. The parallel distance between Li and Lj is
        deﬁned as Formula (2). Suppose the projection points of the
        points sj and ej onto Li are ps and pe, respectively. l∥1 is
        the minimum of the Euclidean distances of ps to si and ei.
        Likewise, l∥2 is the minimum of the Euclidean distances of
        pe to si and ei"
        Lee, J. G., Han, J., & Whang, K. Y. (2007, June)
        '''

        assert len(l1) == 2
        assert len(l2) == 2
        
        import numpy as np
        l1_tmp = np.array(l1)
        l2_tmp = np.array(l2)
        
        #the parallel distance does not handle the zero length line segment case
        #so special handling that such distances are zero

        points = list()
        points.extend( l1 )
        points.extend( l2 )

        d_tmp = dict()
        for p in points:
            d_tmp[tuple(p)] = None

        l_points = list(d_tmp.keys())

        if len(l_points) == 1:
            #all points the same case, perp distance is 0
            return float(0)
        
        sisj = self._vectorize(l1[0], l2[0])
        siei = self._vectorize(l1[0], l1[1])
        siej = self._vectorize(l1[0], l2[1])
        
        import numpy as np
        u1 = np.vdot(sisj, siei) / (self._vector_magnitude( siei ) ** 2)
        u2 = np.vdot(siej, siei) / (self._vector_magnitude( siei ) ** 2)
        
        #BUG
        #here we start running into precision errors
        #so, rounding to 10 decimal places
        #source: https://stackoverflow.com/questions/32952941/numpy-floating-point-rounding-errors
        #and: https://stackoverflow.com/questions/25181642/how-set-numpy-floating-point-accuracy
        u1 = round(u1, 10)
        u2 = round(u2, 10)
        
        si = l1[0]
        
        ps = si + u1*siei
        pe = si + u2*siei
        
        ei = l1[1]
        
        from scipy.spatial.distance import euclidean
        l_para_1 = euclidean(si, ps)
        l_para_2 = euclidean(pe, ei)
        
        return min([l_para_1, l_para_2])
    
    def _angle_distance(self, l1, l2):
        '''
        Given two line segments, get the angle distance between them
        which is defined as:

        "The angle distance between Li and Lj is de-
        ﬁned as Formula (3). Here, ∥Lj∥ is the length of Lj, and θ
        (0◦ ≤ θ ≤ 180◦) is the smaller intersecting angle between Li
        and Lj."
        '''
        
        assert len(l1) == 2
        assert len(l2) == 2
        
        points = list()
        points.extend( l1 )
        points.extend( l2 )

        #this distance function does not handle zero length vectors
        #so, special handling here of returning a zero length 
        d_tmp = dict()
        for p in points:
            d_tmp[tuple(p)] = None

        l_points = list(d_tmp.keys())

        if len(l_points) < 3:
            #this covers the case of a single point
            #or a single line segment
            return float(0)
        
        siei = self._vectorize(l1[0], l1[1])
        sjej = self._vectorize(l2[0], l2[1])

        #find the angle between siei and sjej
        theta_rad = self._vector_angle(siei, sjej)

        import numpy as np
        theta_deg = np.degrees(theta_rad)
        
        if 0 <= theta_deg < 90:
            return self._vector_magnitude(sjej)*np.sin(theta_rad)
        else:
            return self._vector_magnitude(sjej)
        
    def _distance(self, l1, l2, w=[1, 1, 1]):
        '''
        return the weighted distance between two line segments
        the weights are set to the recommended values of 1 for this project
        '''
        
        #treat the longer line segment as li (the first line segment) as per the paper
        li = None
        lj = None
        
        from scipy.spatial.distance import euclidean
        if euclidean(l1[0], l1[1]) > euclidean(l2[0], l2[1]):
            li = l1
            lj = l2
        else:
            li = l2
            lj = l1
        
        w_perp, w_para, w_ang = w
        ret = w_perp*self._perpendicular_distance(li, lj) + \
            w_para*self._parallel_distance(li, lj) + \
            w_ang*self._angle_distance(li, lj)
        return ret

    '''
    Methods used to approximate trajectories:
    '''
    
    def _L_H(self, cp1, cp2):
        '''
        given a pair of 2d characteristic points, calculate the log_2() of the length bewteen the points
        
        cp1, cp2, 2d points
        '''

        assert len(cp1) == 2
        assert len(cp2) == 2
        
        from scipy.spatial.distance import euclidean
        import numpy as np
        
        cp_len = euclidean(cp1, cp2)
        
        #checking for zero magnitude
        if cp_len == 0:
            #if we find  zero length line segment
            #then set this distance to zero
            return 0
        
        return np.log2( cp_len )
    
    def _L_DH(self, trajectory):
        '''
        Given a trajectory, calculate L(D|H)
        the sum of the perpendicular and angular distances from the line segment formed from the
        first to the last point in the trajectory
        to each line segment in the trajectory
        '''
        
        assert len(trajectory) > 1
        
        #in the case of a single line segment
        #the distance between the line segment and itself is zero
        if len(trajectory) == 2:
            return float(0)
        
        import numpy as np
        _first_point = np.array(trajectory[0])
        _last_point = np.array(trajectory[-1])

        _perps = list()
        _angs = list()
        _prev_point = None
        for _point in trajectory:
            if _prev_point is None:
                _prev_point = _point
                continue

            #one line segment is defined by (_first_point, _last_point), while the other is (_prev_point, _point)
            #find the perpendicular and angular distance between the two line segments
            import numpy as np
            _l1 = np.array([_first_point, _last_point])
            _l2 = np.array([_prev_point, _point])

            _perp = self._perpendicular_distance(_l1, _l2)
            
            _perps.append(_perp)
            
            _ang = self._angle_distance(_l1, _l2)
            
            _angs.append(_ang)

            _prev_point = _point
        
        #return the sum of the log2 of the perpendicular distances and the log2 of the sum of the angle distances
        ret_perp = None
        ret_angle = None
        
        #checking for the zero case
        #as, if the line segments overlap, then the perpendicular and angle distances will be zero
        #thus causing an issue for the log
        #so, just as with the perpendular distance case
        #as this function is a measure of distance between trajectories
        #trajectoies that have no space between them, have the L(D|H) distance equal to zero
        sum_perpendicular_distances = sum(_perps)
        if sum_perpendicular_distances == 0:
            ret_perp = float(0)
        else:
            import numpy as np
            ret_perp = np.log2(sum_perpendicular_distances)
        
        sum_angle_distances = sum(_angs)
        if sum_angle_distances == 0:
            ret_angle = float(0)
        else:
            import numpy as np
            ret_angle = np.log2(sum_angle_distances)
        
        return ret_perp + ret_angle
    
    def _MDL_par(self, critical_points, trajectory):
        '''
        calculates the cost of partitioning a trajectory at a given set of critical points
        '''
        assert len(critical_points) == 2
        
        _cp1 = critical_points[0]
        _cp2 = critical_points[1]
        
        assert len(_cp1) == len(_cp2) == 2
        
        return self._L_H(_cp1, _cp2) + self._L_DH(trajectory)
    
    def _MDL_nopar(self, critical_points):
        '''
        calculates the cost of NOT partitioning a trajectory at a given set of critical points
        '''
        assert len(critical_points) == 2
        
        _cp1 = critical_points[0]
        _cp2 = critical_points[1]
        
        assert len(_cp1) == len(_cp2) == 2
        
        return self._L_H(_cp1, _cp2)
    
    def _approximate_trajectory_partitioning(self, trajectory):
        '''
        an algoithm that finds a set of critical points along the trajectory
        these points form a trajectory that approximates the input trajectory
        '''
        
        _characteristic_points = list()
        
        _previous_characteristic_point_index = None
        for i in range(len(trajectory)):
            #initially, add the first point in the trajectory as a characteristic point
            if _previous_characteristic_point_index is None:
                _previous_characteristic_point_index = i
                _characteristic_points.append( trajectory[i] )
                
                continue
                
            _current_point = trajectory[i]
            _start_point = trajectory[_previous_characteristic_point_index]
            _current_trajectory_piece = trajectory[_previous_characteristic_point_index:i+1]
            
            #get the cost of partitioning the trajectory with a line segment between the last
            #critical point and the current point
            _partition_cost = self._MDL_par([_start_point, _current_point], _current_trajectory_piece)
            
            #then the cost of avoiding partitioning at this point
            _no_partition_cost = self._MDL_nopar([_start_point, _current_point])

            #then adding a small amount to the cost of not partitioning to improving clustering quality
            #as per recommendation in the paper (see page 599)
            _no_partition_cost += (_no_partition_cost*.25)
            
            #then if the cost of partition outweighs the cost of not partitioning
            #then get the previous point as the next characteristic point
            #and store this pair of points to indicate a line segment that can approximate the trajectory between those points
            if _partition_cost > _no_partition_cost:
                _characteristic_points.append( trajectory[i-1] )
                _previous_characteristic_point_index = i-1
        
        #add the last point in the original trajectory as a characteristic point
        _characteristic_points.append( trajectory[-1] )
        
        return _characteristic_points

    '''
    Methods used to cluster line segments beloning to trajectories:
    '''
    
    class _line_segment:
        '''
        This class stores information about a line segment:

        its values
        what trajectory it's associated with
        references to "nearby" line segments
        its clusterid

        ----

        constants:

        self._ls, holds a reference for the line segment for this object
        sef._t, holds a reference for the trajectory associated with this line segment

        A line segment can be in one of the following states:
            classified
            unclassified
            noise
        where 'classified' indicates associated with a cluster
        'unclassified' indicates it has not been associated with a cluster
        and 'noise' indicates that the line segment has been classified as noise

        self._d_distances, a dict matching the key of a line segment and an epsilon to a distance value

        self._d_e_neighborhood, references to other line segments that are epsilon distance away
        '''

        def __init__(self, line_segment, trajectory):
            #check that the line segment is in the proper form:
            #[ [a, b], [c, d] ]
            assert len(line_segment) == 2

            import numpy as np
            assert np.array(line_segment).shape == (2, 2)

            import numpy as np
            self._ls = np.array(line_segment)
            self._t = np.array(trajectory)

            #keep track of the distance to line segments that are tested via is_nearby()
            self._d_distances = dict()

            #labels the line segment as either being: part of a cluster, noise, or unspecified
            self.label = None

            #keep track of the epsilon neighborhood around this line segment
            #for different values of epsilon
            #maps: epsilon -> [ls1, ls2, ...] (nearby line segments)
            self._d_e_neighborhood = dict()

        def is_nearby(self, line_segment, epsilon):
            return self._is_nearby(line_segment, epsilon)

        def _is_nearby(self, line_segment, epsilon):
            '''
            Given a line segment and a value epsilon
            determine if the line segment symbolized by this object
            is epsilon distance from it
            '''

            assert epsilon >= 0

            ls = None

            #if line_segment is an object, then get the values of its line segment
            if hasattr(line_segment, "get_line_segment"):
                ls = line_segment.get_line_segment()
            else:
                ls = line_segment

            assert len(ls) == 2

            #make a key out of the line segment coordinates and the epislon
            _t_key = tuple( [ls[0][0], ls[0][1], ls[1][0], ls[1][1], epsilon] )

            #store the distance to this line segment if it hasn't been seen already
            if not _t_key in self._d_distances:
                _tmp_t = TRACLUS()
                self._d_distances[_t_key] = _tmp_t._distance(self._ls, ls)

            if self._d_distances[_t_key] <= epsilon:
                return True
            return False

        def get_epsilon_neighborhood(self, D, epsilon):
            '''
            Given a set of line segments (D)
            and a distance (epsilon)
            find those line segment that are at most epsilon distance from this line segment
            '''

            #ensure epsilon is hashable
            epsilon = float(epsilon)

            if not epsilon in self._d_e_neighborhood:
                self._d_e_neighborhood[epsilon] = list()

            e_neighborhood = list()
            if len(self._d_e_neighborhood[epsilon]) == 0:
                for _ls in D:
                    if self._is_nearby(_ls, epsilon):
                        e_neighborhood.append( _ls )
                self._d_e_neighborhood[epsilon] = e_neighborhood
            return self._d_e_neighborhood[epsilon]

        def get_line_segment(self):
            return self._ls

        def get_trajectory(self):
            return self._t

        def belongs_to_trajectory(self, trajectory):
            '''
            this determines if the trajectory passed is the one that this line segment is associated with
            '''

            import numpy as np
            _tmp = np.array(trajectory)
            return np.all(self._t == _tmp)

        def equal(self, line_segment):
            ls = None
            #if line_segment is an object, then get the values of its line segment
            if hasattr(line_segment, "get_line_segment"):
                ls = line_segment.get_line_segment()
            else:
                ls = line_segment

            import numpy as np
            return np.all(np.array(ls) == self._ls)

        def __repr__(self):
            return str(self._ls)
            
    def _line_segment_clustering(self, D, epsilon, minlns):
        '''
        given a set of line segments
        find clusters that:
            have cardinality at least "minlns"
            and
            whose line segments are "epsilon" distance apart
            or are density connected
        --

        D, an iterable of line segment objects
        epsilon, a float, the minimum distance between line segments to: count them as in the same cluster or close enough to a
            cluster to be included
        minlns, an int, the minimum number of line segments that should belong to a cluster
        '''

        assert epsilon >= 0
        assert minlns >= 0

        #first, add line segments to clusters or classify as noise
        cluster_counter = 0
        for ls1 in D:
            #check if line segment had been previously labeled
            if not ls1.label is None:
                continue

            #find this line segment's epsilon neighborhood
            _l_neighbors = ls1.get_epsilon_neighborhood( D, epsilon )

            #if the neighborhood is too small, then this line segment is probably noise
            if len(_l_neighbors) < minlns:
                ls1.label = "noise"
                continue

            cluster_counter += 1

            #we've found a core point, so label it with its cluster id
            ls1.label = cluster_counter

            #next, we need to check the neighbors for other nearby line segments
            S = ls1.get_epsilon_neighborhood(D, epsilon)

            while (True):
                #if we're out of line segments to process then get out of the loop
                if len(S) == 0:
                    break
                
                _ls = S.pop(0)
            
                #if it's the core line segment, then skip
                if _ls.equal(ls1):
                    continue

                #change line segments initially flagged as "noise" to a border points on this cluster
                if _ls.label == "noise":
                    _ls.label = cluster_counter
                    continue

                #skip previously labeled line segments
                if not _ls.label is None:
                    continue

                _ls.label = cluster_counter

                _neighbors = _ls.get_epsilon_neighborhood( D, epsilon )

                #then if this line segment is also a core point, then add its neighbors for checking
                if len(_neighbors) >= minlns:
                    S.extend( _neighbors )

        #after we've found the initial clusters, add them to a dict for processing and return
        d_ret = dict()

        for ls2 in D:
            if not ls2.label in d_ret:
                d_ret[ls2.label] = list()
            d_ret[ls2.label].append( ls2 )

        #then check for trajectory cardinality of each of the clusters
        for label in d_ret.copy():
            #skip the 'noise' cluster
            #since we always want to return that to the caller
            if label == "noise":
                continue

            _d_seen_trajectories = dict()
            for ls3 in d_ret[label]:
                _trajectory = ls3.get_trajectory()

                #flatten and convert to a tuple for comparison
                import numpy as np
                _d_seen_trajectories[ tuple( np.array( _trajectory ).flatten() ) ] = None

            #then drop the cluster if its line segments are associated with too few unique trajectories
            if len(_d_seen_trajectories.keys()) < minlns:
                del d_ret[label]

        return d_ret

    def _convert_trajectory_to_line_segments(self, trajectory):
        '''
        Given a trajectory, an iterable of 2d points
        return a list of line_segment objects that define the trajectory
        '''
        assert len(trajectory) > 0

        prev = None
        ret = list()
        for p in trajectory:
            if prev is None:
                prev = p
                continue
            new_ls = self._line_segment( [prev, p], trajectory )

            ret.append( new_ls )

            prev = p
        return ret

    '''
    This method creates representative trajectories, given the line segments in a cluster:
    '''

    def _representative_trajectory_generation(self, cluster, minlns, gamma=0):
        '''
        Given a cluster of line segment objects (cluster)

        generate a representative trajectory that captures much of the information about the cluster

        minlns, a value that determines how many line segments are needed to meaningful contribute to the representative trajectory
            e.g. a few line segments should contribute less to the shape of the representative trajectory than more line segments

        gamma, a smoothing parameter
        '''

        #DEBUG
        #print("_representative_trajectory_generation()")
        #print("cluster:")
        #print(cluster)
        #print()

        #convert from line segment objects to raw values
        _cluster = list()
        for ls in cluster:
            if hasattr(ls, "get_line_segment"):
                _cluster.append( ls.get_line_segment() )
                continue
            _cluster.append( ls )

        assert len(cluster) > 0
        assert minlns >= 0
        assert gamma >= 0

        #convert the line segments to vectors
        vectors = list()
        for ls in _cluster:
            vectors.append( self._vectorize(ls[0], ls[1]) )

        #Compute the average direction vector
        import numpy as np
        v_avg_direction_vector = np.sum(vectors, axis=0) / len(vectors)

        #Rotate the axes so that the X axis is parallel to v_avg_direction_vector

        #compute the angle phi
        #or the angle between the average direction vector and the x-axis
        #used when rotating the coordinate system in the next step
        phi_rad = self._vector_angle(v_avg_direction_vector, [1, 0])

        def rotate_coordinates(x, y, phi):
            '''
            A function to rotate coordinates by an angle (rad) phi
            '''

            #make a column vector of the coordinates
            import numpy as np
            v1 = np.array([x, y])
            v1.shape = (2, 1)

            rotation_matrix = [ [np.cos(phi), np.sin(phi)],
                                [-np.sin(phi), np.cos(phi)] ]

            #the matrix multiplication
            import numpy as np
            ret = np.matmul(rotation_matrix, v1)

            #then, massaging the output to contain less lists
            ret2 = list()
            for v in ret:
                ret2.append( v[0] )

            return [ ret2[0], ret2[1] ]

        def undo_rotate_coordinates(x_prime, y_prime, phi):
            '''
            A function to undo the rotation caused by rotate_coordinates()
            '''

            import numpy as np
            v1 = np.array([x_prime, y_prime])
            v1.shape = (2, 1)

            undo_rotation_matrix = [ [np.cos(phi), -np.sin(phi)],
                                    [np.sin(phi), np.cos(phi)] ]

            import numpy as np
            ret = np.matmul(undo_rotation_matrix, v1)

            #then, massaging the output to contain less lists
            ret2 = list()
            for v in ret:
                ret2.append( v[0] )

            return [ ret2[0], ret2[1] ]

        #gather the set of points that make up the line segments in the cluster
        #and rotate them by phi
        P = list()
        for ls in _cluster:
            #get the points from the line segment
            p1, p2 = ls

            #rotate the points
            p1_prime = rotate_coordinates(p1[0], p1[1], phi_rad)
            p2_prime = rotate_coordinates(p2[0], p2[1], phi_rad)

            #store them for later sorting
            P.append(p1_prime)
            P.append(p2_prime)

        #sort the rotated points by their x_prime value
        def key_sort_by_x_prime(point):
            return point[0]

        P_sorted = sorted(P, key=key_sort_by_x_prime)

        #line 5 of, Algorithm Representative Trajectory Generation
        #loop through the sorted points, checking them for x-values that lie along the same vertical sweep / line
        #the idea is to check each point and see if a vertical line through that point would intersect more than one of the line segments (minlns number of them)

        def rotate_line_segment(line_segment, phi):
            ret = list()
            for point in line_segment:
                ls_rotated = rotate_coordinates(point[0], point[1], phi)
                ret.append(ls_rotated)
            return ret

        d_points_to_num_intersections = dict()
        previous_point = None
        rtr = list()
        for p1 in P_sorted:
            #get the x` portion of this point
            _x = p1[0]

            nump = 0
            t_p1 = None
            for ls in _cluster:
                ls_rotated = rotate_line_segment(ls, phi_rad)

                t_p1 = tuple(p1)

                if not t_p1 in d_points_to_num_intersections:
                    d_points_to_num_intersections[t_p1] = list()

                #then if a sweep line intersects this line segment, keep track of the intersection
                if ls_rotated[0][0] <= _x <= ls_rotated[1][0]:
                    d_points_to_num_intersections[t_p1].append(ls_rotated)

            #line 6 of, Algorithm Representative Trajectory Generation
            #find the number of intersections
            nump = len(d_points_to_num_intersections[t_p1])

            def compute_average_coordinate(x, line_segments):
                ys = list()
                for ls in line_segments:
                    p1, p2 = ls

                    ys.append( p1[1] )
                    ys.append( p2[1] )

                avg_y = float(sum(ys) / len(ys))
                return [x, avg_y]

            #line 7 of, Algorithm Representative Trajectory Generation
            if nump >= minlns:
                #BUG
                #the algorithm in the paper does not define what to do with the first point
                #so the change is to add it to the representative trajectory if its sweep intersects enough line segments
                if previous_point is None:
                    rtr.append( undo_rotate_coordinates(p1[0], p1[1], phi_rad) )
                    previous_point = p1
                    continue

                #line 8 of, Algorithm Representative Trajectory Generation
                diff = abs( _x - previous_point[0] )

                #line 9 of, Algorithm Representative Trajectory Generation
                #smoothing the representative trajectory
                if diff >= gamma:
                    #line 10 of, Algorithm Representative Trajectory Generation
                    #find the average coordinate of the line segments that _x intersections with
                    average_coordinate = compute_average_coordinate(_x, d_points_to_num_intersections[t_p1])

                    #line 11 of, Algorithm Representative Trajectory Generation
                    average_coordinate = undo_rotate_coordinates(average_coordinate[0], average_coordinate[1], phi_rad)

                    #line 12 of, Algorithm Representative Trajectory Generation
                    rtr.append( average_coordinate )

            previous_point = p1

        #then returning the representative trajectory for this cluster
        return rtr

    '''
    These methods estimate the parameters epsilon and minlns:
    '''

    def _parameter_value_selection(self, trajectories, _maxiter=100):
        '''
        given a set of trajectories (trajectories)
           it is assumed that the raw trajectories have been processed through _approximate_trajectory_partitioning() before calling this
        break them up into line segments, and wrap them in line_segment objects
        then use simulated annealing to find an optimal value for epsilon, and thus minlns
           note, dual annealing is used here as it should be superior to traditional simulated annealing for this task
        then the optimal epsilon and minlns are returned

        maxiter, the maximum number of iterations for the simulated annealing process
        '''
        
        assert len(trajectories) > 0

        lines = list()
        for t in trajectories:
            _lss = self._convert_trajectory_to_line_segments(t)

            lines.extend(_lss)

        #first, finding the optimal epsilon
        #with these functions and via simulated annealing
        def H(line_segments, epsilon):
            '''
            line_segments, the set of line segments 
            epsilon, the epsilon distance used to compute the epsilon neighborhood
            '''

            def p(x, line_segments, epsilon):
                '''
                x, a single line segment
                    to find the epsilon neighborhood around
                line_segments, the set of line segments where neighbors are found
                epsilon, the epsilon distance used to compute the epsilon neighborhood
                '''

                Ne_xi = len( x.get_epsilon_neighborhood(line_segments, epsilon) )

                Ne_xj_sum = 0
                for ls in line_segments:
                    Ne_xj = len( ls.get_epsilon_neighborhood(line_segments, epsilon) )

                    Ne_xj_sum += Ne_xj

                return float(Ne_xi/Ne_xj_sum)

            ret = 0
            for ls in line_segments:
                import numpy as np
                ret += -p(ls, line_segments, epsilon)*np.log2(p(ls, line_segments, epsilon))

            return ret

        def func(x):
            '''
            a wrapper function to use with scipy.optimize.dual_annealing
            '''

            return H(lines, epsilon=x)

        #find the largest and smallest values for the distance between line segments, to use with simulated annealing
        ds = list()
        prev = None
        for l in lines:
            if prev is None:
                prev = l
                continue

            ds.append( self._distance( prev.get_line_segment(), l.get_line_segment() ) )

            prev = l

        _max = max(ds)
        _min = min(ds)

        #then find the optimal epsilon with simulated annealing
        optimal_epsilon = None

        #approach = 'scipy'
        approach = 'pygmo'

        if approach == 'scipy':
            from scipy.optimize import dual_annealing
            result = dual_annealing(func, [[_min, _max]], maxiter=_maxiter)
            optimal_epsilon = result.x[0]

        if approach == 'pygmo':
            import pygmo as pg
            algo = pg.algorithm( pg.simulated_annealing() )

            class epsilon_problem:
                def fitness(self, x):
                    return [func(x)]

                def get_bounds(self):
                    return ([_min], [_max])

            prob = pg.problem( epsilon_problem() )

            pop = pg.population(prob, 1300)

            optimal_epsilon = pop.champion_x[0]

        #then with the optimal epsilon in hand, we can calculate the average size of the epsilon neighborhoods
        Nes = list()
        for l in lines:
            Nes.append( len( l.get_epsilon_neighborhood(lines, optimal_epsilon) ) )
        
        avg_Ne = float(sum(Nes) / len(Nes))

        #given this average, a range for minlns can be formulated
        minlns_range = [avg_Ne+1, avg_Ne+3]

        return [optimal_epsilon, minlns_range]

#end TRACLUS