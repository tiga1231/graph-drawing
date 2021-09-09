# import math
import networkx as nx
import math

label_factor_magnifier = 72


def getAngleLineSegDegree(x1,y1,x2,y2,p_int_x,p_int_y):
    '''
        Computes the angular resolution of two intersecting edges (in degree).
        x1,y1 belong to one edge
        x2, y2 belong to the other edge
        p_int_x, p_int_y are the intersection point
    '''

    dc1x = x1-p_int_x
    dc2x = x2-p_int_y
    dc1y = y1-y3
    dc2y = y2-y3

    norm1 = math.sqrt(math.pow(dc1x,2) + math.pow(dc1y,2))
    norm2 = math.sqrt(math.pow(dc2x,2) + math.pow(dc2y,2))

    if norm1==0 or norm2==0:
        return -1

    theta = math.acos((dc1x*dc2x + dc1y*dc2y)/(norm1*norm2))

    if theta > math.pi/2.0:
        theta = math.pi - theta

    return to_deg(theta)

# Given three colinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(px, py, qx, qy, rx, ry):
    if (qx <= max(px, rx) and qx >= min(px, rx) and qy <= max(py, ry) and qy >= min(py, ry)):
        return True
    return False

def strictlyOnSegment(px, py, qx, qy, rx, ry):
    if (qx < max(px, rx) and qx > min(px, rx) and qy < max(py, ry) and qy > min(py, ry)):
        intersection_point = (qx, qy)
        return (True, intersection_point)
    return (False, None)

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are colinear
# 1 --> Clockwise
# 2 --> Counterclockwise

def orientation(px, py, qx, qy, rx, ry):
    # See http://www.geeksforgeeks.org/orientation-3-ordered-points/
    # for details of below formula.
    val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

    if (val == 0):return 0

    # clock or counterclock wise
    if (val > 0):
        return 1

    return 2

def yInt(x1, y1, x2, y2):
    if (y1 == y2):return y1
    return y1 - slope(x1, y1, x2, y2) * x1

def slope(x1, y1, x2, y2):
    #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2))
    if (x1 == x2):return False
    return (y1 - y2) / (x1 - x2)

# The main function that returns true if line segment 'p1q1'
# and 'p2q2' intersect.
def doSegmentsIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    # Find the four orientations needed for general and
    # special cases
    o1 = orientation(p1x, p1y, q1x, q1y, p2x, p2y)
    o2 = orientation(p1x, p1y, q1x, q1y, q2x, q2y)
    o3 = orientation(p2x, p2y, q2x, q2y, p1x, p1y)
    o4 = orientation(p2x, p2y, q2x, q2y, q1x, q1y)

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and onSegment(p1x, p1y, p2x, p2y, q1x, q1y)):
        intersection_point = (p2x, p2y)
        return (True, intersection_point, 'O')

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and onSegment(p1x, p1y, q2x, q2y, q1x, q1y)):
        intersection_point = (q2x, q2y)
        return (True, intersection_point, 'O')

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and onSegment(p2x, p2y, p1x, p1y, q2x, q2y)):
        intersection_point =  (p1x, p1y)
        return (True, intersection_point, 'O')

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and onSegment(p2x, p2y, q1x, q1y, q2x, q2y)):
        intersection_point = (q1x, q1y)
        return (True, intersection_point, 'O')

    #if(o1==0 or o2==0 or o3==0 or o4==0):return False
    # General case
    if (o1 != o2 and o3 != o4):
        intersection_point = compute_intersection_point(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)
        return (True, intersection_point, 'I')

    return (False, None, '') # Doesn't fall in any of the above cases

def isSameCoord(x1, y1, x2, y2):
    if x1==x2 and y1==y2:
        return True
    return False

# do p is an end point of edge (u,v)
def isEndPoint(ux, uy, vx, vy, px, py):
 if isSameCoord(ux, uy, px, py) or isSameCoord(vx, vy, px, py):
  return True
 return False

# is (p1,q1) is adjacent to (p2,q2)?
def areEdgesAdjacent(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    if isEndPoint(p1x, p1y, q1x, q1y, p2x, p2y):
        return True
    elif isEndPoint(p1x, p1y, q1x, q1y, q2x, q2y):
        return True
    return False

def isColinear(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    x1 = p1x-q1x
    y1 = p1y-q1y
    x2 = p2x-q2x
    y2 = p2y-q2y
    cross_prod_value = x1*y2 - x2*y1
    if cross_prod_value==0:
        return True
    return False

# here p1q1 is one segment, and p2q2 is another
# this function checks first whether there is a shared vertex
# then it checks whether they are colinear
# finally it checks the segment intersection
def doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    if areEdgesAdjacent(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
        if isColinear(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
            (intersection_exists, intersection_point) =  strictlyOnSegment(p1x, p1y, p2x, p2y, q1x, q1y)
            if intersection_exists:
                return (intersection_exists, intersection_point, 'C')
            (intersection_exists, intersection_point) =  strictlyOnSegment(p1x, p1y, q2x, q2y, q1x, q1y)
            if intersection_exists:
                return (intersection_exists, intersection_point, 'C')
            (intersection_exists, intersection_point) =  strictlyOnSegment(p2x, p2y, p1x, p1y, q2x, q2y)
            if intersection_exists:
                return (intersection_exists, intersection_point, 'C')
            (intersection_exists, intersection_point) =  strictlyOnSegment(p2x, p2y, q1x, q1y, q2x, q2y)
            if intersection_exists:
                return (intersection_exists, intersection_point, 'C')
            else:
                return (False, None, '')
        else: #//collinear
            return (False, None, '') #//collinear
    return doSegmentsIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)

def getIntersection(x11, y11, x12, y12, x21, y21, x22, y22):
    slope1 = 0
    slope2 = 0
    yint1 = 0
    yint2 = 0
    intx = 0
    inty = 0


    if (x11 == x21 and y11 == y21):return [x11, y11]
    if (x12 == x22 and y12 == y22):return [x12, y22]
    # Check 1st point of edge 1 with 2nd point of edge 2 and viceversa

    slope1 = slope(x11, y11, x12, y12)
    slope2 = slope(x21, y21, x22, y22)
    #print('slope1:'+str(slope1))
    #print('slope2:'+str(slope2))
    if (slope1 == slope2):return False

    yint1 = yInt(x11, y11, x12, y12)
    yint2 = yInt(x21, y21, x22, y22)
    #print('yint1:'+str(yint1))
    #print('yint2:'+str(yint2))
    if (yint1 == yint2):
        if (yint1 == False):return False
        else:return [0, yint1]

    if(x11 == x12):return [x11, slope2*x11+yint2]
    if(x21 == x22):return [x21, slope1*x21+yint1]
    if(y11 == y12):return [(y11-yint2)/slope2,y11]
    if(y21 == y22):return [(y21-yint1)/slope1,y21]

    if (slope1 == False):return [y21, slope2 * y21 + yint2]
    if (slope2 == False):return [y11, slope1 * y11 + yint1]
    intx = (yint1 - yint2)/ (slope2-slope1)
    return [intx, slope1 * intx + yint1]

def to_deg(rad):
    return rad*180/math.pi

# x1,y1 is the 1st pt, x2,y2 is the 2nd pt, x3,y3 is the intersection pt
def getAngleLineSegDegree(x1,y1,x2,y2,x3,y3):

    angle = float('-inf')
    try:
        #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2)+',x3:'+str(x3)+',y3:'+str(y3))
        # Uses dot product
        dc1x = x1-x3
        dc2x = x2-x3
        dc1y = y1-y3
        dc2y = y2-y3
        norm1 = math.sqrt(math.pow(dc1x,2) + math.pow(dc1y,2))
        norm2 = math.sqrt(math.pow(dc2x,2) + math.pow(dc2y,2))
        if norm1==0 or norm2==0:
            return -1
        angle = math.acos((dc1x*dc2x + dc1y*dc2y)/(norm1*norm2))
        # if angle > math.pi/2.0:
        #  angle = math.pi - angle
        #print('angle:'+str(angle))
        #return angle
        return to_deg(angle)
    except Exception as e:
        angle = float('-inf')
        return angle

# x1,y1 is the 1st pt, x2,y2 is the 2nd pt, x3,y3 is the intersection pt
def getAngleLineSeg(x1,y1,x2,y2,x3,y3):
    #print('x1:'+str(x1)+',y1:'+str(y1)+',x2:'+str(x2)+',y2:'+str(y2)+',x3:'+str(x3)+',y3:'+str(y3))
    # Uses dot product
    dc1x = x1-x3
    dc2x = x2-x3
    dc1y = y1-y3
    dc2y = y2-y3
    norm1 = math.sqrt(math.pow(dc1x,2) + math.pow(dc1y,2))
    norm2 = math.sqrt(math.pow(dc2x,2) + math.pow(dc2y,2))
    if norm1==0 or norm2==0:
        return -1
    angle = math.acos((dc1x*dc2x + dc1y*dc2y)/(norm1*norm2))
    # if angle > math.pi/2.0:
    #  angle = math.pi - angle
    #print('angle:'+str(angle))
    return angle


def compute_intersection_point(x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4):
    '''
    Computes the intersection point
    Note that the intersection point is for the infinitely long lines defined by the points,
    and can produce an intersection point beyond the lengths of the line segments
    but here we know that there is an intersection, and it is unique.
    '''

    den_intersection = (x_1 - x_2)*(y_3 - y_4) - (y_1 - y_2)*(x_3 - x_4)

    num_intersection_x = (x_1*y_2-y_1*x_2)*(x_3-x_4)-(x_1-x_2)*(x_3*y_4-y_3*x_4)
    num_intersection_y = (x_1*y_2-y_1*x_2)*(y_3-y_4)-(y_1-y_2)*(x_3*y_4-y_3*x_4)

    if(den_intersection == 0):
        return None

    intersection_x = num_intersection_x/den_intersection
    intersection_y = num_intersection_y/den_intersection

    intersection_point = (intersection_x, intersection_y)

    return intersection_point


def get_rectangle(p1x, p1y, w_1, h_1):
    '''
    Return the corner coordinates of the rectangle centered in
    p1x p1y and with width w_1 and h_1
    '''

    ax, ay = p1x-(w_1/2), p1y-(h_1/2)
    bx, by = p1x+(w_1/2), p1y+(h_1/2)
    cx, cy = p1x+(w_1/2), p1y-(h_1/2)
    dx, dy = p1x-(w_1/2), p1y-(h_1/2)


    return (ax, ay, bx, by, cx, cy, dx, dy, p1x, p1y)




def remove_label_edge_crossings(G, crossings_edges):
    '''
    Removes from the given list the crossings between edges and labels
    Instead of starting from the vertex position, the edges start from the boundary of the
    labels.
    '''
    to_remove_crossings = []
    edge_edge_crossings = []

    all_pos = nx.get_node_attributes(G, "pos")
    all_width = nx.get_node_attributes(G, "width")
    all_heigth = nx.get_node_attributes(G, "height")

    for cr in crossings_edges:
        (e1, e2, intersection_point) = cr
        (v1, v2) = e1
        (v3, v4) = e2

        p1x, p1y = float(all_pos[v1].split(",")[0]), float(all_pos[v1].split(",")[1])
        p2x, p2y = float(all_pos[v2].split(",")[0]), float(all_pos[v2].split(",")[1])
        p3x, p3y = float(all_pos[v3].split(",")[0]), float(all_pos[v3].split(",")[1])
        p4x, p4y = float(all_pos[v4].split(",")[0]), float(all_pos[v4].split(",")[1])


        w_1, h_1 = float(all_width[v1]), float(all_heigth[v1])
        w_2, h_2 = float(all_width[v2]), float(all_heigth[v2])
        w_3, h_3 = float(all_width[v3]), float(all_heigth[v3])
        w_4, h_4 = float(all_width[v4]), float(all_heigth[v4])

        (ax, ay, bx, by, cx, cy, dx, dy, px, py) = get_rectangle(p1x, p1y, w_1*label_factor_magnifier, h_1*label_factor_magnifier)
        in_shape = check_point_in_rectangle(ax, ay, bx, by, cx, cy, dx, dy, px, py)
        if in_shape:
            to_remove_crossings.append(cr)
            continue

        (ax, ay, bx, by, cx, cy, dx, dy, px, py) = get_rectangle(p2x, p2y, w_2*label_factor_magnifier, h_2*label_factor_magnifier)
        in_shape = check_point_in_rectangle(ax, ay, bx, by, cx, cy, dx, dy, px, py)

        if in_shape:
            to_remove_crossings.append(cr)
            continue

        (ax, ay, bx, by, cx, cy, dx, dy, px, py) = get_rectangle(p3x, p3y, w_3*label_factor_magnifier, h_3*label_factor_magnifier)
        in_shape = check_point_in_rectangle(ax, ay, bx, by, cx, cy, dx, dy, px, py)
        if in_shape:
            to_remove_crossings.append(cr)
            continue

        (ax, ay, bx, by, cx, cy, dx, dy, px, py) = get_rectangle(p4x, p4y, w_4*label_factor_magnifier, h_4*label_factor_magnifier)
        in_shape = check_point_in_rectangle(ax, ay, bx, by, cx, cy, dx, dy, px, py)
        if in_shape:
            to_remove_crossings.append(cr)
            continue

        edge_edge_crossings.append(cr)


    print("removed crossings: " + str(len(to_remove_crossings)))

    return edge_edge_crossings


def triangle_area(x1, y1, x2, y2, x3, y3):
    '''
    A utility function to calculate
    area of triangle formed by (x1, y1),
    (x2, y2) and (x3, y3)
    '''

    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)

# A function to check whether point
# P(x, y) lies inside the rectangle
# formed by A(x1, y1), B(x2, y2),
# C(x3, y3) and D(x4, y4)
def check_point_in_rectangle(x1, y1, x2, y2, x3,
          y3, x4, y4, x, y):
    '''
    A function to check whether point
    P(x, y) lies inside the rectangle
    formed by A(x1, y1), B(x2, y2),
    C(x3, y3) and D(x4, y4)
    '''

    # Calculate area of rectangle ABCD
    A = (triangle_area(x1, y1, x2, y2, x3, y3) +
         triangle_area(x1, y1, x4, y4, x3, y3))

    # Calculate area of triangle PAB
    A1 = triangle_area(x, y, x1, y1, x2, y2)

    # Calculate area of triangle PBC
    A2 = triangle_area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PCD
    A3 = triangle_area(x, y, x3, y3, x4, y4)

    # Calculate area of triangle PAD
    A4 = triangle_area(x, y, x1, y1, x4, y4);

    # Check if sum of A1, A2, A3
    # and A4 is same as A
    return (A == A1 + A2 + A3 + A4)


def count_crossings(G, edges_to_compare=None, stop_when_found=False, ignore_label_edge_cr=False):
    '''
    Counts the crossings of the given graph <tt>G<\tt>. The crossing count can
    be executed only on the given list of edges of <tt>G<\tt> passed as input in
    <tt>edges_to_compare<\tt>.
    Also the execution can stop as soon as a crossing is found if the boolean value
    <tt>stop_when_found<\tt> is set to True.
    If the vertices have labels with given height and width the crossings that occur
    below the labels can be ignored if the boolean value <tt>ignore_label_edge_cr<\tt>
    is set to True.
    Return a list of crossings where each element has the crossing edge and the intersection point.
    '''


    count = 0

    crossings_edges = []

    all_pos = nx.get_node_attributes(G, "pos")

    edge_list = list(nx.edges(G))
    edge_set_1 = list(nx.edges(G))

    if edges_to_compare is not None:
        edge_set_1 = list(edges_to_compare)


    for c1 in range(0, len(edge_set_1)):

        edge1 = edge_set_1[c1]
        (s1,t1) = edge1

        ppos1 = all_pos[s1]
        qpos1 = all_pos[t1]

        p1x, p1y = float(ppos1.split(",")[0]), float(ppos1.split(",")[1])
        q1x, q1y = float(qpos1.split(",")[0]), float(qpos1.split(",")[1])

        j_start=c1+1
        if edges_to_compare is not None:
            j_start=0

        for c2 in range(j_start, len(edge_list)):
            edge2 = edge_list[c2]
            (s2,t2) = edge2

            ppos2 = all_pos[s2]
            qpos2 = all_pos[t2]

            p2x, p2y = float(ppos2.split(",")[0]), float(ppos2.split(",")[1])
            q2x, q2y = float(qpos2.split(",")[0]), float(qpos2.split(",")[1])

            (intersection_exists, intersection_point, type) = doIntersect(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)

            if(intersection_exists):
                (intersection_x, intersection_y) = intersection_point
                crossing_angle = getAngleLineSegDegree(p1x, p1y, p2x, p2y,intersection_x, intersection_y)
                if type == 'C':
                    crossing_angle = 0
                crossings_edges.append((edge1, edge2, intersection_point, crossing_angle))
                count = count + 1

                if stop_when_found:
                    return crossings_edges

    if ignore_label_edge_cr:
        crossings_edges = remove_label_edge_crossings(G, crossings_edges)

    return crossings_edges