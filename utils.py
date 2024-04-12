import numpy as np
def length_2d(v: tuple[int,int])-> float:
    v=np.array(v)
    return(v[0]*v[0]+v[1]*v[1])**0.5

def signed_dist_to_circle(point,center,radius):
    return length_2d(center-point)-radius

def signed_dist_to_square(point: tuple[int,int],center: tuple[int,int],size: tuple[int,int]):
    point=np.array(point)
    center=np.array(center)
    size=np.array(size)/2
    offset=abs(point-center)-size
    unsigned_dist=length_2d((max(offset[0],0),max(offset[1],0)))
    dist_inside_square=max(min(offset[0],0),min(offset[1],0))
    return unsigned_dist+dist_inside_square

def normalize(vector):
    length = length_2d(vector)
    return (vector[0] / length, vector[1] / length)
