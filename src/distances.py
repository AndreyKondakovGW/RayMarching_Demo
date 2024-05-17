import numpy as np
import numpy.typing as npt

def distance_from_sphere(point:npt.ArrayLike,radius:float) -> float:
    return np.linalg.norm(point)-radius

def distance_from_box(point:npt.ArrayLike,half_sides:npt.ArrayLike) -> float:
    q= np.abs(point)-half_sides
    return np.linalg.norm(np.maximum(q,0)) + np.min((np.max(q),0))

def distance_from_frame_box(point:npt.ArrayLike,half_sides:npt.ArrayLike, thickness: float) -> float:
    point=np.abs(point)-half_sides
    q= np.abs(np.add(point,thickness))-thickness
    return np.min((np.linalg.norm(np.maximum([point[0],q[1],q[2]],0.0))+min(max(point[0],q[1],q[2]),0.0),np.linalg.norm(np.maximum([q[0],point[1],q[2]],0.0))+min(max(q[0],point[1],q[2]),0.0),np.linalg.norm(np.maximum([q[0],q[1],point[2]],0.0))+min(max(q[0],q[1],point[2]),0.0)))

def distance_from_round_box(point:npt.ArrayLike,half_sides:npt.ArrayLike, rounding: float) -> float:
    q= np.abs(point)-half_sides +rounding
    return np.linalg.norm(np.maximum(q,0)) + np.min((np.max(q),0))-rounding

def distance_from_torus(point:npt.ArrayLike,radi:npt.ArrayLike) -> float:
    q=np.array([np.linalg.norm([point[0],point[2]])-radi[0],point[1]])
    return np.linalg.norm(q)-radi[1]

def distance_from_cylinder(point:npt.ArrayLike,radius:float, height: float):
    d=np.abs(np.array([np.linalg.norm([point[0],point[2]]),point[1]]))-np.array([radius,height])
    return min(max(d),0)+np.linalg.norm(np.maximum(d,0))

def distance_from_cone(point:npt.ArrayLike,c:npt.ArrayLike, height: float):
    q=height*np.array([c[0]/c[1],-1])
    w=np.array([np.linalg.norm([point[0],point[2]]),point[1]])
    a= w-q*np.clip(np.dot(w,q)/np.dot(q,q),0,1)#is a a vec2?
    b=w-q*np.array([np.clip(w[0]/q[0],0,1),1])
    k=np.sign(q[1])
    d=min(np.dot(a,a),np.dot(b,b))
    s=np.max((k*(w[0]*q[1]-w[1]*q[0]),k*(w[1]-q[1])))
    return np.sqrt(d)*np.sign(s)

def distance_from_plane():
    pass

def distance_from_prism():
    pass

def distance_from_capsule():
    pass