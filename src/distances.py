
def norm(vec: list[float],*args,**kwargs)->float:
    sum_tot=0
    for el in vec:
        sum_tot+=el*el
    return sum_tot**0.5

def clip(a:list[float]|float,c_min:float,c_max:float):
    flag=False
    if not isinstance(a, list):
        flag=True
        a=[a,]
    d=[min(max(a[i],c_max),c_min) for i in range(len(a))]
    if flag:
        d=d[0]
    return d

def sign(a:float):
    if a==0:
        return 0
    return abs(a)/a

def dot(a:list[float], b:list[float]):
    print(a)
    print(b)
    if len(a) != len(b):
        raise ValueError(f"Lists must be of the same length, got {len(a), len(b)}")
    c = sum(x * y for x, y in zip(a, b))
    return c

def vec_sub(a:list[float],b:list[float])->list[float]:
    if len(a)!=len(b):
        raise ValueError(f"Lists must be of the same length, got {len(a), len(b)}")
    c=[a[i]-b[i] for i in range(len(a))]
    return c

def vec_add(a:list[float],b:list[float])->list[float]:
    if len(a)!=len(b):
        raise ValueError(f"Lists must be of the same length, got {len(a), len(b)}")
    c=[a[i]+b[i] for i in range(len(a))]
    return c

def vec_abs(a:list[float])->list[float]:
    c=[abs(a[i]) for i in range(len(a))]
    return c

def vec_sqrt(a:list[float])->list[float]:
    c=[a[i]**0.5 for i in range(len(a))]
    return c

def vec_add_c(a:list[float],c:float)->list[float]:
    d=[a[i]+c for i in range(len(a))]
    return d

def vec_sub_c(a:list[float],c:float)->list[float]:
    d=[a[i]-c for i in range(len(a))]
    return d

def vec_mult_c(a:list[float],c:float)->list[float]:
    #print(c)
    d=[a[i]*c for i in range(len(a))]
    return d

def vec_maximum(a:list[float],c:float)->list[float]:
    d=[max(a[i],c) for i in range(len(a))]
    return d

def distance_from_sphere(point:list[float],radius:float) -> float:
    return norm(point)-radius

def distance_from_box(point:list[float],half_sides:list[float]) -> float:
    q= vec_sub(vec_abs(point),half_sides)
    return norm(vec_maximum(q,0)) + min((max(q),0))

def distance_from_frame_box(point:list[float],half_sides:list[float], thickness: float) -> float:
    point=vec_sub(vec_abs(point),half_sides)
    q= vec_sub_c(vec_abs(vec_add_c(point,thickness)),thickness)
    return min((norm(vec_maximum([point[0],q[1],q[2]],0.0))+min(max(point[0],q[1],q[2]),0.0),norm(vec_maximum([q[0],point[1],q[2]],0.0))+min(max(q[0],point[1],q[2]),0.0),norm(vec_maximum([q[0],q[1],point[2]],0.0))+min(max(q[0],q[1],point[2]),0.0)))

def distance_from_round_box(point:list[float],half_sides:list[float], rounding: float) -> float:
    q= vec_add_c(vec_sub(vec_abs(point),half_sides),rounding)
    return norm(vec_maximum(q,0)) + min((max(q),0))-rounding

def distance_from_torus(point:list[float],radi:list[float]) -> float:
    q=[norm([point[0],point[2]])-radi[0],point[1]]
    return norm(q)-radi[1]

def distance_from_cylinder(point:list[float],radius:float, height: float):
    d=vec_sub(vec_abs([norm([point[0],point[2]]),point[1]]),[radius,height])
    return min(max(d),0)+norm(vec_maximum(d,0))

def distance_from_cone(point:list[float],c:list[float], height: float):
    q=vec_mult_c([c[0]/c[1],-1],height)
    w=[norm([point[0],point[2]]),point[1]]
    a= vec_sub(w,vec_mult_c(q,clip(dot(w,q)/dot(q,q),0,1)))
    b=vec_sub(w,vec_mult_c([clip(w[0]/q[0],0,1),1],q))
    k=sign(q[1])
    d=min(dot(a,a),dot(b,b))
    s=max((k*(w[0]*q[1]-w[1]*q[0]),k*(w[1]-q[1])))
    return vec_mult_c(vec_sqrt(d),sign(s))

def distance_from_plane():
    pass

def distance_from_prism():
    pass

def distance_from_capsule():
    pass