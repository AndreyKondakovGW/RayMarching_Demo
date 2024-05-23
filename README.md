# RayMarching_Demo
Python implementation of Raymarching rendering methods. Made for Scientific python course. This project fully writen on python with use of Numba and Cuda technology for parallelization on GPU. Project also use pygame for realtime visualization and matplotlib for gif creation.


<h2> Some digrams with project code organization </h2>

<image
  src="./diagrams/class.svg"
  alt=""
  caption="Class diagram">
  Class diagram

<image
  src="./diagrams/usecase.svg"
  alt=""
  caption="Usecase diagram">  
  Usecase diagram

<image
  src="./diagrams/sequence.svg"
  alt=""
  caption="Sequence diagram">  
  Sequence diagram

<h2>Examples of working rendering</h2>
<image
  src="./examples/animation_1.gif"
  alt="Example1"
  caption="">


<h2> Requirments </h2>
To make project work you need have GPU with cuda instaled
You also need install all pakages with:

```
pip install -r ./requiremts
```
And then you can start pygame example with 
```
python ./main_pygame.py
```
Or create gif with
```
python ./main_gif.py
```

<h2> Sources </h2>

1. http://osgl.ethz.ch/training/Story_Raymarcher_in_Python_I.pdf
2. https://michaelwalczyk.com/blog-ray-marching.html
3. https://michaelwalczyk.com/blog-ray-marching.html
4. https://iquilezles.org/articles/distfunctions/
5. https://www.shadertoy.com/view/7l2cWW
6. https://iquilezles.org/articles/smin/