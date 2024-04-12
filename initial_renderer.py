# import the pygame module, so you can use it
import pygame
from utils import *
 
# define a main function
def main():
    # initialize the pygame module
    pygame.init()
     
    # create a surface on screen that has the size of 240 x 180
    screen = pygame.display.set_mode((600,600))
     
    # define a variable to control the main loop
    running = True
     
    # main loop
    while running:
        shapes=[]
        screen.fill((255,255,255))
        eye=pygame.draw.circle(screen,(0,0,0),(100,100),5)
        shapes.append(pygame.draw.rect(screen,(0,255,0),(200,100,40,40)))
        shapes.append(pygame.draw.rect(screen,(0,255,0),(230,150,49,49)))
        #shapes.append(pygame.draw.circle(screen,(255,0,0),(500,500),30))
        calculating=True
        p=eye.center
        while calculating:
            d=float("inf")
            for el in shapes:
                d=min(d,signed_dist_to_square(p,el.center,el.size))
                m=pygame.mouse.get_pos()
                if d<=1 or d>100:
                    calculating=False
                    break
            pygame.draw.circle(screen,(255,0,0),p,d,2)
            direction = normalize((m[0] - p[0], m[1] - p[1]))
            p = (p[0] + d * direction[0], p[1] + d * direction[1])

        pygame.draw.line(screen,(0,0,0),eye.center,pygame.mouse.get_pos())
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
        pygame.display.flip()
    print(shapes)
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()