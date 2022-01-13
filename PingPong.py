
"""
Sanjeev Das
  dassanjeev08.2000@gmail.com
A simple same using open CV and pygame

using hand detection 
  
"""

import pygame 
from pygame import display ,draw , event , font
import sys
import cv2
import handDetector as hd
import numpy as np

pygame.init()
cap = cv2.VideoCapture(0)
detector = hd.HandDetector(HandNo=1)



screen = display.set_mode((766,500))
display.set_caption("PingPong")
def drawcircle(x,y,r):
  draw.circle(screen,(0,0,0),(x,y),r+2)
  draw.circle(screen,(255,255,255),(x,y),r)

def drawrect(x,position,color):
  draw.rect(screen,color,(x-100,position,200,20))
  draw.rect(screen,(255,255,255),(x-98,position+2,196,16))

def checkhit(x,y,playerPosX,changeY,color,playerScore):
  if y > 440 and (playerPosX-100<x<playerPosX+100):
    changeY = -changeY
    playerScore = playerScore+1
    color = (0,255,0)
  else:
    color = (0,0,0)
  return x,y,playerPosX,changeY,color,playerScore
  

def moveball(x,y,changeX,changeY,compScore):
  if y != 520:
    if x > 490 or x < 10 :
      changeX = -changeX
    if y < 40 :
      changeY = -changeY
      compScore = compScore+1
    if y > 490:      
      y = 520
      changeY = 0
      changeX = 0
    
    x = x + changeX
    y = y + changeY
  return x,y,changeX,changeY ,compScore 

myfont = font.Font("freesansbold.ttf",20)

x = 250
y= 250
changeX = 0.5*5
changeY = -1*5
playerPosX = 100
compScore = 0
playerScore = 0
color =(0,0,0)
color1 =(0,255,0)


def score(posX,posY,text,score,color):
  myfont = font.Font('C:\Windows\Fonts\ARLRDBD.TTF',20)
  myrender = myfont.render(text+str(score),True,color)
  screen.blit(myrender,(posX,posY))


def gameOver(color):
  myfont = font.Font('C:\Windows\Fonts\ARLRDBD.TTF',50)
  myrender = myfont.render("GAME OVER",True,(0,0,0))
  screen.blit(myrender,(90,200))
  color =(255,0,0)
  return color




while True:
  success , img = cap.read()
  img = detector.process(img, draw=False)
  lmList = detector.fingerdetector(img)
  #cv2.rectangle(img,(70,0),(330,479),(255,0,0),2)
  if lmList:
    playerPosX = np.interp(lmList[8][1],[70,330],[400,100])
    cv2.circle(img,(lmList[8][1],lmList[8][2]),5,(0,255,0),cv2.FILLED)
      
  allevents = event.get()
  for myevent in allevents:
    if myevent.type == pygame.QUIT:
      sys.exit()
    
  screen.fill((193, 225, 193))
  drawcircle(x,y,10)
  x,y,playerPosX,changeY,color,playerScore = checkhit(x,y,playerPosX,changeY,color,playerScore)
  x,y,changeX,changeY,compScore = moveball(x,y,changeX,changeY,compScore)
  drawrect(x,10,(0,0,0))
  if y == 520:
    color1 = gameOver(color)
    playerPosX = 250
  drawrect(playerPosX,450,color)
  draw.rect(screen,(0,0,0),(500,0,300,500))
  score(505,20,"Computer : ",compScore,(255,255,255))
  score(505,420,"Player : ",playerScore,color1)
  
  img = cv2.resize(img, (256, 192))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
  #img = cv2.flip(img,1)
  #cv2.imshow("PingPong",img)
  img = pygame.surfarray.make_surface(img)
  screen.blit(img,(505,154))
  display.flip()
  cv2.waitKey(1)
