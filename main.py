import pygame 
from tensorflow.keras.models import load_model

pygame.init()
plot = pygame.display.set_mode((200,200))

model = load_model('handwritten.h5')

number = []
run = True 
start_trav = False
while run :
    plot.fill((0,0,0))
                
    
    if start_trav:
        position = pygame.mouse.get_pos()
        number.append(position)
            
    pos = pygame.mouse.get_pos()

    pygame.draw.circle(plot,(250,250,250),pos,5.0)

    for i in number:
        pygame.draw.circle(plot,(250,250,250),i,5.0)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            start_trav = True
        if event.type == pygame.MOUSEBUTTONUP:
            start_trav = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                image_name = 'digit.jpg'

                pygame.image.save(plot,image_name)
                # pygame.transform.scale(plot,(28,28))
                run = False

                

            

    pygame.display.update()

from tensorflow.keras.preprocessing import image
from numpy import argmax
import tensorflow as tf
from matplotlib import pyplot as plt 
img = image.load_img(image_name, target_size=(28,28))
# img = image.img_to_array(img)/255.0

converted = tf.image.rgb_to_grayscale(img)
converted = image.img_to_array(converted)/255.0
converted = converted.reshape(1,28,28,1)
p = model.predict([converted])
print(argmax(p))
# print(converted.shape)

