##https://pub.towardsai.net/monte-carlo-simulation-an-in-depth-tutorial-with-python-bcf6eb7856c8

import random 
import matplotlib.pyplot as plt
import turtle
import math

myPen = turtle.Turtle()
myPen.hideturtle()
myPen.speed(0)

myPen.up()
myPen.setposition(-100, -100)
myPen.down()
myPen.fd(200)
myPen.left(90)
myPen.fd(200)

myPen.left(90)
myPen.fd(200)
myPen.left(90)
myPen.fd(200)
myPen.left(90)

myPen.up()
myPen.setposition(0, -100)
myPen.down()
myPen.circle(100)

in_circle = 0
out_circle = 0
rest_area = []

R = 100 # 원의 반지름
real_area = (4 - math.pi) * (R ** 2)    # 실제 파란 부분의 넓이

for i in range(5):
    for j in range(1000):
        x = random.randrange(-R, R)
        y = random.randrange(-R, R)

        if x ** 2 + y ** 2 > R ** 2:
            myPen.color("black")
            myPen.up()
            myPen.goto(x, y)
            myPen.down()
            myPen.dot()
            out_circle = out_circle + 1

        else:
            myPen.color("red")
            myPen.up()
            myPen.goto(x, y)
            myPen.down()
            myPen.dot()
            in_circle = in_circle + 1

        monte_area = 4.0 * (R ** 2) * out_circle / (in_circle + out_circle)
        rest_area.append(monte_area)

        avg_area_errors = [abs(real_area - temp) for temp in rest_area]

    print(rest_area[-1])

plt.axhline(y = real_area, color = 'g', linestyle = '-')
plt.plot(rest_area)
plt.xlabel("Iterations")
plt.ylabel("Value of Rest Area")
plt.show()

plt.axhline(y = 0.0, color = 'g', linestyle = '-')
plt.plot(avg_area_errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()
