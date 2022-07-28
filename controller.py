
from pyfirmata import Arduino, SERVO


port = 'COM6'

pin = 10
board = Arduino(port)

board.digital[pin].mode = SERVO


def rotatesrvo(pin, angle):
    board.digital[pin].write(angle)


def doorautomate(detectface):
    if detectface == "without mask":
        #print("no mask")
        rotatesrvo(pin, 20)
    elif detectface == "mask":
        #print("mask")
        rotatesrvo(pin, 180)
    else:
        rotatesrvo(pin, 20)
